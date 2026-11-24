"""
Purpose:          Faithfulness benchmark.  For translation pairs in four
                  languages, rank the words of one sentence with each
                  attribution method, remove words in that order, and measure
                  how fast the model's similarity score falls (deletion test,
                  book Unit 2 ch. 9).  v2 adds a second removal operation
                  (mask instead of delete) so occlusion is not graded only by
                  its own operation, Grad-CAM at every layer, gradient x
                  input, integrated gradients on the same path the browser
                  uses, and an optional second model.
Verification:     Asserts closed-form Grad-CAM == autograd Grad-CAM and that
                  the pair decomposition sums to the similarity; results in
                  eval/faithfulness*.json (v0 results kept as history/faithfulness_v0.json)

Run:  .venv/bin/python build/eval_attribution.py                       # shipped model, about 15 min on 48 cores
      .venv/bin/python build/eval_attribution.py --model intfloat/multilingual-e5-small --prefix "query: " --tag e5 --n 102
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from common import EVAL, LANGS, MODEL_ID, SEED, WEB, load_model, load_sib, run_layers, set_seed, word_groups, write_json

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
IG_STEPS = 24
GRADCAM = [f"gradcam_L{l}" for l in range(12, 0, -1)]
METHODS = ["pair_decomposition", "occlusion", "occlusion_mask", "integrated_gradients", "grad_x_input", *GRADCAM,
           "attention_rollout", "token_count", "random", "random_word"]
# graded under the second removal operation too (the methods shown on the page, plus two baselines)
MASK_METHODS = ["pair_decomposition", "occlusion", "occlusion_mask", "integrated_gradients", "gradcam_L12",
                "attention_rollout", "token_count", "random_word"]
# Baselines: "random" = random score per SUB-WORD token, summed per word (so longer words rank higher by chance);
# "token_count" = number of sub-word tokens in the word (pure length heuristic); "random_word" = random per word (true zero).


class Explainer:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.pad, self.mask = tok.pad_token_id, tok.mask_token_id

    @torch.no_grad()
    def sims(self, id_lists, e_other):
        """cosine(sentence, other) for a list of token-id lists (padded batch)."""
        out = []
        for i in range(0, len(id_lists), 64):
            chunk = id_lists[i:i + 64]
            T = max(len(c) for c in chunk)
            ids = torch.full((len(chunk), T), self.pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for r, c in enumerate(chunk):
                ids[r, :len(c)] = torch.tensor(c); mask[r, :len(c)] = 1
            m = mask.unsqueeze(-1).float()
            e = (self.model(input_ids=ids, attention_mask=mask).last_hidden_state * m).sum(1) / m.sum(1)
            out.append(F.cosine_similarity(e, e_other[None], dim=1))
        return torch.cat(out).numpy()

    def explain(self, ids, e_other, rng):
        """Token-level scores for every gradient / attention / decomposition method.  ids: python list incl. <s> </s>."""
        x = torch.tensor([ids])
        T = len(ids)
        with torch.no_grad():
            attentions = self.model(input_ids=x, output_attentions=True).attentions
            h0 = self.model.embeddings(input_ids=x)
            base_ids = x.clone(); base_ids[0, 1:-1] = self.pad             # all-<pad> baseline, special tokens kept
            hb = self.model.embeddings(input_ids=base_ids)
        # one forward from the layer-0 states with gradients kept at every layer
        hs = run_layers(self.model, h0.clone().requires_grad_(True))
        for h in hs:
            h.retain_grad()
        mean = hs[-1][0].mean(0)
        s = F.cosine_similarity(mean, e_other, dim=0)
        s.backward()
        H = hs[-1][0].detach()
        e_self = mean.detach() / mean.detach().norm()
        s0 = s.item()
        scores = {}

        # (a) exact pair decomposition, row-summed:  sum_t r_t == cosine
        r = (H @ e_other) / (T * mean.detach().norm())
        assert abs(float(r.sum()) - s0) < 1e-4
        scores["pair_decomposition"] = r.numpy()

        # (b) Grad-CAM over tokens at every layer: alpha_k = mean_t dS/dh_tk ; cam_t = ReLU(sum_k alpha_k h_tk)
        for L in range(1, 13):
            g = hs[L].grad[0]
            scores[f"gradcam_L{L}"] = F.relu((hs[L][0].detach() * g.mean(0, keepdim=True)).sum(1)).numpy()
        closed = F.relu(H @ ((e_other - s0 * e_self) / (T * mean.detach().norm())))
        assert torch.allclose(closed, torch.tensor(scores["gradcam_L12"]), atol=1e-5), "closed-form Grad-CAM mismatch"
        scores["grad_x_input"] = (hs[0][0].detach() * hs[0].grad[0]).sum(1).numpy()

        # (c) attention rollout (Abnar & Zuidema 2020), read out for the mean-pooled output
        roll = torch.eye(T)
        for a in attentions:
            a = 0.5 * a[0].mean(0) + 0.5 * torch.eye(T)
            roll = a @ roll
        scores["attention_rollout"] = roll.mean(0).numpy()

        # (d) integrated gradients in layer-0 space from the all-<pad> baseline, midpoint rule.
        #     Same path, steps and baseline as web/js/grad.js.
        alphas = (torch.arange(IG_STEPS).float() + 0.5) / IG_STEPS
        path = (hb + alphas[:, None, None] * (h0 - hb)).requires_grad_(True)
        sp = F.cosine_similarity(run_layers(self.model, path)[-1].mean(1), e_other[None], dim=1)
        grad = torch.autograd.grad(sp.sum(), path)[0].mean(0)
        ig = ((h0 - hb)[0] * grad).sum(1)
        with torch.no_grad():
            sb = float(F.cosine_similarity(run_layers(self.model, hb)[-1][0].mean(0), e_other, dim=0))
        scores["integrated_gradients"] = ig.detach().numpy()
        scores["random"] = rng.random(T)
        self.model.zero_grad(set_to_none=True)
        return s0, scores, {"ig_sum": float(ig.sum()), "ig_target": s0 - sb}

    def word_scores(self, ids, groups, e_other, rng):
        """Word-level scores for every method in METHODS (occlusion needs extra forward passes)."""
        s0, tok_scores, ig_info = self.explain(ids, e_other, rng)
        ws = {m: np.array([tok_scores[m][g["tokens"]].sum() for g in groups]) for m in tok_scores}
        ws["occlusion"] = s0 - self.sims([remove(ids, g["tokens"]) for g in groups], e_other)
        ws["occlusion_mask"] = s0 - self.sims([replace(ids, g["tokens"], self.mask) for g in groups], e_other)
        ws["token_count"] = np.array([len(g["tokens"]) for g in groups], dtype=float) + 1e-6 * rng.random(len(groups))
        ws["random_word"] = rng.random(len(groups))
        return s0, ws, ig_info


def remove(ids, positions):
    drop = set(positions)
    return [t for i, t in enumerate(ids) if i not in drop]


def replace(ids, positions, token):
    out = list(ids)
    for p in positions:
        out[p] = token
    return out


def content_groups(tok, ids, n_prefix: int = 0):
    """Display words of a sentence, without punctuation and without an instruction prefix such as 'query: '."""
    return [g for g in word_groups(tok.convert_ids_to_tokens(ids)) if not g["punct"] and g["tokens"][0] > n_prefix]


def run_shard(job):
    model_id, prefix, ks, threads = job
    torch.set_num_threads(threads)
    tok, model = load_model(model_id=model_id)
    ex = Explainer(tok, model)
    data = load_sib("test")
    n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
    rows = []
    for k in ks:
        rng = np.random.default_rng([SEED, k])
        enc = {l: tok(prefix + data[l][k]["text"], truncation=True, max_length=128)["input_ids"] for l in LANGS}
        with torch.no_grad():
            embs = {l: F.normalize(model(input_ids=torch.tensor([enc[l]])).last_hidden_state[0].mean(0), dim=0) for l in LANGS}
        for other in ("es", "hi", "zh"):
            for explained, partner in (("en", other), (other, "en")):
                ids, e_o = enc[explained], embs[partner]
                groups = content_groups(tok, ids, n_prefix)
                if len(groups) < 4:
                    continue
                s0, ws, ig_info = ex.word_scores(ids, groups, e_o, rng)
                row = {"id": data["en"][k]["id"], "explained": explained, "partner": partner, "n_words": len(groups),
                       "s0": s0, "ig_completeness_err": abs(ig_info["ig_sum"] - ig_info["ig_target"]),
                       "ig_target": ig_info["ig_target"],
                       "occ_delete_vs_mask_spearman": float(spearmanr(ws["occlusion"], ws["occlusion_mask"]).statistic),
                       "rank_corr_vs_occlusion": {m: float(spearmanr(ws[m], ws["occlusion"]).statistic) for m in METHODS if m != "occlusion"},
                       "delete": {"morf": {}, "lerf": {}}, "mask": {"morf": {}, "lerf": {}}}
                # removal curves: most-relevant-first and least-relevant-first, under two removal operations
                variants, index = [], []
                for op, methods in (("delete", METHODS), ("mask", MASK_METHODS)):
                    for m in methods:
                        order = np.argsort(-ws[m], kind="stable")
                        for tag, o in (("morf", order), ("lerf", order[::-1])):
                            for f in FRACS:
                                kdel = max(1, int(round(f * len(groups))))
                                pos = [p for gi in o[:kdel] for p in groups[gi]["tokens"]]
                                variants.append(remove(ids, pos) if op == "delete" else replace(ids, pos, ex.mask))
                                index.append((op, m, tag))
                sims = ex.sims(variants, e_o) / s0
                for (op, m, tag), v in zip(index, sims):
                    row[op][tag].setdefault(m, []).append(float(v))
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=204, help="number of SIB-200 test sentences (per language pair)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prefix", default="", help="instruction prefix the model expects, e.g. 'query: ' for E5")
    ap.add_argument("--tag", default="", help="suffix for the output file; empty = the shipped model")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    set_seed()
    rng = np.random.default_rng(SEED)
    n = min(args.n, len(load_sib("test")["en"]))
    t0 = time.time()
    jobs = [(args.model, args.prefix, list(range(w, n, args.workers)), args.threads) for w in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = [r for part in pool.map(run_shard, jobs) for r in part]
    rows.sort(key=lambda r: (r["id"], r["explained"], r["partner"]))
    print(f"{len(rows)} explanations in {time.time() - t0:.0f}s", flush=True)

    def boot_ci(values, ids):
        """95% interval of the mean by a clustered bootstrap: whole source sentences are resampled, because the same
        sentence (in its four translations) is explained up to six times and those rows are not independent."""
        uniq, inv = np.unique(ids, return_inverse=True)
        sums = np.bincount(inv, weights=values, minlength=len(uniq)); cnts = np.bincount(inv, minlength=len(uniq))
        draws = rng.integers(0, len(uniq), (1000, len(uniq)))
        means = sums[draws].sum(1) / cnts[draws].sum(1)
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    def gaps(sel, op, m):
        return np.array([np.mean(r[op]["lerf"][m]) - np.mean(r[op]["morf"][m]) for r in sel])

    def agg(sel, op, methods):
        out, ids = {}, np.array([r["id"] for r in sel])
        for m in methods:
            morf = np.array([np.mean(r[op]["morf"][m]) for r in sel]); lerf = np.array([np.mean(r[op]["lerf"][m]) for r in sel])
            gap = lerf - morf
            out[m] = {"morf_auc": float(morf.mean()), "lerf_auc": float(lerf.mean()), "gap": float(gap.mean()),
                      "gap_ci95": boot_ci(gap, ids),
                      "morf_curve": np.mean([r[op]["morf"][m] for r in sel], 0).round(4).tolist(),
                      "lerf_curve": np.mean([r[op]["lerf"][m] for r in sel], 0).round(4).tolist()}
            if m != "occlusion":
                out[m]["rank_corr_vs_occlusion"] = float(np.nanmean([r["rank_corr_vs_occlusion"][m] for r in sel]))
        return out

    # paired differences between methods on the same explanations (clustered bootstrap), for the comparisons the text makes
    PAIRS = [("occlusion", "integrated_gradients"), ("occlusion_mask", "integrated_gradients"), ("occlusion", "occlusion_mask"),
             ("integrated_gradients", "attention_rollout"), ("gradcam_L12", "token_count"), ("pair_decomposition", "token_count"),
             ("pair_decomposition", "random_word"), ("attention_rollout", "token_count")]

    def paired(sel, op, methods):
        ids, out = np.array([r["id"] for r in sel]), {}
        for a, b in PAIRS:
            if a in methods and b in methods:
                d = gaps(sel, op, a) - gaps(sel, op, b)
                out[f"{a}__minus__{b}"] = {"diff": float(d.mean()), "ci95": boot_ci(d, ids)}
        return out

    report = {
        "model": args.model, "prefix": args.prefix, "dataset": "Davlan/sib200 test split, translation pairs en<->{es,hi,zh}", "seed": SEED,
        "target": "cosine similarity between the two mean-pooled sentence embeddings",
        "removal": {"delete": "the top-k% words of the explained sentence are deleted (tokens removed)",
                    "mask": "the top-k% words are replaced by <mask> tokens (length and positions kept)"},
        "value": "similarity / original similarity, averaged over k in 10..100%",
        "integrated_gradients": {"space": "layer-0 states (embedding output)", "baseline": "all <pad>, special tokens kept", "steps": IG_STEPS, "rule": "midpoint"},
        "fractions": FRACS, "n_explanations": len(rows), "mean_words": float(np.mean([r["n_words"] for r in rows])),
        "mean_s0": float(np.mean([r["s0"] for r in rows])),
        "all": agg(rows, "delete", METHODS),
        "by_explained_language": {l: agg([r for r in rows if r["explained"] == l], "delete", METHODS) for l in LANGS},
        "mask_all": agg(rows, "mask", MASK_METHODS),
        "mask_by_explained_language": {l: agg([r for r in rows if r["explained"] == l], "mask", MASK_METHODS) for l in LANGS},
        "interval": "95% clustered bootstrap over source sentences (1,000 draws); each sentence contributes up to six explanations",
        "n_sentences": len({r["id"] for r in rows}),
        "paired_differences": {"delete": paired(rows, "delete", METHODS), "mask": paired(rows, "mask", MASK_METHODS)},
        "paired_differences_by_explained_language": {l: {"delete": paired([r for r in rows if r["explained"] == l], "delete", METHODS),
                                                         "mask": paired([r for r in rows if r["explained"] == l], "mask", MASK_METHODS)} for l in LANGS},
        "ig_completeness_abs_err_mean": float(np.mean([r["ig_completeness_err"] for r in rows])),
        "ig_target_abs_mean": float(np.mean([abs(r["ig_target"]) for r in rows])),
        "occlusion_delete_vs_mask_spearman_mean": float(np.nanmean([r["occ_delete_vs_mask_spearman"] for r in rows])),
        "by_language_n": {l: sum(r["explained"] == l for r in rows) for l in LANGS},
        "seconds": round(time.time() - t0),
    }
    name = f"faithfulness_{args.tag}.json" if args.tag else "faithfulness.json"
    write_json(EVAL / name, report)
    write_json(WEB / "data" / (f"eval_{args.tag}.json" if args.tag else "eval.json"), report)
    # per-explanation gaps, so intervals and comparisons can be recomputed without re-running the model
    write_json(EVAL / (f"faithfulness_{args.tag}_rows.json" if args.tag else "faithfulness_rows.json"),
               [{"id": r["id"], "explained": r["explained"], "partner": r["partner"], "n_words": r["n_words"], "s0": round(r["s0"], 5),
                 "gap_delete": {m: round(float(gaps([r], "delete", m)[0]), 5) for m in METHODS},
                 "gap_mask": {m: round(float(gaps([r], "mask", m)[0]), 5) for m in MASK_METHODS}} for r in rows])
    print(f"mean words {report['mean_words']:.1f}, mean s0 {report['mean_s0']:.3f}")
    print(f"{'method':22s} {'gap(delete)':>11s}  ci95            {'gap(mask)':>9s}  rho_vs_occ")
    for m, v in report["all"].items():
        mk = report["mask_all"].get(m)
        print(f"{m:22s} {v['gap']:11.3f}  [{v['gap_ci95'][0]:.3f},{v['gap_ci95'][1]:.3f}]  {(mk['gap'] if mk else float('nan')):9.3f}  {v.get('rank_corr_vs_occlusion', float('nan')):.3f}")
    for op in ("delete", "mask"):
        for k, v in report["paired_differences"][op].items():
            print(f"  {op:6s} {k:50s} {v['diff']:+.3f}  [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
    print("IG completeness |err|:", round(report["ig_completeness_abs_err_mean"], 4), "of", round(report["ig_target_abs_mean"], 3),
          " occlusion delete-vs-mask rho:", round(report["occlusion_delete_vs_mask_spearman_mean"], 3))


if __name__ == "__main__":
    main()
