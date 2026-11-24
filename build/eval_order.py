"""
Purpose:          Word-order probe.  "The dog bit the man" and "The man bit
                  the dog" score 0.97.  This script asks, layer by layer:
                  (1) how far does the pooled sentence vector move when the
                      words are scrambled?            (SIB-200, 4 languages)
                  (2) can a linear probe still read "natural vs scrambled"
                      from that vector?                (same data)
                  (3) can the similarity score tell a true paraphrase from a
                      re-ordered sentence that means something else, and can
                      a probe on the same vectors?     (PAWS-X: en, es, zh)
                  The probes are shipped to the page so it can run them live.
Verification:     eval/order.json; probe settings chosen on dev splits only;
                  a shuffled-label control is reported next to every probe

Run:  .venv/bin/python build/eval_order.py      (about 3 min on 16 CPU threads; downloads PAWS-X, 3 languages)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")          # scikit-learn's solver otherwise takes every core on a shared machine

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import CACHE, EVAL, LANG_NAMES, LANGS, SEED, WEB, load_model, load_sib, set_seed, word_groups, write_json

KINDS = ("shuffle", "swap", "reverse")
PAWS_LANGS = ("en", "es", "zh")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def units(tokens):
    """Movable units of a sentence: one per display word, as token positions.  MUST mirror web/js/order.js.

    Every non-special token belongs to exactly one unit: a stray '▁' piece travels with the word after it.
    Returns (units, movable) where movable[i] is False for punctuation (it keeps its slot).
    """
    groups = word_groups(tokens)
    owner = {}
    for gi, g in enumerate(groups):
        for p in g["tokens"]:
            owner[p] = gi
    out = [[] for _ in groups]
    pending = []
    for p in range(1, len(tokens) - 1):
        if p in owner:
            out[owner[p]].extend(pending + [p]); pending = []
        else:
            pending.append(p)
    if pending and out:
        out[-1].extend(pending)
    return out, [not g["punct"] for g in groups]


def reorder(ids, us, movable, perm):
    """Token ids with the movable units put in the order `perm` (a permutation of range(n_movable))."""
    slots = [i for i, m in enumerate(movable) if m]
    order = list(range(len(us)))
    for slot, src in zip(slots, perm):
        order[slot] = slots[src]
    return [ids[0]] + [ids[p] for u in order for p in us[u]] + [ids[-1]]


def permutation(kind, n, rng):
    if kind == "reverse":
        return list(range(n))[::-1]
    if kind == "swap":
        i, j = rng.choice(n, 2, replace=False)
        p = list(range(n)); p[i], p[j] = p[j], p[i]
        return p
    while True:
        p = list(rng.permutation(n))
        if p != list(range(n)):
            return p


@torch.no_grad()
def layer_pools(model, pad, id_lists, batch: int = 64):
    """Unit-normalized mean-pooled vectors at every layer: [N, 13, 384]."""
    out = []
    for i in range(0, len(id_lists), batch):
        chunk = id_lists[i:i + batch]
        T = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long); mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for r, c in enumerate(chunk):
            ids[r, :len(c)] = torch.tensor(c); mask[r, :len(c)] = 1
        hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        m = mask.unsqueeze(-1).float()
        pooled = torch.stack([(h * m).sum(1) / m.sum(1) for h in hs], 1)
        out.append(torch.nn.functional.normalize(pooled, dim=-1))
    return torch.cat(out).numpy()


def fit_probe(xtr, ytr, xdv, ydv):
    """Standardized logistic regression; C chosen on dev.  Returns (model-on-raw-features weights, bias, C)."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000).fit((xtr - mu) / sd, ytr)
        acc = clf.score((xdv - mu) / sd, ydv)
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    _, C, clf = best
    w = clf.coef_[0] / sd
    return w, float(clf.intercept_[0] - (w * mu).sum()), C


def main() -> None:
    set_seed(); torch.set_num_threads(16)
    rng = np.random.default_rng(SEED)
    tok, model = load_model()
    pad = tok.pad_token_id

    # ------------------------------------------------------------------ 1 + 2: scrambled SIB-200 sentences
    sib = {}
    for split in ("train", "dev", "test"):
        rows = []
        data = load_sib(split)
        for li, lang in enumerate(LANGS):
            for r in data[lang]:
                ids = tok(r["text"], truncation=True, max_length=128)["input_ids"]
                us, mv = units(tok.convert_ids_to_tokens(ids))
                n = sum(mv)
                if n < 4:
                    continue
                rows.append({"lang": li, "ids": ids, **{k: reorder(ids, us, mv, permutation(k, n, rng)) for k in KINDS}})
        P = {k: layer_pools(model, pad, [r[k] for r in rows]) for k in ("ids", *KINDS)}
        sib[split] = {"lang": np.array([r["lang"] for r in rows]), **P}
        print(split, len(rows), "sentences", flush=True)

    te = sib["test"]
    L = te["ids"].shape[1]
    cos = {k: (te["ids"] * te[k]).sum(-1) for k in KINDS}                          # [N, 13]
    move = {k: {"all": cos[k].mean(0).round(4).tolist(),
                **{lang: cos[k][te["lang"] == li].mean(0).round(4).tolist() for li, lang in enumerate(LANGS)}} for k in KINDS}
    q = lambda a: np.percentile(a, [10, 50, 90], axis=0).round(4).tolist()
    band = {k: q(cos[k]) for k in KINDS}

    probes, order_probe = [], {"accuracy": [], "control_accuracy": [], "by_language": {l: [] for l in LANGS}, "swap_detect": [], "C": []}
    for l in range(L):
        def xy(split, kind="shuffle"):
            s = sib[split]
            return np.concatenate([s["ids"][:, l], s[kind][:, l]]), np.concatenate([np.ones(len(s["ids"])), np.zeros(len(s["ids"]))])
        xtr, ytr = xy("train"); xdv, ydv = xy("dev"); xte, yte = xy("test")
        w, b, C = fit_probe(xtr, ytr, xdv, ydv)
        pred = (xte @ w + b) > 0
        order_probe["accuracy"].append(float((pred == yte).mean()))
        lang2 = np.concatenate([te["lang"], te["lang"]])
        for li, lang in enumerate(LANGS):
            order_probe["by_language"][lang].append(float((pred == yte)[lang2 == li].mean()))
        xs, ys = xy("test", "swap")                                                 # trained on full shuffles, tested on one swapped pair
        order_probe["swap_detect"].append(float((((xs @ w + b) > 0) == ys).mean()))
        # control: the same probe trained on shuffled labels cannot beat chance unless it memorizes
        wc, bc, _ = fit_probe(xtr, rng.permutation(ytr), xdv, ydv)
        order_probe["control_accuracy"].append(float((((xte @ wc + bc) > 0) == yte).mean()))
        order_probe["C"].append(C)
        probes.append({"w": np.round(w, 4).tolist(), "b": round(b, 4)})
        print(f"layer {l:2d}  cos shuffle {move['shuffle']['all'][l]:.3f} swap {move['swap']['all'][l]:.3f}  "
              f"probe acc {order_probe['accuracy'][-1]:.3f} (control {order_probe['control_accuracy'][-1]:.3f}, swap {order_probe['swap_detect'][-1]:.3f})", flush=True)

    # ------------------------------------------------------------------ 3: PAWS-X, re-ordered sentences that change meaning
    from datasets import load_dataset
    paws = {}
    for lang in PAWS_LANGS:
        for split in ("validation", "test"):
            ds = load_dataset("google-research-datasets/paws-x", lang, split=split, cache_dir=str(CACHE / "hf"))
            keep = [r for r in ds if r["sentence1"].strip() and r["sentence2"].strip()]
            enc = lambda key: [tok(r[key], truncation=True, max_length=128)["input_ids"] for r in keep]
            paws[lang, split] = {"a": layer_pools(model, pad, enc("sentence1")), "b": layer_pools(model, pad, enc("sentence2")),
                                 "y": np.array([r["label"] for r in keep])}
            print("paws-x", lang, split, len(keep), "pairs, paraphrase share", round(float(paws[lang, split]["y"].mean()), 3), flush=True)
    cat = lambda split, key: np.concatenate([paws[lang, split][key] for lang in PAWS_LANGS])
    feats = lambda a, b: np.abs(a - b)
    paws_out = {"n_test": {lang: int(len(paws[lang, "test"]["y"])) for lang in PAWS_LANGS}, "cosine_auc": {"all": []}, "probe_auc": {"all": []},
                "control_auc": [], "cos_paraphrase": [], "cos_not_paraphrase": []}
    for lang in PAWS_LANGS:
        paws_out["cosine_auc"][lang] = []; paws_out["probe_auc"][lang] = []
    pair_probe = None
    for l in range(L):
        a, b, y = cat("test", "a")[:, l], cat("test", "b")[:, l], cat("test", "y")
        c = (a * b).sum(-1)
        paws_out["cosine_auc"]["all"].append(float(roc_auc_score(y, c)))
        paws_out["cos_paraphrase"].append(float(c[y == 1].mean())); paws_out["cos_not_paraphrase"].append(float(c[y == 0].mean()))
        # probe on |a - b|: trained on the first 70% of the PAWS-X dev set, C chosen on the rest, scored on test
        xa, xb, yd = cat("validation", "a")[:, l], cat("validation", "b")[:, l], cat("validation", "y")
        idx = rng.permutation(len(yd)); cut = int(0.7 * len(yd)); tr, dv = idx[:cut], idx[cut:]
        fd = feats(xa, xb)
        w, b0, C = fit_probe(fd[tr], yd[tr], fd[dv], yd[dv])
        s = feats(a, b) @ w + b0
        paws_out["probe_auc"]["all"].append(float(roc_auc_score(y, s)))
        wc, bc, _ = fit_probe(fd[tr], rng.permutation(yd[tr]), fd[dv], yd[dv])
        paws_out["control_auc"].append(float(roc_auc_score(y, feats(a, b) @ wc + bc)))
        off = 0
        for lang in PAWS_LANGS:
            n = len(paws[lang, "test"]["y"]); sl = slice(off, off + n); off += n
            paws_out["cosine_auc"][lang].append(float(roc_auc_score(y[sl], c[sl])))
            paws_out["probe_auc"][lang].append(float(roc_auc_score(y[sl], s[sl])))
        if l == L - 1:
            pair_probe = {"w": np.round(w, 4).tolist(), "b": round(b0, 4), "C": C, "features": "|a - b| on unit vectors, last layer"}
        print(f"layer {l:2d}  PAWS-X cosine AUC {paws_out['cosine_auc']['all'][-1]:.3f}  probe AUC {paws_out['probe_auc']['all'][-1]:.3f} "
              f"(control {paws_out['control_auc'][-1]:.3f})  cos para {paws_out['cos_paraphrase'][-1]:.3f} / not {paws_out['cos_not_paraphrase'][-1]:.3f}", flush=True)

    # a richer readout at the last layer: |a - b| and a * b together (768 features), same split rule
    rich = lambda a, b: np.concatenate([np.abs(a - b), a * b], 1)
    xa, xb, yd = cat("validation", "a")[:, L - 1], cat("validation", "b")[:, L - 1], cat("validation", "y")
    idx = rng.permutation(len(yd)); cut = int(0.7 * len(yd)); tr, dv = idx[:cut], idx[cut:]
    fd = rich(xa, xb)
    w, b0, _ = fit_probe(fd[tr], yd[tr], fd[dv], yd[dv])
    paws_out["rich_probe_auc_last_layer"] = float(roc_auc_score(cat("test", "y"), rich(cat("test", "a")[:, L - 1], cat("test", "b")[:, L - 1]) @ w + b0))
    print("PAWS-X richer probe (|a-b| and a*b), last layer: AUC", round(paws_out["rich_probe_auc_last_layer"], 3), flush=True)

    report = {
        "seed": SEED, "layers": L,
        "scramble": {"dataset": "SIB-200 (en, es, hi, zh); probes fitted on train, C on dev, everything reported on test",
                     "n_test": int(len(te["ids"])), "n_train": int(len(sib["train"]["ids"])),
                     "kinds": {"shuffle": "all words in random order", "swap": "two words exchanged", "reverse": "words in reverse order"},
                     "cos_by_layer": move, "cos_band_p10_p50_p90": band, "probe": order_probe,
                     "languages": {k: LANG_NAMES[k] for k in LANGS}},
        "paws": {"dataset": "PAWS-X test (Yang et al. 2019): sentence pairs with high word overlap; label 1 = paraphrase", **paws_out,
                 "paraphrase_share": float(cat("test", "y").mean()), "pair_probe_trained_on": "70% of PAWS-X dev (en+es+zh), C on the other 30%"},
    }
    write_json(EVAL / "order.json", report)
    write_json(WEB / "data" / "order.json", report | {"probes": probes, "pair_probe": pair_probe})
    print("wrote eval/order.json and web/data/order.json")


if __name__ == "__main__":
    main()
