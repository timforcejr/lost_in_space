"""
Purpose:          Model-randomization sanity check (Adebayo et al. 2018).
                  An explanation that says something about the model must
                  change when the model's weights are replaced by noise.
                  The top 1, 4, 8 and all 12 encoder layers are re-initialized
                  (cascading from the output down) and each method's word
                  ranking is compared with its ranking on the trained model.
Verification:     eval/sanity.json; "word length only" is a built-in control
                  that ignores the model and must stay at 1.0

Run:  .venv/bin/python build/eval_sanity.py [--n 40]      (about 4 min on 16 CPU threads)
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from common import EVAL, LANGS, SEED, WEB, load_model, load_sib, set_seed, write_json
from eval_attribution import Explainer, content_groups

# 2026-11-19: Grad-CAM was removed from the page; its rows are still computed so eval/sanity.json stays reproducible.
METHODS = ["occlusion", "occlusion_mask", "integrated_gradients", "grad_x_input", "pair_decomposition", "gradcam_L12", "gradcam_L6",
           "attention_rollout", "token_count"]
DEPTHS = [1, 4, 8, 12]                                       # how many encoder layers, counted from the top, are re-initialized


def randomized(model, top_k: int, seed: int):
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for layer in list(m.encoder.layer)[len(m.encoder.layer) - top_k:]:
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data.normal_(0.0, m.config.initializer_range); mod.bias.data.zero_()
            elif isinstance(mod, torch.nn.LayerNorm):
                mod.weight.data.fill_(1.0); mod.bias.data.zero_()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="SIB-200 test sentences (6 explanations each)")
    args = ap.parse_args()
    set_seed(); torch.set_num_threads(16)
    tok, model = load_model()
    data = load_sib("test")
    models = {0: model, **{k: randomized(model, k, SEED + k) for k in DEPTHS}}
    corr = {k: {m: [] for m in METHODS} for k in DEPTHS}
    sims = {k: [] for k in models}
    for i in range(args.n):
        enc = {l: tok(data[l][i]["text"], truncation=True, max_length=128)["input_ids"] for l in LANGS}
        scores = {}
        for k, mdl in models.items():
            ex = Explainer(tok, mdl)
            with torch.no_grad():
                embs = {l: F.normalize(mdl(input_ids=torch.tensor([enc[l]])).last_hidden_state[0].mean(0), dim=0) for l in LANGS}
            for other in ("es", "hi", "zh"):
                for explained, partner in (("en", other), (other, "en")):
                    groups = content_groups(tok, enc[explained])
                    if len(groups) < 4:
                        continue
                    s0, ws, _ = ex.word_scores(enc[explained], groups, embs[partner], np.random.default_rng([SEED, i]))
                    scores[k, explained, partner] = ws
                    sims[k].append(s0)
        for (k, explained, partner), ws in scores.items():
            if k == 0:
                continue
            ref = scores[0, explained, partner]
            for m in METHODS:
                corr[k][m].append(spearmanr(ref[m], ws[m]).statistic)     # nan when a ranking is constant (e.g. Grad-CAM all zero)
        if (i + 1) % 10 == 0:
            print(i + 1, "/", args.n, flush=True)

    report = {"seed": SEED, "n_explanations": len(corr[DEPTHS[0]][METHODS[0]]), "depths": DEPTHS,
              "what": "mean Spearman rank correlation between a method's word scores on the trained model and on a model whose top-k "
                      "encoder layers were re-initialized; near 1 = the explanation did not notice, near 0 = it changed completely",
              "mean_similarity": {str(k): float(np.mean(v)) for k, v in sims.items()},
              "rank_corr": {m: [float(np.nanmean(corr[k][m])) for k in DEPTHS] for m in METHODS},
              "undefined_share": {m: [float(np.mean(np.isnan(corr[k][m]))) for k in DEPTHS] for m in METHODS}}
    write_json(EVAL / "sanity.json", report)
    write_json(WEB / "data" / "sanity.json", report)
    print(f"{'method':22s} " + " ".join(f"top{k:>2d}" for k in DEPTHS))
    for m in METHODS:
        print(f"{m:22s} " + " ".join(f"{v:5.2f}" for v in report["rank_corr"][m]))
    print("mean similarity by model:", report["mean_similarity"])


if __name__ == "__main__":
    main()
