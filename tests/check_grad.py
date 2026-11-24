"""
Purpose:          Compare the gradient explanations the BROWSER computed
                  (tests/out/results.json, written by tests/smoke.mjs) with
                  PyTorch autograd, for the same sentences, in two ways:
                  (A) same inputs: the layer-0 states and the other
                      sentence's embedding come from the 8-bit model, as in
                      the browser.  What is left is the float16 weights, the
                      hand-written backward pass and the JavaScript code.
                      This is the implementation check and it must pass.
                  (B) end to end: everything in full precision.  The gap is
                      the effect of 8-bit weights on the explanation; it is
                      reported, not asserted.
Verification:     exits non-zero if (A) fails: word-level integrated
                  gradients below r = 0.98 (the 8-bit model's own outputs differ between
                  the Python and browser runtimes in the third decimal)

Run:  .venv/bin/python tests/check_grad.py        (after a smoke run; writes tests/out/grad_browser_parity.json)
      .venv/bin/python tests/check_grad.py --record (also overwrites eval/grad_browser_parity.json, the results input)
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
from common import load_model, run_layers, word_groups  # noqa: E402
from eval_attribution import IG_STEPS, Explainer  # noqa: E402

res = json.loads((ROOT / "tests" / "out" / "results.json").read_text(encoding="utf-8"))["results"]
tok, model = load_model()
ex = Explainer(tok, model)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
q8 = ort.InferenceSession(str(ROOT / "web" / "model" / "mmini" / "onnx" / "model_quantized.onnx"), so, providers=["CPUExecutionProvider"])


def q8_hidden(id_lists):
    """Run the 8-bit model on a padded batch, as web/js/engine.js does.  Returns [B, 13, T, 384]."""
    T = max(len(i) for i in id_lists)
    ids = np.full((len(id_lists), T), tok.pad_token_id, dtype=np.int64); mask = np.zeros_like(ids)
    for r, i in enumerate(id_lists):
        ids[r, :len(i)] = i; mask[r, :len(i)] = 1
    return q8.run(["hidden"], {"input_ids": ids, "attention_mask": mask})[0]


def from_states(h0, hb, e_other):
    """Integrated gradients by autograd, starting from given layer-0 states (same maths as grad.js)."""
    with torch.no_grad():
        s = F.cosine_similarity(run_layers(model, h0.clone())[-1][0].mean(0), e_other, dim=0)
    alphas = (torch.arange(IG_STEPS).float() + 0.5) / IG_STEPS
    path = (hb + alphas[:, None, None] * (h0 - hb)).requires_grad_(True)
    sp = F.cosine_similarity(run_layers(model, path)[-1].mean(1), e_other[None], dim=1)
    ig = ((h0 - hb)[0] * torch.autograd.grad(sp.sum(), path)[0].mean(0)).sum(1)
    model.zero_grad(set_to_none=True)
    return s.item(), ig.detach().numpy()


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / (np.abs(np.asarray(a)).max() + 1e-12))


rows = []
for name, r in res.items():
    g = r.get("grad")
    if not g:
        continue
    ids, ids_b = g["idsA"], g["idsB"]
    groups = word_groups(tok.convert_ids_to_tokens(ids))
    keep = [i for i, gr in enumerate(groups) if not gr["punct"]]
    per_word = lambda v: np.array([v[gr["tokens"]].sum() for gr in groups])[keep]
    ig_web = np.array(g["igA"])[keep]

    # (A) same inputs as the browser
    blank = lambda i: [i[0]] + [tok.pad_token_id] * (len(i) - 2) + [i[-1]]
    hq, hbq = q8_hidden([ids, ids_b]), q8_hidden([blank(ids), blank(ids_b)])
    n, nb = len(ids), len(ids_b)
    e_b = F.normalize(torch.tensor(hq[1, 12, :nb]).mean(0), dim=0)
    sA, igA = from_states(torch.tensor(hq[0:1, 0, :n]), torch.tensor(hbq[0:1, 0, :n]), e_b)
    a = {"sim": sA, "ig_r": float(np.corrcoef(per_word(igA), ig_web)[0, 1]), "ig_d": float(np.abs(per_word(igA) - ig_web).max())}

    # (B) full precision end to end
    with torch.no_grad():
        e_full = F.normalize(model(input_ids=torch.tensor([ids_b])).last_hidden_state[0].mean(0), dim=0)
    s0, scores, _ = ex.explain(ids, e_full, np.random.default_rng(0))
    b = {"sim": s0, "ig_r": float(np.corrcoef(per_word(scores["integrated_gradients"]), ig_web)[0, 1]),
         "ig_d": float(np.abs(per_word(scores["integrated_gradients"]) - ig_web).max())}
    rows.append({"pair": name, "words": len(keep), "sim_browser": g["sim"], "sim_browser_main_model": r["sim"], "grad_ms": g.get("ms"),
                 "same_inputs": a, "full_precision": b})

for key, title in (("same_inputs", "(A) same inputs as the browser: checks the exported backward pass and the JavaScript"),
                   ("full_precision", "(B) full precision end to end: shows what 8-bit weights do to the explanation")):
    print(title)
    print(f"  {'pair':10s} words  sim(torch) sim(web)   IG r   IG max|d|")
    for r in rows:
        v = r[key]
        print(f"  {r['pair']:10s} {r['words']:5d}  {v['sim']:9.4f} {r['sim_browser']:8.4f}  {v['ig_r']:6.4f}  {v['ig_d']:8.4f}")
worst_r = min(r["same_inputs"]["ig_r"] for r in rows)
summary = {"worst_score_diff_gradient_vs_main_model": max(abs(r["sim_browser"] - r["sim_browser_main_model"]) for r in rows),
           "worst_score_diff_browser_vs_full_precision": max(abs(r["sim_browser_main_model"] - r["full_precision"]["sim"]) for r in rows),
           "gradient_seconds_range": [min(r["grad_ms"] for r in rows) / 1000, max(r["grad_ms"] for r in rows) / 1000],
           "timing_note": "headless Chrome on the build machine (64-core desktop CPU), one WASM thread, model already loaded, both sentences",
           "worst_ig_pearson_same_inputs": worst_r,
           "worst_ig_pearson_full_precision": min(r["full_precision"]["ig_r"] for r in rows),
           "worst_ig_abs_diff_full_precision": max(r["full_precision"]["ig_d"] for r in rows)}
print(json.dumps(summary, indent=1))
out_path = ROOT / "eval" / "grad_browser_parity.json" if "--record" in sys.argv else ROOT / "tests" / "out" / "grad_browser_parity.json"
out_path.write_text(json.dumps({"pairs": rows, **summary}), encoding="utf-8")
print("wrote", out_path.relative_to(ROOT))
ok = worst_r >= 0.98
print("PASS" if ok else "FAIL", "(implementation check A)")
sys.exit(0 if ok else 1)
