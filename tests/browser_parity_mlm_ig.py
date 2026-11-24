"""
Title: browser_parity_mlm_ig.py, Python side of the browser parity for model B (and A as the control).
What it does: reads eval/variants/mlm/browser_parity.json (written by tests/browser_parity_mlm.mjs) and, for every probe pair and model, recomputes in Python
  (onnxruntime, CPU) with the SAME inputs as the browser: the 8-bit encoder's layer-0 states and the other sentence's embedding, the blank-<pad> baseline, and
  integrated gradients through the exported gradient graph (24 midpoint steps, same maths as web/js/grad.js). Reports max |browser IG - Python IG| per token and
  per word (word groups from the browser), the completeness gap, and also Python q8 similarities for the 2-sentence batch and the 7-sentence batch (explains
  batch-padding noise in the similarity parity). Adds the result under models.<id>.python and ig_vs_python and rewrites the same file.
Run (repo root, the venv's interpreter is broken, so):
  PYTHONPATH=.venv/lib/python3.13/site-packages python3 tests/browser_parity_mlm_ig.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "eval" / "variants" / "mlm" / "browser_parity.json"
d = json.loads(F.read_text(encoding="utf-8"))
DIRS = {"b": ("mlm", "mgrad_mlm"), "a": ("mmini", "mgrad")}
IG_STEPS, H, PAD = 24, 384, 1
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = lambda p: ort.InferenceSession(str(ROOT / "web" / "model" / p), so, providers=["CPUExecutionProvider"])


def q8_hidden(enc, lists):
    T = max(len(i) for i in lists)
    ids = np.full((len(lists), T), PAD, dtype=np.int64); mask = np.zeros_like(ids)
    for r, i in enumerate(lists):
        ids[r, :len(i)] = i; mask[r, :len(i)] = 1
    return enc.run(["hidden"], {"input_ids": ids, "attention_mask": mask})[0]


def unit_mean(h, n):
    e = h[:n].mean(0); return e / np.linalg.norm(e)


def ig(gr, x, base, e_other):
    T = x.shape[0]
    both = np.stack([x, base]).astype(np.float32)
    first = gr.run(None, {"h0": both, "e_other": e_other.astype(np.float32)})
    sim, simb = float(first[0][0]), float(first[0][1])
    diff = x - base
    acc = np.zeros_like(x, dtype=np.float64)
    for k0 in range(0, IG_STEPS, 8):
        S = min(8, IG_STEPS - k0)
        path = np.stack([base + ((k0 + s + 0.5) / IG_STEPS) * diff for s in range(S)]).astype(np.float32)
        g = gr.run(None, {"h0": path, "e_other": e_other.astype(np.float32)})[2]   # [S,13,T,H]
        acc += g[:, 0].sum(0)
    tok = (diff * acc).sum(1) / IG_STEPS
    return {"sim": sim, "sim_base": simb, "ig_tokens": tok, "ig_sum": float(tok.sum()), "ig_target": sim - simb}


for mid, (mdir, gdir) in DIRS.items():
    enc, gr = sess(f"{mdir}/onnx/model_quantized.onnx"), sess(f"{gdir}/onnx/model.onnx")
    m = d["models"][mid]; per, worst_tok, worst_word, worst_word_np = {}, 0.0, 0.0, 0.0
    probes = d["probes"]
    for name, p in m["pairs"].items():
        ia, ib = p["idsA"], p["idsB"]
        blank = lambda i: [i[0]] + [PAD] * (len(i) - 2) + [i[-1]]
        h = q8_hidden(enc, [ia, ib]); hb = q8_hidden(enc, [blank(ia), blank(ib)])
        eA, eB = unit_mean(h[0, 12], len(ia)), unit_mean(h[1, 12], len(ib))
        sim_pair = float(eA @ eB)
        row = {"sim_python_q8_pair_batch": sim_pair, "sim_browser": p["sim_browser"], "sim_abs_diff_same_batch": abs(sim_pair - p["sim_browser"])}
        for side, ids, e_other, groups, key in (("A", ia, eB, p["groupsA"], "igA"), ("B", ib, eA, p["groupsB"], "igB")):
            r = 0 if side == "A" else 1
            n = len(ids)
            res = ig(gr, h[r:r + 1, 0, :n][0], hb[r, 0, :n], e_other)
            br = np.array(p[key]["ig_tokens"])
            dt = np.abs(res["ig_tokens"] - br)
            words = [(g, sum(res["ig_tokens"][t] for t in g["tokens"]), sum(br[t] for t in g["tokens"])) for g in groups]
            dw = [abs(a - b) for g, a, b in words]; dwn = [abs(a - b) for g, a, b in words if not g["punct"]]
            row[side] = {"max_abs_diff_token": float(dt.max()), "max_abs_diff_word": float(max(dw)), "max_abs_diff_word_no_punct": float(max(dwn)),
                         "per_word": [{"word": g["text"], "python": float(a), "browser": float(b), "abs_diff": abs(float(a - b))} for g, a, b in words],
                         "python_ig_sum": res["ig_sum"], "python_ig_target": res["ig_target"], "python_completeness_gap": res["ig_sum"] - res["ig_target"],
                         "browser_ig_sum": p[key]["ig_sum"], "browser_ig_target": p[key]["ig_target"], "browser_completeness_gap": p[key]["ig_sum"] - p[key]["ig_target"],
                         "ig_max_abs_token_score_browser": float(np.abs(br).max()), "pearson_words": float(np.corrcoef([a for _, a, _ in words], [b for _, _, b in words])[0, 1]) if len(words) > 2 else None}
            worst_tok = max(worst_tok, row[side]["max_abs_diff_token"]); worst_word = max(worst_word, row[side]["max_abs_diff_word"]); worst_word_np = max(worst_word_np, row[side]["max_abs_diff_word_no_punct"])
        per[name] = row
    # python q8 sims for the 7-sentence batch (as export_model.py did) vs the 2-sentence batch
    order = [tuple(i) for i in (m["pairs"]["cat_en_es"]["idsA"], m["pairs"]["cat_en_es"]["idsB"], m["pairs"]["cat_en_hi"]["idsB"], m["pairs"]["cat_en_zh"]["idsB"], m["pairs"]["cat_vs_stocks"]["idsB"], m["pairs"]["dog_bites_man_vs_man_bites_dog"]["idsA"], m["pairs"]["dog_bites_man_vs_man_bites_dog"]["idsB"])]
    h7 = q8_hidden(enc, [list(o) for o in order]); E = np.stack([unit_mean(h7[i, 12], len(order[i])) for i in range(7)]); S7 = E @ E.T
    seven = {"cat_en_es": S7[0, 1], "cat_en_hi": S7[0, 2], "cat_en_zh": S7[0, 3], "cat_vs_stocks": S7[0, 4], "dog_bites_man_vs_man_bites_dog": S7[5, 6]}
    m["python"] = {"per_pair": per, "sim_python_q8_7_batch": {k: float(v) for k, v in seven.items()},
                   "sim_max_abs_diff_browser_vs_python_q8_7_batch": max(abs(seven[k] - m["pairs"][k]["sim_browser"]) for k in seven),
                   "sim_max_abs_diff_browser_vs_python_q8_same_pair_batch": max(r["sim_abs_diff_same_batch"] for r in per.values()),
                   "ig_max_abs_diff_token": worst_tok, "ig_max_abs_diff_word": worst_word, "ig_max_abs_diff_word_no_punct": worst_word_np,
                   "ig_max_browser_completeness_gap_abs": max(abs(r[s]["browser_completeness_gap"]) for r in per.values() for s in "AB"),
                   "ig_max_browser_completeness_gap_rel": max(abs(r[s]["browser_completeness_gap"]) / abs(r[s]["browser_ig_target"]) for r in per.values() for s in "AB"),
                   "note": "Python IG uses the same exported gradient graph and the same layer-0 states/other-sentence embedding as the browser (8-bit encoder run in onnxruntime CPU), so the difference measures the JavaScript + wasm runtime, not the graph or quantization."}
    print(mid, "sim vs q8(7-batch)", round(m["python"]["sim_max_abs_diff_browser_vs_python_q8_7_batch"], 5), "vs q8(pair batch)", round(m["python"]["sim_max_abs_diff_browser_vs_python_q8_same_pair_batch"], 5),
          "| IG max|d| token", round(worst_tok, 6), "word", round(worst_word, 6), "completeness gap rel", round(m["python"]["ig_max_browser_completeness_gap_rel"], 4))
d["ig_vs_python_summary"] = {k: {kk: d["models"][k]["python"][kk] for kk in ("ig_max_abs_diff_token", "ig_max_abs_diff_word", "ig_max_abs_diff_word_no_punct", "ig_max_browser_completeness_gap_abs", "ig_max_browser_completeness_gap_rel")} for k in d["models"]}
F.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
