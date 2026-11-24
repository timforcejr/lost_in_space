"""Checks for web/data/mlm/* (masked-word model): format vs A, same ids/order/texts, finite embeddings, layer lens vs the alignment-vs-drift study's sib_results.json,
A's layer_lens vs its 'shipped' rows. Writes eval/variants/mlm/data_checks.json. Run: python -I build/check_mlm_data.py"""
import json, pathlib, numpy as np
R = pathlib.Path(__file__).resolve().parents[1]
A = json.load(open(R / "web/data/cloud.json")); B = json.load(open(R / "web/data/mlm/cloud.json"))
sib = json.load(open(R / "experiments/p2_25_alignment_vs_drift/sib_results.json"))["models"]
out = {}
out["top_keys_equal"] = list(A) == list(B)
out["n_equal"] = A["n"] == B["n"] == 4016 and len(A["points"]) == len(B["points"])
out["points_identical_t_l_c_i_s"] = all(a == b for a, b in zip(A["points"], B["points"])) and len(A["points"]) == len(B["points"])
out["langs_topics_equal"] = A["langs"] == B["langs"] and A["topics"] == B["topics"]
out["heads_keys_equal"] = {k: sorted(A["heads"][k]) == sorted(B["heads"][k]) for k in A["heads"]}
out["head_shapes_equal"] = {k: {kk: np.array(A["heads"][k][kk]).shape == np.array(B["heads"][k][kk]).shape for kk in A["heads"][k]} for k in A["heads"]}
out["layer_means_shape"] = [list(np.array(A["layer_means"]).shape), list(np.array(B["layer_means"]).shape)]
out["layer_lens_keys_equal"] = A["layer_lens"][0].keys() == B["layer_lens"][0].keys() and len(B["layer_lens"]) == 13
out["sim_ref_keys_equal"] = A["sim_ref"].keys() == B["sim_ref"].keys()
out["metrics_spaces_equal"] = list(A["metrics"]) == list(B["metrics"])
for nm, f in (("A", R / "web/data/cloud_emb.i8.bin"), ("B", R / "web/data/mlm/cloud_emb.i8.bin")):
    q = np.frombuffer(f.read_bytes(), np.int8); out[f"emb_{nm}_bytes"] = q.size; out[f"emb_{nm}_ok"] = q.size == 4016 * 384
qb = np.frombuffer((R / "web/data/mlm/cloud_emb.i8.bin").read_bytes(), np.int8).reshape(4016, 384).astype(np.float64) * B["emb_scale"]
n = np.linalg.norm(qb, axis=1); out["B_emb_finite"] = bool(np.isfinite(qb).all()); out["B_emb_norm_min_max"] = [float(n.min()), float(n.max())]
out["B_emb_zero_rows"] = int((np.abs(qb).sum(1) == 0).sum())
for k, h in B["heads"].items():
    out[f"B_head_{k}_finite"] = all(np.isfinite(np.array(v)).all() for v in h.values())
out["B_layer_means_finite"] = bool(np.isfinite(np.array(B["layer_means"])).all())
def cmp(rows, ref, tag):
    d = {"translation_cos": 0, "random_cos": 0, "language_probe_acc": 0}
    for L in range(13):
        r = ref[L]["raw"]
        d["translation_cos"] = max(d["translation_cos"], abs(rows[L]["translation_cos"] - r["translation_cos"]))
        d["random_cos"] = max(d["random_cos"], abs(rows[L]["random_cos"] - r["random_cos"]))
        d["language_probe_acc"] = max(d["language_probe_acc"], abs(rows[L]["language_probe_acc"] - r["language_probe_acc"]))
    out[tag] = d
cmp(B["layer_lens"], sib["mminilm"]["layers"], "B_layer_lens_vs_sibling_mminilm_max_abs_diff")
cmp(A["layer_lens"], sib["shipped"]["layers"], "A_layer_lens_vs_sibling_shipped_max_abs_diff")
out["B_layer12"] = B["layer_lens"][12]; out["A_layer12"] = A["layer_lens"][12]
out["B_sim_ref"] = B["sim_ref"]; out["A_sim_ref"] = A["sim_ref"]
out["B_overlap_rule_translation5_le_unrelated95"] = B["sim_ref"]["translation"][0] <= B["sim_ref"]["unrelated"][4]
out["B_model_field"] = B["model"]
json.dump(out, open(R / "eval/variants/mlm/data_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
