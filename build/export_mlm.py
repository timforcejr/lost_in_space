"""
Purpose:          Wrapper that builds model B (microsoft/Multilingual-MiniLM-L12-H384, MIT, sha 6e8c1ec) browser assets
                  into web/model/mlm with the same export contract as build/export_model.py (hidden [B,13,T,384], attn
                  [B,12,12,T,T], q8 ONNX split into graph + onnx_data) and writes three parity gates.
                  Reuses build/export_model.py (Lens, split_external, PROBE), build/common.py and
                  experiments/p2_25_alignment_vs_drift/embed_any.py (load_any('mminilm'), hand_tokenizer) by import; edits none of them.
Verification:     eval/variants/mlm/export_parity.json (gates i, ii, iii)

Run (detached, see eval/variants/mlm/README.md):
  PYTHONDONTWRITEBYTECODE=1 python3 -I build/export_mlm.py
The tokenizer is NOT copied: B reuses web/model/mmini/tokenizer.json (token-id equality is gate i).
The fp32 graph and the quantization candidates stay in .cache/variants/mlm (gitignored).
"""
import sys, json, time, pathlib, shutil, hashlib
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / ".venv/lib/python3.13/site-packages", HERE, ROOT / "experiments/p2_25_alignment_vs_drift"):   # python -I ignores PYTHONPATH
    if str(p) not in sys.path:
        sys.path.append(str(p))
import numpy as np
import torch

from common import EVAL, LANGS, WEB, load_sib, write_json, mean_pool          # noqa: E402
import export_model as em                                                      # noqa: E402  (Lens, split_external, PROBE; its main() is not run)
import embed_any as ea                                                         # noqa: E402

REPO = "microsoft/Multilingual-MiniLM-L12-H384"
SHA = "6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf"
OUT = WEB / "model" / "mlm"
VAR = EVAL / "variants" / "mlm"
TMP = ROOT / ".cache" / "variants" / "mlm"
SHIPPED_TOK = WEB / "model" / "mmini" / "tokenizer.json"
T_GATE = {"min_cos": 0.99, "pair_err": 0.03, "gap_diff_l12": 0.02}
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def shipped_tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=str(SHIPPED_TOK), bos_token="<s>", eos_token="</s>", sep_token="</s>",
                                   cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", model_max_length=512)


def split_texts(splits):
    texts, lang, ids = [], [], []
    for s in splits:
        d = load_sib(s)
        for li, l in enumerate(LANGS):
            for r in d[l]:
                texts.append(r["text"]); lang.append(li); ids.append(r["id"])
    return texts, np.array(lang), np.array(ids)


unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.no_grad()
def torch_layers(tok, model, texts, bs=32):
    res = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        hs = model(**enc, output_hidden_states=True).hidden_states
        res.append(torch.stack([mean_pool(h, enc["attention_mask"]) for h in hs], 1).numpy())
    return np.concatenate(res)


def onnx_layers(path, tok, texts, bs=16):
    """Per-layer mean pools [N,13,384] from the ONNX 'hidden' output, batches sorted by length."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 16
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 13, 384), np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=128, return_tensors="np")
        m = enc["attention_mask"].astype(np.int64)
        h, _ = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": m})
        mm = m[:, None, :, None].astype(np.float32)
        out[idx] = (h * mm).sum(2) / mm.sum(2)
    return out


def lens_stats(layers, lang, ids):
    """Translation / unrelated mean cosine per layer, as build_cloud's layer lens (unit-normalized pools, test split)."""
    same = (ids[:, None] == ids[None]) & (lang[:, None] != lang[None])
    diff = (ids[:, None] != ids[None]) & (lang[:, None] != lang[None])
    rows = []
    for L in range(layers.shape[1]):
        e = unit(layers[:, L]); s = e @ e.T
        rows.append({"layer": L, "translation_cos": float(s[same].mean()), "random_cos": float(s[diff].mean()),
                     "gap": float(s[same].mean() - s[diff].mean())})
    return rows


def q_dynamic(fp32, dst, per_channel):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst.unlink(missing_ok=True)
    quantize_dynamic(str(fp32), str(dst), weight_type=QuantType.QUInt8, per_channel=per_channel)


def q_embedding_only(fp32, dst):
    """Fallback 2: all weights stay fp32 except the word-embedding table, stored int8 per row + DequantizeLinear."""
    import onnx
    from onnx import numpy_helper, helper
    m = onnx.load(str(fp32))
    emb = next(i for i in m.graph.initializer if list(i.dims) == [250037, 384])
    w = numpy_helper.to_array(emb)
    sc = (np.abs(w).max(1, keepdims=True) / 127).clip(min=1e-12).astype(np.float32)
    q = np.clip(np.round(w / sc), -127, 127).astype(np.int8)
    old = emb.name
    m.graph.initializer.remove(emb)
    m.graph.initializer.extend([numpy_helper.from_array(q, old + "_q8"), numpy_helper.from_array(sc.reshape(-1), old + "_scale")])
    m.graph.node.insert(0, helper.make_node("DequantizeLinear", [old + "_q8", old + "_scale"], [old], axis=0, name="emb_dequant"))
    assert m.opset_import[0].version >= 13
    onnx.save(m, str(dst))


def main():
    VAR.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    assert ea.mmini_snapshot().name == SHA, ea.mmini_snapshot().name
    tok, model = ea.load_any("mminilm")
    torch.set_num_threads(16)
    report = {"model": REPO, "sha": SHA, "license": "MIT", "tokenizer": "reuses web/model/mmini/tokenizer.json (not copied)",
              "thresholds": T_GATE}

    # ---------- gate (i): token ids from the shipped tokenizer.json == hand tokenizer on all 4016 cloud sentences + probes
    st = shipped_tokenizer()
    texts4016, _, _ = split_texts(("train", "dev", "test"))
    assert len(texts4016) == 4016
    probes = list(em.PROBE)
    bad = sum(st(t, truncation=True, max_length=128)["input_ids"] != tok(t, truncation=True, max_length=128)["input_ids"]
              for t in texts4016 + probes)
    ea_, eb_ = st(probes, padding=True, return_tensors="np"), tok(probes, padding=True, return_tensors="np")
    bad_batch = int(not (np.array_equal(ea_["input_ids"], eb_["input_ids"]) and np.array_equal(ea_["attention_mask"], eb_["attention_mask"])))
    report["gate_i_tokenizer"] = {"n_sentences": len(texts4016) + len(probes), "n_mismatch": int(bad), "probe_batch_mismatch": bad_batch,
                                  "tokenizer_file_sha256": hashlib.sha256(SHIPPED_TOK.read_bytes()).hexdigest(),
                                  "pass": bad == 0 and bad_batch == 0}
    log("gate i", report["gate_i_tokenizer"])

    # ---------- fp32 export, same contract as export_model.main
    fp32 = TMP / "model.onnx"
    enc = tok(probes[:2], padding=True, return_tensors="pt")
    torch.onnx.export(
        em.Lens(model).eval(), (enc["input_ids"], enc["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["hidden", "attn"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "hidden": {0: "batch", 2: "seq"}, "attn": {0: "batch", 3: "seq", 4: "seq2"}},
        opset_version=17, dynamo=False)
    log("fp32 exported", round(fp32.stat().st_size / 2**20, 1), "MB")
    cfg = model.config.to_dict()
    write_json(OUT / "config.json", {k: cfg[k] for k in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                                          "vocab_size", "max_position_embeddings", "pad_token_id") if k in cfg})

    # ---------- PyTorch references
    tt, tl, ti = split_texts(("test",))
    assert len(tt) == 816
    ref_layers = torch_layers(tok, model, tt)
    ref_lens = lens_stats(ref_layers, tl, ti)
    ref_probe = unit(torch_layers(tok, model, probes)[:, -1])
    log("torch layer-12 lens", ref_lens[12])
    import onnxruntime as ort
    penc = tok(probes, padding=True, return_tensors="np")
    feeds = {"input_ids": penc["input_ids"].astype(np.int64), "attention_mask": penc["attention_mask"].astype(np.int64)}

    def probe_eval(path):
        h, a = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)
        m = feeds["attention_mask"][..., None].astype(np.float32)
        return unit((h[:, -1] * m).sum(1) / m.sum(1)), h.shape, a.shape

    e32, hshape, ashape = probe_eval(fp32)
    report["hidden_shape"], report["attn_shape"] = list(hshape), list(ashape)
    report["cos_torch_vs_fp32_min"] = float(np.min(np.sum(ref_probe * e32, 1)))
    sim = lambda e: e @ e.T
    names = {"cat_en_es": (0, 1), "cat_en_hi": (0, 2), "cat_en_zh": (0, 3), "cat_vs_stocks": (0, 4), "dog_bites_man_vs_man_bites_dog": (5, 6)}
    report["probe_sims_torch"] = {k: float(sim(ref_probe)[i, j]) for k, (i, j) in names.items()}

    # ---------- quantize; fall back until gates ii and iii pass
    final = OUT / "onnx" / "model_quantized.onnx"
    attempts, chosen = [], None
    for name in ("dynamic_quint8_per_tensor", "dynamic_quint8_per_channel", "fp32_weights_int8_embedding_only"):
        for f in final.parent.glob("model_quantized.onnx*"):
            f.unlink()
        cand = TMP / f"cand_{name}.onnx"
        if name.startswith("dynamic"):
            q_dynamic(fp32, cand, name.endswith("channel"))
            shutil.copyfile(cand, final); em.split_external(final)      # parity is measured on the split file the browser loads
        else:
            q_embedding_only(fp32, cand); shutil.copyfile(cand, final)
        e8, _, _ = probe_eval(final)
        cos_min = float(np.min(np.sum(ref_probe * e8, 1)))
        pair_err = float(np.max(np.abs(sim(ref_probe) - sim(e8))))
        q_layers = onnx_layers(final, tok, tt)
        q_lens = lens_stats(q_layers, tl, ti)
        gap_diffs = [abs(a["gap"] - b["gap"]) for a, b in zip(ref_lens, q_lens)]
        et, eq = unit(ref_layers[:, 12]), unit(q_layers[:, 12])
        ij = np.random.default_rng(20261115).integers(0, 816, (20000, 2))
        pair_err_test = float(np.max(np.abs(np.sum(et[ij[:, 0]] * et[ij[:, 1]], 1) - np.sum(eq[ij[:, 0]] * eq[ij[:, 1]], 1))))
        files = {f.name: round(f.stat().st_size / 2**20, 1) for f in sorted(final.parent.glob("model_quantized.onnx*"))}
        ok_ii = cos_min >= T_GATE["min_cos"] and pair_err <= T_GATE["pair_err"]
        ok_iii = gap_diffs[12] <= T_GATE["gap_diff_l12"]
        a = {"variant": name, "cos_torch_vs_q8_min_probe": cos_min, "pair_sim_abs_err_q8_max_probe": pair_err,
             "test816_cos_min_l12": float(np.sum(et * eq, 1).min()), "test816_pair_err_max_l12_20000_random_pairs": pair_err_test,
             "gap_l12_torch": ref_lens[12]["gap"], "gap_l12_q8": q_lens[12]["gap"], "gap_l12_abs_diff": gap_diffs[12],
             "gap_all_layers_max_abs_diff": float(max(gap_diffs)), "files_mb": files, "gate_ii": bool(ok_ii), "gate_iii": bool(ok_iii),
             "probe_sims_q8": {k: float(sim(e8)[i, j]) for k, (i, j) in names.items()}}
        attempts.append(a)
        log("attempt", json.dumps({k: v for k, v in a.items() if k != "probe_sims_q8"}))
        if ok_ii and ok_iii and all(v < 100 for v in files.values()):
            chosen = (a, q_lens); break
    report["quantization_attempts"] = attempts
    if chosen is None:
        for f in final.parent.glob("model_quantized.onnx*"):
            f.unlink()
        report["gate_ii_onnx_vs_torch"] = {"pass": False}; report["gate_iii_int8_gap"] = {"pass": False}; report["all_gates_pass"] = False
        report["note"] = "no quantization variant passed; nothing installed in web/model/mlm/onnx"
        write_json(VAR / "export_parity.json", report)
        log("FAILED all variants"); return 2
    a, q_lens = chosen
    report["installed_variant"] = a["variant"]
    report["cos_torch_vs_q8_min"], report["pair_sim_abs_err_q8_max"] = a["cos_torch_vs_q8_min_probe"], a["pair_sim_abs_err_q8_max_probe"]
    report["probe_sims_q8"] = a["probe_sims_q8"]
    report["size_mb"] = dict(a["files_mb"], fp32_in_cache=round(fp32.stat().st_size / 2**20, 1))
    report["gate_ii_onnx_vs_torch"] = {"min_cos_probe": a["cos_torch_vs_q8_min_probe"], "pair_err_probe": a["pair_sim_abs_err_q8_max_probe"],
                                       "test816_cos_min_l12": a["test816_cos_min_l12"], "test816_pair_err_max": a["test816_pair_err_max_l12_20000_random_pairs"],
                                       "fp32_onnx_cos_min": report["cos_torch_vs_fp32_min"], "pass": a["gate_ii"]}
    report["gate_iii_int8_gap"] = {"layer12_gap_torch": a["gap_l12_torch"], "layer12_gap_q8": a["gap_l12_q8"], "abs_diff": a["gap_l12_abs_diff"],
                                   "max_abs_diff_all_layers": a["gap_all_layers_max_abs_diff"], "tolerance": T_GATE["gap_diff_l12"],
                                   "n_test_sentences": 816, "per_layer_torch": ref_lens, "per_layer_q8": q_lens, "pass": a["gate_iii"]}
    report["all_gates_pass"] = bool(report["gate_i_tokenizer"]["pass"] and a["gate_ii"] and a["gate_iii"])
    shutil.rmtree(OUT / "__pycache__", ignore_errors=True)
    write_json(VAR / "export_parity.json", report)
    log("DONE all_gates_pass =", report["all_gates_pass"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
