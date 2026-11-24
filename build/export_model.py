"""
Purpose:          Export the multilingual sentence encoder to ONNX with every
                  layer's hidden states and attention maps as outputs, quantize
                  it for the browser, and check parity against PyTorch.
Verification:     Parity numbers written to eval/export_parity.json

Run:  .venv/bin/python build/export_model.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from common import CACHE, EVAL, MODEL_DIRNAME, MODEL_ID, WEB, embed, load_model, write_json

OUT = WEB / "model" / MODEL_DIRNAME
PROBE = [
    "The cat sleeps on the sofa.",
    "El gato duerme en el sofá.",
    "बिल्ली सोफे पर सो रही है।",
    "猫在沙发上睡觉。",
    "The stock market fell sharply today.",
    "Dog bites man.",
    "Man bites dog.",
]


class Lens(torch.nn.Module):
    """Encoder that returns all hidden states and all attention maps as two stacked tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, output_attentions=True)
        hidden = torch.stack(out.hidden_states, dim=1)      # [B, L+1, T, H]
        attn = torch.stack(out.attentions, dim=1)           # [B, L, heads, T, T]
        return hidden, attn


def split_external(path, threshold: int = 10_000_000) -> None:
    """Move the one very large weight (the word-embedding table, 96 MB) into `<name>.onnx_data`.

    GitHub refuses files over 100 MB.  After the split the graph file is about 16 MB and the data file 96 MB,
    and nothing about the model changes.  transformers.js loads the pair with use_external_data_format: true.
    """
    import onnx

    m = onnx.load(str(path))
    data = path.with_name(path.name + "_data")
    data.unlink(missing_ok=True)
    onnx.save_model(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=data.name,
                    size_threshold=threshold, convert_attribute=False)


def main() -> None:
    tok, model = load_model()
    lens = Lens(model).eval()
    enc = tok(PROBE[:2], padding=True, return_tensors="pt")
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    fp32 = OUT / "onnx" / "model.onnx"
    torch.onnx.export(
        lens, (enc["input_ids"], enc["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["hidden", "attn"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "hidden": {0: "batch", 2: "seq"}, "attn": {0: "batch", 3: "seq", 4: "seq2"}},
        opset_version=17, dynamo=False,
    )

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q8 = OUT / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QUInt8)

    # tokenizer + minimal config for transformers.js
    # transformers.js 3.x reads the ORIGINAL tokenizer.json (Metaspace with add_prefix_space); a file re-saved by
    # transformers 5 drops that key and the browser then tokenizes without the word-boundary marker.
    from huggingface_hub import hf_hub_download
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(hf_hub_download(MODEL_ID, name, cache_dir=CACHE / "hf"), OUT / name)
    cfg = model.config.to_dict()
    write_json(OUT / "config.json", {k: cfg[k] for k in (
        "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
        "max_position_embeddings", "pad_token_id") if k in cfg})

    # parity: PyTorch vs ONNX fp32 vs ONNX q8 on the probe sentences (padded batch)
    import onnxruntime as ort
    enc = tok(PROBE, padding=True, return_tensors="np")
    feeds = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
    ref = embed(tok, model, PROBE)

    def pooled(path):
        h, a = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)
        m = feeds["attention_mask"][..., None].astype(np.float32)
        e = (h[:, -1] * m).sum(1) / m.sum(1)
        return e / np.linalg.norm(e, axis=1, keepdims=True), h.shape, a.shape

    e32, hs, as_ = pooled(fp32)
    split_external(q8)
    e8, _, _ = pooled(q8)                               # parity is measured on the split file the browser loads
    sim = lambda e: (e @ e.T)
    report = {
        "model": MODEL_ID,
        "hidden_shape": list(hs), "attn_shape": list(as_),
        "size_mb": {"fp32": round(fp32.stat().st_size / 2**20, 1), "q8_graph": round(q8.stat().st_size / 2**20, 1),
                    "q8_data": round(q8.with_name(q8.name + "_data").stat().st_size / 2**20, 1)},
        "cos_torch_vs_fp32_min": float(np.min(np.sum(ref * e32, 1))),
        "cos_torch_vs_q8_min": float(np.min(np.sum(ref * e8, 1))),
        "pair_sim_abs_err_q8_max": float(np.max(np.abs(sim(ref) - sim(e8)))),
        "probe_sims_torch": {"cat_en_es": float(sim(ref)[0, 1]), "cat_en_hi": float(sim(ref)[0, 2]),
                             "cat_en_zh": float(sim(ref)[0, 3]), "cat_vs_stocks": float(sim(ref)[0, 4]),
                             "dog_bites_man_vs_man_bites_dog": float(sim(ref)[5, 6])},
        "probe_sims_q8": {"cat_en_es": float(sim(e8)[0, 1]), "cat_en_hi": float(sim(e8)[0, 2]),
                          "cat_en_zh": float(sim(e8)[0, 3]), "cat_vs_stocks": float(sim(e8)[0, 4]),
                          "dog_bites_man_vs_man_bites_dog": float(sim(e8)[5, 6])},
    }
    write_json(EVAL / "export_parity.json", report)
    print(json.dumps(report, indent=1))
    shutil.rmtree(OUT / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
