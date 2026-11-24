# Model B assets

`microsoft/Multilingual-MiniLM-L12-H384` (MIT, sha `6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf`), masked-word training only.
Built by `build/export_mlm.py` (imports `build/export_model.py`, `build/common.py`, `experiments/p2_25_alignment_vs_drift/embed_any.py`; edits none of them).

- Run: `PYTHONDONTWRITEBYTECODE=1 python3 -I build/export_mlm.py` (about 20 s on CPU; the model weights come from `.cache/p2_25/hf`).
- Output: `web/model/mlm/config.json`, `web/model/mlm/onnx/model_quantized.onnx` (21.0 MB) and `model_quantized.onnx_data` (91.6 MB). Outputs `hidden` [B,13,T,384] and `attn` [B,12,12,T,T], inputs `input_ids`, `attention_mask`, mean pooling in the page as for the default model.
- Tokenizer: not copied. B uses `web/model/mmini/tokenizer.json` (and `tokenizer_config.json`); the fp32 graph stays in `.cache/variants/mlm/`.
- Results: `export_parity.json` (gates i, ii, iii, all pass; quantization used: dynamic uint8, per-tensor, the same as the default model, so no fallback was needed).
