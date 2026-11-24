# Model B data files

Model B = `microsoft/Multilingual-MiniLM-L12-H384` (masked words only). Files in `web/data/mlm/` use the same names and JSON keys as model A's `web/data/` files, so the page can load either by registry path.

| File | Size | Shape / content | Made by |
|---|---|---|---|
| `web/data/mlm/cloud.json` | 1,802,281 B | keys identical to A's, in the same order: `model, dim, n, emb_scale, langs, topics, points (4016: t,l,c,i,s), heads {pca,fc,ae}, metrics {full_384d,pca,fc,ae,full_384d_lang_centred}, layer_lens (13 rows), layer_means [13][384], sim_ref, head_info`. `model` = the B repo id | `build/build_mlm_data.py` -> unmodified `build/build_cloud.py` main() |
| `web/data/mlm/cloud_emb.i8.bin` | 1,542,144 B | int8 [4016, 384], scale `emb_scale` in cloud.json; unit last-layer mean-pooled vectors | same |
| `web/data/mlm/placement.json` | 755 B | same keys as A's (heads pca/fc/ae: held-out and example-sentence overlap@10, head vs neighbor-center; 816 held-out, 16 example sentences) | `build/eval_placement.py` main(), unmodified, run on B |

Not built here (see Tier 1 assets below): `tsne.json`, `order.json`, grad export. t-SNE is not in `web/data/mlm/`, so the t-SNE tab hides itself.

Reports: `eval/variants/mlm/map_fidelity.json` (B's full metrics and train logs), `placement.json`, `data_checks.json`.

## How it was produced
`build/build_mlm_data.py` (new; edits no existing script) patches `common.MODEL_ID` (B repo id), `common.load_model` (-> `embed_any.load_any('mminilm')`, fp32 PyTorch, hand tokenizer), `common.WEB` and `common.EVAL` (scratch dirs under `.cache/variants/mlm/`, `webroot/js` symlinks to `web/js` so placement reads EXAMPLES), then imports and runs `build_cloud.main()` and `eval_placement.main()` with the same seed, steps and tau grid as A. The three output files were then copied into `web/data/mlm/`. About 4 min on CPU. Run: `PYTHONDONTWRITEBYTECODE=1 python3 -I build/build_mlm_data.py`. Embeddings are fp32 PyTorch (as A's), not the q8 ONNX.

## Checks (`build/check_mlm_data.py` -> `data_checks.json`, all pass)
- 4016 points; `t,l,c,i,s` of every point identical to A's, same order; langs and topics equal; head keys and shapes equal to A's; `layer_means` [13,384].
- B embeddings: int8 file size 4016x384, all finite, no zero rows, dequantized norms 0.997 to 1.005.
- B `layer_lens` vs the alignment-vs-drift study `sib_results.json` `mminilm` raw rows, 13 layers: max abs diff translation_cos 1.2e-7, random_cos 1.8e-7, language_probe 0 (limit 0.005).
- A's `web/data/cloud.json` `layer_lens` vs the study `shipped` rows: max abs diff 1.2e-7 / 6e-8 / 0. A's rows still match.
- A files: md5 and mtime of every `web/data/*.json|bin` unchanged (checked before and after).

## Numbers for the page Coder
- Layer 12, B: translation_cos 0.9427, random_cos 0.8918 (gap 0.051), language probe 1.000. A: 0.8364, 0.0553 (gap 0.781), probe 0.3995. For registry `lensCompare` (A): probe12 = 0.3995 (40%), gap12 = 0.781.
- B `sim_ref` (5/25/50/75/95): translation 0.9135, 0.9321, 0.9431, 0.9542, 0.9709; unrelated 0.8420, 0.8768, 0.8955, 0.9110, 0.9295. translation[0] 0.9135 <= unrelated[4] 0.9295, so the overlap-aware verdict applies.
- Map metrics (test split), B vs A: trust@10 pca 0.910 vs 0.785, fc 0.822 vs 0.808, ae 0.884 vs 0.836. Full 384-d translation retrieval P@1 0.863 vs 0.995; language probe in 3-d: pca 0.884, ae 0.806, fc 0.473 (A about 0.25). So the 3-d maps for B keep the languages visibly apart (PCA most), and translation retrieval in 3-d is low (pca 0.029).
- Placement (held-out overlap@10, head): pca 0.119, fc 0.074, ae 0.111 (neighbor-center 0.137, 0.092, 0.143).

## Deviations and notes
- The tau selected on dev for FC and AE is 0.05, the edge of the grid (0.05, 0.1, 0.2), for B (A picked 0.1 and 0.2). Recipe unchanged on purpose; noted only.
- Placement was run now with this step because A ships one.
- `lensCompare` is not a file here: the numbers above are for the registry; also in `data_checks.json` (`A_layer12`, `B_layer12`).
- No existing A file or build script was modified (new: `build/build_mlm_data.py`, `build/check_mlm_data.py`).

## Tier 1 assets (2026-11-22)
Wrapper `build/build_mlm_tier1.py order|tsne|grad` (patches `common.MODEL_ID/load_model/WEB/EVAL` as `build_mlm_data.py` does; `eval_order.py`, `build_tsne.py`, `export_grad.py` unmodified, same seeds). Checks: `build/check_mlm_tier1.py` -> `data_checks.json` key `tier1_P2_27d`.

| Asset | Status | Size | Checks |
|---|---|---|---|
| `web/data/mlm/order.json` | built | 112,406 B | keys, key order and value types equal A's; `probes` / `pair_probe` shapes equal A's; all finite. Report copy `order_eval.json` |
| `web/data/mlm/tsne.json` | built | 237,436 B | keys equal A's; 12,048 coords (4016 x 3, same as A); all finite; seed 20261115, perplexity 50 (dev choice). Report copy `tsne_eval.json` |
| `web/model/mgrad_mlm/onnx/model.onnx` + `config.json` | built | 42,878,593 B (40.9 MB) + 62 B | built-in asserts passed: manual backward vs autograd grad_rel 3.6e-5 (limit 1e-4), sim 4.8e-7 (1e-5); exported fp16 graph vs PyTorch autograd grad_rel 1.46e-5 (plan limit 2e-2). `grad_parity.json` |

- A's files: md5 and mtime of all 9 `web/data/*.json|bin` identical before and after (`data_checks.json`: `A_files_unchanged_md5_and_mtime` true). No edit to `web/js`, `web/index.html`, `web/css`, or existing build scripts.
- Parity is vs the fp32 PyTorch model on 6 probe sentences from `export_grad.py`; the browser feeds layer-0 states from the q8 model, so real IG values differ slightly. That difference is now measured in the browser (`browser_parity.md`): browser IG on B differs from Python IG through the same graph by at most 0.0015 per word, and the attributions add up to the score change within 1.5% of the change. IG on B is still not benchmarked for faithfulness (no deletion test has been run on it).
- B order probe, layer 12 (honest wording matters): vector cosine after shuffle 0.928, after one swap 0.996; the scramble probe reads natural vs scrambled at 0.999 (control 0.551). PAWS-X: cosine AUC 0.531, probe AUC 0.506 (control 0.514), i.e. no better than chance at telling paraphrase from reordered non-paraphrase. Compare these with A's order.json before showing them side by side.
- B t-SNE (tau 0.05 placement): trust@10 0.967, continuity@10 0.931, language probe 0.930, EN->es/hi/zh retrieval P@1 0.17 / 0.015 / 0.020, topic kNN 0.585. Placement inside tsne.json: held-out overlap@10 fitted 0.395 vs neighbor-center 0.269.
- Placement (`placement.json`) was already built above.
