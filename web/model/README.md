# Model files: source, license, changes

**Source checkpoint:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The license text is also in
`web/vendor/transformers/LICENSE`.

**The model files were modified.** Apart from the two tokenizer files, they are converted from the checkpoint above, not copies of it:

| File | What it is | Made by |
|---|---|---|
| `mmini/onnx/model_quantized.onnx` (+ `_data`) | The encoder exported to ONNX (opset 17) with every hidden state and attention map as outputs, then dynamically quantized to 8-bit weights (`QUInt8`). The full-precision export (`model.onnx`, 470 MB) is made by the same script but is not in this repo. | `build/export_model.py` |
| `mmini/tokenizer.json`, `tokenizer_config.json` | The checkpoint's own tokenizer files, copied unchanged | `build/export_model.py` |
| `mmini/config.json` | A shortened config (7 fields) for transformers.js | `build/export_model.py` |
| `mgrad/onnx/model.onnx` | A new graph: the 12 encoder layers' forward pass and a hand-written backward pass, giving d(similarity)/d(hidden state) at every layer. Weights from the same checkpoint, stored as float16. | `build/export_grad.py` |
| `mgrad/config.json` | Three config fields | `build/export_grad.py` |

Parity of the converted files against PyTorch is logged in `eval/export_parity.json` and `eval/grad_parity.json`.

---

## Second model (header dropdown, "masked-word model"): source, license, changes

**Source checkpoint:** [`microsoft/Multilingual-MiniLM-L12-H384`](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384),
revision (git sha) `6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf` (short: `6e8c1ec`). The model card declares `license: mit`. The Hugging Face repository
has no LICENSE file of its own; its card points to Microsoft's `microsoft/unilm` repository (`minilm/`), whose root LICENSE is the MIT text below, copied
unchanged (including the copyright line) from `https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE` on 2026-11-22.

**The model files were modified.** They are converted from the checkpoint above, not copies of it:

| File | What it is | Made by |
|---|---|---|
| `mlm/onnx/model_quantized.onnx` (+ `_data`) | The encoder exported to ONNX with every hidden state and attention map as outputs, then dynamically quantized to 8-bit weights (`QUInt8`). The full-precision export stays in the build cache and is not in this repo. | `build/export_mlm.py` |
| `mlm/config.json` | A shortened config for transformers.js | `build/export_mlm.py` |
| `mgrad_mlm/onnx/model.onnx` | The gradient graph for this model: the encoder's forward pass and a hand-written backward pass (d(similarity)/d(hidden state) at every layer), weights from the same checkpoint stored as float16. Built with the `build/export_grad.py` wrapper. A derived work of the checkpoint, so it carries the same MIT license and notice. | `build/build_mlm_tier1.py` (via `build/export_grad.py`) |

**The tokenizer is shared with the default model.** This model uses the same XLM-R SentencePiece vocabulary as the default model, so no tokenizer is copied: it
reads `mmini/tokenizer.json` and `mmini/tokenizer_config.json` (the default model's own files; identity of token ids on 4,023 sentences is checked, 0 mismatches).
The MIT notice for the weights applies to the `mlm/` and `mgrad_mlm/` folders; the tokenizer files stay under the default model's license above.

Parity of the converted files against PyTorch is logged in `eval/variants/mlm/export_parity.json` and `eval/variants/mlm/grad_parity.json`.

**MIT license of the second model's weights (and of the files converted from them):**

```
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
