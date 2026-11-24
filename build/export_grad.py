"""
Purpose:          Gradients for the browser.  onnxruntime-web cannot
                  backpropagate, so this file writes the encoder's forward
                  AND backward pass out by hand as one ordinary graph and
                  exports it to ONNX.  The browser then gets d(similarity) /
                  d(hidden state) at every layer from a plain inference call,
                  which makes integrated gradients run live on the page.
                  (2026-11-19: the page no longer shows Grad-CAM; the per-layer
                  gradient outputs stay in the graph, unused above layer 0.)
Verification:     The hand-written backward pass is checked against
                  torch.autograd, and the exported file against PyTorch;
                  numbers in eval/grad_parity.json

Graph:  inputs   h0       [S, T, 384]  layer-0 states (embedding output) of ONE sentence, S copies / path points
                 e_other  [384]        unit embedding of the other sentence (held constant)
        outputs  sim      [S]          cosine(mean-pooled last layer, e_other)
                 hidden   [S, 13, T, 384]  the states at every layer
                 grad     [S, 13, T, 384]  d sim / d hidden

Only the 12 encoder layers are inside (21 M weights); the 96 M-weight word-embedding table is not needed
because the browser already has layer-0 states from the main model.  Weights are stored as float16 and
cast to float32 when the file is loaded, so the file is about 43 MB.  Every weight matrix is used by the
forward and the backward pass without a transposed copy (see lin_back).

Run:  .venv/bin/python build/export_grad.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from common import EVAL, WEB, load_model, run_layers, write_json

OUT = WEB / "model" / "mgrad"
PROBE = ["The cat sleeps on the sofa.", "El gato duerme en el sofá.", "बिल्ली सोफे पर सो रही है।", "猫在沙发上睡觉。",
         "The dog bit the man.", "I went to the hospital to see the doctor."]


def lin_back(g, wt):
    """Backward of y = x @ wt (wt: [in, out]):  dx = g @ wt.T, written so wt itself is the MatMul operand."""
    return torch.matmul(wt, g.transpose(-1, -2)).transpose(-1, -2)


class GradLens(torch.nn.Module):
    """BERT-style encoder with a hand-written backward pass for S = cosine(mean pool, e_other)."""

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        assert cfg.hidden_act == "gelu"
        self.heads, self.eps, self.n_layers = cfg.num_attention_heads, cfg.layer_norm_eps, cfg.num_hidden_layers
        self.scale = 1.0 / math.sqrt(cfg.hidden_size // cfg.num_attention_heads)
        P = torch.nn.Parameter
        for i, layer in enumerate(model.encoder.layer):
            a, o = layer.attention.self, layer.attention.output
            named = {"q": a.query, "k": a.key, "v": a.value, "o": o.dense, "i": layer.intermediate.dense, "f": layer.output.dense}
            for name, lin in named.items():
                self.register_parameter(f"w{name}{i}", P(lin.weight.detach().t().contiguous(), requires_grad=False))
                self.register_parameter(f"b{name}{i}", P(lin.bias.detach().clone(), requires_grad=False))
            for name, ln in (("1", o.LayerNorm), ("2", layer.output.LayerNorm)):
                self.register_parameter(f"g{name}{i}", P(ln.weight.detach().clone(), requires_grad=False))
                self.register_parameter(f"c{name}{i}", P(ln.bias.detach().clone(), requires_grad=False))

    def p(self, name, i):
        return getattr(self, f"{name}{i}")

    def split(self, x):                                    # [S, T, H] -> [S, heads, T, d]
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    @staticmethod
    def merge(x):                                          # [S, heads, T, d] -> [S, T, H]
        return x.transpose(1, 2).flatten(-2)

    def ln(self, u, i, which):
        mean = u.mean(-1, keepdim=True)
        std = ((u - mean).pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        xhat = (u - mean) / std
        return xhat * self.p("g" + which, i) + self.p("c" + which, i), (xhat, std)

    def ln_back(self, g, cache, i, which):
        xhat, std = cache
        gx = g * self.p("g" + which, i)
        return (gx - gx.mean(-1, keepdim=True) - xhat * (gx * xhat).mean(-1, keepdim=True)) / std

    def forward(self, h0, e_other):
        x, hidden, caches = h0, [h0], []
        for i in range(self.n_layers):
            q = self.split(x @ self.p("wq", i) + self.p("bq", i))
            k = self.split(x @ self.p("wk", i) + self.p("bk", i))
            v = self.split(x @ self.p("wv", i) + self.p("bv", i))
            att = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
            y1, c1 = self.ln(self.merge(att @ v) @ self.p("wo", i) + self.p("bo", i) + x, i, "1")
            z = y1 @ self.p("wi", i) + self.p("bi", i)
            cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            y2, c2 = self.ln((z * cdf) @ self.p("wf", i) + self.p("bf", i) + y1, i, "2")
            caches.append((q, k, v, att, c1, z, cdf, c2))
            hidden.append(y2)
            x = y2

        # S = cos(mean_t x_t, e_other);  dS/dx_t = (e_other - S * m_hat) / (|m| T)   for every token t
        m = x.mean(1)
        mn = m.norm(dim=-1, keepdim=True)
        m_hat = m / mn
        sim = (m_hat * e_other).sum(-1)
        count = torch.ones_like(x[:, :, :1]).sum(1, keepdim=True)                      # [S, 1, 1] = T, without a shape op
        g = ((e_other - sim[:, None] * m_hat) / mn)[:, None, :] / count + torch.zeros_like(x)
        grads = [g]
        for i in reversed(range(self.n_layers)):
            q, k, v, att, c1, z, cdf, c2 = caches[i]
            du = self.ln_back(g, c2, i, "2")                                             # into (ffn_out + y1)
            dz = lin_back(du, self.p("wf", i)) * (cdf + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))
            du1 = self.ln_back(du + lin_back(dz, self.p("wi", i)), c1, i, "1")           # into (attn_out + x)
            dctx = self.split(lin_back(du1, self.p("wo", i)))
            datt = dctx @ v.transpose(-1, -2)
            dv = att.transpose(-1, -2) @ dctx
            ds = att * (datt - (datt * att).sum(-1, keepdim=True)) * self.scale
            g = (du1 + lin_back(self.merge(ds @ k), self.p("wq", i))
                 + lin_back(self.merge(ds.transpose(-1, -2) @ q), self.p("wk", i))
                 + lin_back(self.merge(dv), self.p("wv", i)))
            grads.append(g)
        return sim, torch.stack(hidden, 1), torch.stack(grads[::-1], 1)


def autograd_reference(model, h0, e_other):
    """The same three outputs from the Hugging Face encoder and torch.autograd."""
    h0 = h0.clone().requires_grad_(True)
    hs = run_layers(model, h0)
    for h in hs:
        h.retain_grad()
    sim = F.cosine_similarity(hs[-1].mean(1), e_other[None], dim=1)
    sim.sum().backward()
    return sim.detach(), torch.stack([h.detach() for h in hs], 1), torch.stack([h.grad for h in hs], 1)


def to_fp16(path_in, path_out, min_elems: int = 1024) -> None:
    """Store large float32 weights as float16 and cast them back to float32 at load time."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path_in))
    casts = []
    for init in list(m.graph.initializer):
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_elems:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        name = init.name
        m.graph.initializer.remove(init)
        m.graph.initializer.append(numpy_helper.from_array(arr, name + "_f16"))
        casts.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name="cast_" + name))
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path_out))


def main() -> None:
    torch.manual_seed(0)
    tok, model = load_model()
    lens = GradLens(model).eval()

    def states(text):
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            h0 = model.embeddings(input_ids=ids)
            e = F.normalize(model(input_ids=ids).last_hidden_state[0].mean(0), dim=0)
        return ids, h0, e

    # 1. hand-written backward == autograd
    embs = [states(t) for t in PROBE]
    err = {"sim": 0.0, "hidden": 0.0, "grad_abs": 0.0, "grad_rel": 0.0}
    for a in range(len(PROBE)):
        _, h0, _ = embs[a]
        e_o = embs[(a + 1) % len(PROBE)][2]
        path = h0 * torch.linspace(0.2, 1.0, 3)[:, None, None]          # three path points, as integrated gradients uses
        s_ref, h_ref, g_ref = autograd_reference(model, path, e_o)
        with torch.no_grad():
            s, h, g = lens(path, e_o)
        err["sim"] = max(err["sim"], float((s - s_ref).abs().max()))
        err["hidden"] = max(err["hidden"], float((h - h_ref).abs().max()))
        err["grad_abs"] = max(err["grad_abs"], float((g - g_ref).abs().max()))
        err["grad_rel"] = max(err["grad_rel"], float((g - g_ref).norm() / g_ref.norm()))
    print("manual backward vs autograd:", err)
    assert err["grad_rel"] < 1e-4 and err["sim"] < 1e-5, "hand-written backward pass does not match autograd"

    # 2. export
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    fp32 = OUT / "onnx" / "model_fp32.onnx"
    final = OUT / "onnx" / "model.onnx"
    _, h0, _ = embs[0]
    torch.onnx.export(
        lens, (h0.repeat(2, 1, 1), embs[1][2]), str(fp32), input_names=["h0", "e_other"],
        output_names=["sim", "hidden", "grad"], opset_version=17, dynamo=False,
        dynamic_axes={"h0": {0: "steps", 1: "seq"}, "sim": {0: "steps"}, "hidden": {0: "steps", 2: "seq"}, "grad": {0: "steps", 2: "seq"}},
    )
    to_fp16(fp32, final)
    write_json(OUT / "config.json", {"model_type": "bert", "hidden_size": model.config.hidden_size,
                                     "num_hidden_layers": model.config.num_hidden_layers})

    # 3. exported file vs autograd, on shapes it was not traced with
    import onnxruntime as ort
    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    onnx_err = {"sim": 0.0, "grad_rel": 0.0, "hidden": 0.0}
    for a in range(len(PROBE)):
        _, h0, _ = embs[a]
        e_o = embs[(a + 2) % len(PROBE)][2]
        path = h0 * torch.linspace(0.1, 1.0, 5)[:, None, None]
        s_ref, h_ref, g_ref = autograd_reference(model, path, e_o)
        s, h, g = sess.run(None, {"h0": path.numpy(), "e_other": e_o.numpy()})
        onnx_err["sim"] = max(onnx_err["sim"], float(np.abs(s - s_ref.numpy()).max()))
        onnx_err["hidden"] = max(onnx_err["hidden"], float(np.abs(h - h_ref.numpy()).max()))
        onnx_err["grad_rel"] = max(onnx_err["grad_rel"], float(np.linalg.norm(g - g_ref.numpy()) / np.linalg.norm(g_ref.numpy())))
    print("exported float16-weight graph vs autograd:", onnx_err)
    assert onnx_err["grad_rel"] < 2e-2
    report = {"manual_vs_autograd_max": err, "onnx_fp16_vs_autograd_max": onnx_err, "probe_sentences": len(PROBE),
              "size_mb": round(final.stat().st_size / 2**20, 1), "inputs": ["h0 [S,T,384]", "e_other [384]"],
              "outputs": ["sim [S]", "hidden [S,13,T,384]", "grad [S,13,T,384]"]}
    write_json(EVAL / "grad_parity.json", report)
    fp32.unlink()
    print("wrote", final, report["size_mb"], "MB")


if __name__ == "__main__":
    main()
