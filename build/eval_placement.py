"""
Purpose:          Decision support for the project plan, decision 4.  Where should a new
                  sentence be drawn on the 3-d map: (a) through the head, or
                  (b) at the similarity-weighted center of its 384-d nearest
                  reference sentences?  Scores both on held-out sentences and
                  on the app's short example sentences.
Verification:     Output written to eval/placement.json

Run:  .venv/bin/python build/eval_placement.py
"""
from __future__ import annotations

import json
import re

import numpy as np

from build_cloud import apply_head
from common import EVAL, ROOT, WEB, embed, load_model, write_json

K, K_PLACE, TAU = 10, 8, 0.05


def overlap(x_q, pos_q, x_ref, c_ref):
    """Share of each query's 10 nearest reference sentences (384-d cosine) that are also its 10 nearest on the map."""
    hi = np.argsort(-(x_q @ x_ref.T), 1)[:, :K]
    d = ((pos_q[:, None] - c_ref[None]) ** 2).sum(-1)
    lo = np.argsort(d, 1)[:, :K]
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(hi, lo)]))


def barycenter(x_q, x_ref, c_ref):
    sim = x_q @ x_ref.T
    idx = np.argsort(-sim, 1)[:, :K_PLACE]
    w = np.exp((np.take_along_axis(sim, idx, 1) - sim.max(1, keepdims=True)) / TAU)
    w /= w.sum(1, keepdims=True)
    return (c_ref[idx] * w[..., None]).sum(1)


def main() -> None:
    meta = json.loads((WEB / "data" / "cloud.json").read_text(encoding="utf-8"))
    q = np.frombuffer((WEB / "data" / "cloud_emb.i8.bin").read_bytes(), dtype=np.int8).reshape(meta["n"], 384)
    x = q.astype(np.float64) * meta["emb_scale"]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    split = np.array([p["s"] for p in meta["points"]])
    ref, te = split == 0, split == 2
    # the app's own example sentences: short, everyday, unlike the news-style reference set
    js = (WEB / "js" / "main.js").read_text(encoding="utf-8")
    block = js[js.index("const EXAMPLES"):js.index("];", js.index("const EXAMPLES"))]
    examples = [s for trio in re.findall(r"\['[^']*', '([^']*)', '([^']*)'\]", block) for s in trio]
    tok, model = load_model()
    x_ex = embed(tok, model, examples).astype(np.float64)

    out = {"k": K, "k_place": K_PLACE, "tau": TAU, "n_heldout": int(te.sum()), "n_examples": len(examples),
           "example_best_neighbour_cos_mean": float((x_ex @ x[ref].T).max(1).mean()),
           "heldout_best_neighbour_cos_mean": float((x[te] @ x[ref].T).max(1).mean()), "heads": {}}
    for name, head in meta["heads"].items():
        h = {k: np.array(v) for k, v in head.items()}
        c_ref = apply_head(h, x[ref])
        out["heads"][name] = {
            "heldout_overlap10_head": overlap(x[te], apply_head(h, x[te]), x[ref], c_ref),
            "heldout_overlap10_neighbour_centre": overlap(x[te], barycenter(x[te], x[ref], c_ref), x[ref], c_ref),
            "examples_overlap10_head": overlap(x_ex, apply_head(h, x_ex), x[ref], c_ref),
            "examples_overlap10_neighbour_centre": overlap(x_ex, barycenter(x_ex, x[ref], c_ref), x[ref], c_ref),
        }
    write_json(EVAL / "placement.json", out)
    write_json(WEB / "data" / "placement.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
