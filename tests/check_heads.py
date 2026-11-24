"""
Purpose: compare 3-d coordinates the browser computed (tests/out/results.json, written by smoke.mjs) with
build/build_cloud.py apply_head() on the same de-quantized embeddings.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
from build_cloud import apply_head

meta = json.loads((ROOT / "web/data/cloud.json").read_text(encoding="utf-8"))
q = np.frombuffer((ROOT / "web/data/cloud_emb.i8.bin").read_bytes(), dtype=np.int8).reshape(meta["n"], 384)
x = q[:20].astype(np.float64) * meta["emb_scale"]
x /= np.linalg.norm(x, axis=1, keepdims=True)
got = json.loads((ROOT / "tests/out/results.json").read_text())["headCoords"]
worst = 0.0
for name, head in meta["heads"].items():
    h = {k: np.array(v) for k, v in head.items()}
    err = float(np.abs(apply_head(h, x) - np.array(got[name]).reshape(20, 3)).max())
    worst = max(worst, err)
    print(f"{name}: max |browser - python| = {err:.2e}")
sys.exit(0 if worst < 1e-3 else 1)
