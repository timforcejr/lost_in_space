"""Re-score only the t-SNE language probe under the train / dev / test protocol of build_cloud.py.

Reads the t-SNE coordinates already written by build_tsne.py (no refit, nothing else changes) and rewrites
`metrics.language_probe_acc` in eval/tsne.json and web/data/tsne.json.

Run:  .venv/bin/python build/rescore_tsne_probe.py
"""
from __future__ import annotations

import json

import numpy as np

from build_cloud import language_probe
from common import EVAL, WEB, write_json


def main() -> None:
    meta = json.loads((WEB / "data" / "cloud.json").read_text(encoding="utf-8"))
    ts = json.loads((WEB / "data" / "tsne.json").read_text(encoding="utf-8"))
    z = np.array(ts["coords"], dtype=np.float64).reshape(-1, 3)
    lang = np.array([p["l"] for p in meta["points"]]); split = np.array([p["s"] for p in meta["points"]])
    tr, dv, te = split == 0, split == 1, split == 2
    acc = language_probe(z[tr], lang[tr], z[dv], lang[dv], z[te], lang[te])
    print("t-SNE language probe (was", ts["metrics"]["language_probe_acc"], ") ->", acc)
    ts["metrics"]["language_probe_acc"] = acc
    rep = json.loads((EVAL / "tsne.json").read_text(encoding="utf-8"))
    rep["metrics"]["language_probe_acc"] = acc
    write_json(EVAL / "tsne.json", rep)
    write_json(WEB / "data" / "tsne.json", ts)


if __name__ == "__main__":
    main()
