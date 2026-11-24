"""
Purpose:          Fit a 3-d t-SNE map of the reference cloud as a fourth map
                  option, score it with the same held-out metrics as the
                  PCA / FC / autoencoder heads, and score the only way a NEW
                  sentence can be drawn on it (center of its 384-d neighbors).
Verification:     eval/tsne.json; perplexity chosen on the SIB-200 dev split only

t-SNE has no formula for a new point, so it differs from the three heads in two ways that the page states:
  1. it is fitted on all 4,016 reference sentences, including the 816 the metrics are scored on;
  2. a typed sentence is placed at the similarity-weighted center of its 8 nearest reference sentences.
The new-sentence score therefore comes from a second fit on the training split only, so the scored
sentences were never seen by that fit.

Run:  .venv/bin/python build/build_tsne.py     (about 2 min on CPU; needs web/data/cloud.json)
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.manifold import TSNE, trustworthiness

from build_cloud import metrics
from common import EVAL, SEED, WEB, embed, load_model, write_json
from eval_placement import K_PLACE, TAU, barycenter, overlap

PERPLEXITIES = (15, 30, 50)


def fit(x, perplexity):
    z = TSNE(n_components=3, perplexity=perplexity, metric="cosine", init="pca", learning_rate="auto",
             random_state=SEED, n_jobs=16).fit_transform(x)
    z = z - z.mean(0)
    return z / z.std(0).mean()          # same scale convention as the heads (mean axis std = 1)


def main() -> None:
    meta = json.loads((WEB / "data" / "cloud.json").read_text(encoding="utf-8"))
    q = np.frombuffer((WEB / "data" / "cloud_emb.i8.bin").read_bytes(), dtype=np.int8).reshape(meta["n"], 384)
    x = q.astype(np.float64) * meta["emb_scale"]
    x /= np.linalg.norm(x, axis=1, keepdims=True)       # exactly the vectors the browser searches
    pts = meta["points"]
    lang = np.array([p["l"] for p in pts]); topic = np.array([p["c"] for p in pts])
    ids = np.array([p["i"] for p in pts]); split = np.array([p["s"] for p in pts])
    tr, dv, te = split == 0, split == 1, split == 2
    rng = np.random.default_rng(SEED)

    best, tried = None, {}
    for perp in PERPLEXITIES:
        z = fit(x, perp)
        t10 = float(trustworthiness(x[dv], z[dv], n_neighbors=10, metric="cosine"))
        tried[str(perp)] = t10
        print("perplexity", perp, "dev trust10", round(t10, 4))
        if best is None or t10 > best[0]:
            best = (t10, perp, z)
    _, perp, z = best

    m = metrics(z[te], x[te], lang[te], topic[te], ids[te], z[tr], topic[tr], rng, (z[tr], lang[tr], z[dv], lang[dv]))

    # new-sentence placement: fit on the training split only, then drop held-out sentences in by neighbor center
    z_tr = fit(x[tr], perp)
    js = (WEB / "js" / "main.js").read_text(encoding="utf-8")
    block = js[js.index("const EXAMPLES"):js.index("];", js.index("const EXAMPLES"))]
    examples = [s for trio in re.findall(r"\['[^']*', '([^']*)', '([^']*)'\]", block) for s in trio]
    tok, model = load_model()
    x_ex = embed(tok, model, examples).astype(np.float64)
    place = {
        "k_place": K_PLACE, "tau": TAU, "n_heldout": int(te.sum()), "n_examples": len(examples),
        # the point's own fitted position on the shipped map (it was part of the fit)
        "heldout_overlap10_fitted": overlap(x[te], z[te], x[tr], z[tr]),
        # an unseen sentence placed by neighbor center on a train-only fit (comparable to eval/placement.json)
        "heldout_overlap10_neighbour_centre": overlap(x[te], barycenter(x[te], x[tr], z_tr), x[tr], z_tr),
        "examples_overlap10_neighbour_centre": overlap(x_ex, barycenter(x_ex, x[tr], z_tr), x[tr], z_tr),
    }
    report = {"method": "sklearn TSNE, 3 components, cosine metric, PCA init", "seed": SEED, "perplexity": perp,
              "dev_trust10_by_perplexity": tried, "fitted_on": "all points (train + dev + test)", "n_points": int(len(pts)),
              "metrics": m, "placement": place}
    write_json(EVAL / "tsne.json", report)
    write_json(WEB / "data" / "tsne.json", report | {"coords": np.round(z, 4).ravel().tolist()})
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))
    print(json.dumps(place, indent=1))


if __name__ == "__main__":
    main()
