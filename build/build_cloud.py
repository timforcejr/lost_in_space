"""
Purpose:          Embed the SIB-200 reference sentences (en/es/hi/zh), fit
                  three 384-d -> 3-d heads (PCA, linear FC, autoencoder),
                  measure how faithful each 3-d map is, and write the static
                  assets the browser app loads.
Verification:     Metrics on the held-out SIB-200 test split written to
                  eval/map_fidelity.json (no number is tuned on that split)

Run:  .venv/bin/python build/build_cloud.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness

from common import EVAL, LANG_NAMES, LANGS, MODEL_ID, SEED, WEB, embed, load_model, load_sib, set_seed, write_json

LANG_LIST = list(LANGS)


# ----------------------------------------------------------------------------- heads
def fit_pca(x):
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return {"mean": mu, "W": vt[:3]}


def neighbor_kl(x_hi, z, tau=0.07):
    """Parametric-SNE style loss: 384-d cosine neighborhoods should survive in 3-d."""
    n = x_hi.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    p = F.softmax((x_hi @ x_hi.T / tau).masked_fill(eye, -1e9), dim=1)
    d2 = torch.cdist(z, z).pow(2)
    logq = F.log_softmax((-torch.log1p(d2)).masked_fill(eye, -1e9), dim=1)
    return (p * (torch.log(p + 1e-12) - logq)).sum(1).mean()


def train_head(kind, x_tr, pair_idx, pca, tau=0.07, steps=2000, lam_align=0.02, lam_rec=0.5, seed=SEED):
    """Both heads start AT the PCA map and are then trained, so any change is the objective's doing.

    kind='fc' -> linear 384->3.
    kind='ae' -> PCA skip + MLP 384->256->3 encoder, with a 3->256->384 decoder (reconstruction loss).
    """
    set_seed(seed)
    x = torch.tensor(x_tr, dtype=torch.float32)
    mu = torch.tensor(pca["mean"], dtype=torch.float32)[None]
    xc = x - mu
    lin = torch.nn.Linear(384, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(pca["W"], dtype=torch.float32))
    if kind == "fc":
        mlp = dec = None
        enc = lin
        params = list(lin.parameters())
    else:
        mlp = torch.nn.Sequential(torch.nn.Linear(384, 256), torch.nn.GELU(), torch.nn.Linear(256, 3))
        torch.nn.init.zeros_(mlp[2].weight); torch.nn.init.zeros_(mlp[2].bias)
        dec = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.GELU(), torch.nn.Linear(256, 384))
        enc = lambda v: lin(v) + mlp(v)
        params = list(lin.parameters()) + list(mlp.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    a, b = (torch.tensor(pair_idx[:, 0]), torch.tensor(pair_idx[:, 1]))
    gen = torch.Generator().manual_seed(seed)
    log = []
    for step in range(steps):
        idx = torch.randperm(x.shape[0], generator=gen)[:1400]
        z = enc(xc[idx])
        loss_n = neighbor_kl(x[idx], z, tau)
        za, zb = enc(xc[a]), enc(xc[b])
        scale = z.std(0).mean().detach() + 1e-6
        loss_a = ((za - zb) / scale).pow(2).sum(1).mean()
        loss = loss_n + lam_align * loss_a
        if dec is not None:
            loss = loss + lam_rec * (1 - F.cosine_similarity(dec(z), x[idx], dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            log.append({"step": step, "loss": loss.item(), "neigh": loss_n.item(), "align": loss_a.item()})
    with torch.no_grad():
        s = float(enc(xc).std(0).mean())
    head = {"mean": mu[0].numpy(), "W": lin.weight.detach().numpy() / s}
    if kind == "ae":
        head.update(W1=mlp[0].weight.detach().numpy(), b1=mlp[0].bias.detach().numpy(),
                    W2=mlp[2].weight.detach().numpy() / s, b2=mlp[2].bias.detach().numpy() / s)
    return head, log


def gelu(v):
    from scipy.special import erf
    return 0.5 * v * (1 + erf(v / np.sqrt(2)))


def apply_head(head, x):
    xc = x - head["mean"]
    z = xc @ head["W"].T
    if "W1" in head:
        z = z + gelu(xc @ head["W1"].T + head["b1"]) @ head["W2"].T + head["b2"]
    return z


# ----------------------------------------------------------------------------- metrics
C_GRID = (0.01, 0.1, 1.0, 10.0)


def language_probe(z_tr, l_tr, z_dv, l_dv, z_te, l_te):
    """Standardized logistic regression trained on the train split, C chosen on dev, accuracy on test."""
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-8
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000).fit((z_tr - mu) / sd, l_tr)
        acc = clf.score((z_dv - mu) / sd, l_dv)
        if best is None or acc > best[0]:
            best = (acc, clf)
    return float(best[1].score((z_te - mu) / sd, l_te))


def metrics(z_te, x_te, lang_te, topic_te, id_te, z_tr, topic_tr, rng, probe_split):
    """All on the held-out test split.  z_* are coordinates in the space being scored."""
    out = {}
    is_hi = z_te.shape[1] > 3
    norm = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

    def dist(a, b):
        if is_hi:
            return 1 - norm(a) @ norm(b).T
        return np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))

    # 1. translation retrieval: en sentence -> same-id sentence in the other language (204 candidates)
    en = np.where(lang_te == 0)[0]
    p1 = {}
    for li, lang in enumerate(LANG_LIST[1:], 1):
        other = np.where(lang_te == li)[0]
        d = dist(z_te[en], z_te[other])
        p1[lang] = float(np.mean(id_te[other][d.argmin(1)] == id_te[en]))
    out["retrieval_p1_en_to"] = p1
    out["retrieval_p1_mean"] = float(np.mean(list(p1.values())))

    # 2. pair ratio: median distance to own translation / median distance to a random other-language sentence
    d_all = dist(z_te, z_te)
    same_id = id_te[:, None] == id_te[None]
    cross = lang_te[:, None] != lang_te[None]
    out["pair_ratio"] = float(np.median(d_all[same_id & cross]) / np.median(d_all[~same_id & cross]))

    # 3. topic kNN (k=10): label each test point from the training cloud
    d_tr = dist(z_te, z_tr)
    nn = np.argsort(d_tr, 1)[:, :10]
    pred = np.array([np.bincount(topic_tr[r], minlength=topic_tr.max() + 1).argmax() for r in nn])
    out["topic_knn10_acc"] = float(np.mean(pred == topic_te))

    # 4. language probe: can a linear model tell the language from the coordinates? (chance 0.25; lower = more shared)
    # trained on the train split, C chosen on dev, scored on test (probe_split = z_tr, l_tr, z_dv, l_dv)
    out["language_probe_acc"] = language_probe(*probe_split, z_te, lang_te)

    # 5. trustworthiness@10 of the 3-d map against the 384-d space
    if not is_hi:
        out["trust10"] = float(trustworthiness(x_te, z_te, n_neighbors=10, metric="cosine"))
        out["continuity10"] = float(trustworthiness(z_te, x_te, n_neighbors=10, metric="euclidean"))
    return out


def main() -> None:
    set_seed()
    torch.set_num_threads(16)
    rng = np.random.default_rng(SEED)
    tok, model = load_model()
    splits = {s: load_sib(s) for s in ("train", "dev", "test")}
    topics = sorted({r["topic"] for r in splits["train"]["en"]})

    pts, texts = [], []
    for s, data in splits.items():
        for li, lang in enumerate(LANG_LIST):
            for r in data[lang]:
                pts.append({"t": r["text"], "l": li, "c": topics.index(r["topic"]), "i": r["id"], "s": s})
                texts.append(r["text"])
    print(f"embedding {len(texts)} sentences ...")
    x, layers = embed(tok, model, texts, layers=True)
    lang = np.array([p["l"] for p in pts]); topic = np.array([p["c"] for p in pts])
    ids = np.array([p["i"] for p in pts]); split = np.array([p["s"] for p in pts])
    tr, te = split == "train", split == "test"

    # translation pairs inside the training split (en <-> each other language)
    pos = {(int(i), int(l)): k for k, (i, l, s) in enumerate(zip(ids[tr], lang[tr], split[tr]))}
    pair_idx = np.array([[pos[(i, 0)], pos[(i, l)]] for i in sorted(set(ids[tr])) for l in (1, 2, 3)])

    heads, logs = {"pca": fit_pca(x[tr])}, {}
    s = apply_head(heads["pca"], x[tr]).std(0).mean()
    heads["pca"]["W"] = heads["pca"]["W"] / s
    # temperature picked on the DEV split (trustworthiness@10); the test split is never used for selection
    dv = split == "dev"
    for kind in ("fc", "ae"):
        best = None
        for tau in (0.05, 0.1, 0.2):
            head, log = train_head(kind, x[tr], pair_idx, heads["pca"], tau=tau)
            t10 = float(trustworthiness(x[dv], apply_head(head, x[dv]), n_neighbors=10, metric="cosine"))
            print(kind, "tau", tau, "dev trust10", round(t10, 4), log[-1])
            if best is None or t10 > best[0]:
                best = (t10, tau, head, log)
        heads[kind], logs[kind] = best[2], {"tau": best[1], "dev_trust10": best[0], "log": best[3]}
    logs["pca_dev_trust10"] = float(trustworthiness(x[dv], apply_head(heads["pca"], x[dv]), n_neighbors=10, metric="cosine"))

    report = {"model": MODEL_ID, "dataset": "Davlan/sib200 (CC BY-SA 4.0), eng_Latn/spa_Latn/hin_Deva/zho_Hans",
              "n_points": len(pts), "n_test": int(te.sum()), "seed": SEED,
              "spaces": {"full_384d": metrics(x[te], x[te], lang[te], topic[te], ids[te], x[tr], topic[tr], rng, (x[tr], lang[tr], x[dv], lang[dv]))}}
    for name, head in heads.items():
        report["spaces"][name] = metrics(apply_head(head, x[te]), x[te], lang[te], topic[te], ids[te],
                                         apply_head(head, x[tr]), topic[tr], rng,
                                         (apply_head(head, x[tr]), lang[tr], apply_head(head, x[dv]), lang[dv]))
    # per-language mean-centering variant of the full space (per-language mean-centering, an idea reused from the group NLP class project), for reference
    xc = x.copy()
    for li in range(4):
        xc[lang == li] -= x[tr & (lang == li)].mean(0)
    report["spaces"]["full_384d_lang_centred"] = metrics(xc[te], xc[te], lang[te], topic[te], ids[te], xc[tr], topic[tr], rng,
                                                       (xc[tr], lang[tr], xc[dv], lang[dv]))

    # layer lens: at which layer do translations converge?
    lens = []
    lt, it, gt = layers[te], ids[te], lang[te]
    same = (it[:, None] == it[None]) & (gt[:, None] != gt[None])
    diff = (it[:, None] != it[None]) & (gt[:, None] != gt[None])
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    for L in range(layers.shape[1]):
        e = lt[:, L] / np.linalg.norm(lt[:, L], axis=1, keepdims=True)
        sim = e @ e.T
        lens.append({"layer": L, "translation_cos": float(sim[same].mean()), "random_cos": float(sim[diff].mean()),
                     "language_probe_acc": language_probe(unit(layers[tr][:, L]), lang[tr], unit(layers[dv][:, L]), lang[dv], e, gt)})
    report["layer_lens"] = lens
    # per-layer mean of the unit-normalized pooled vectors (training split): lets the viewer re-center a layer-L
    # vector onto the layer-12 map ("layer lens").  Reference similarity distributions for the gauge.
    ltr = layers[tr] / np.linalg.norm(layers[tr], axis=2, keepdims=True)
    layer_means = ltr.mean(0)
    e12 = x[te]
    sim12 = e12 @ e12.T
    qs = [5, 25, 50, 75, 95]
    sim_ref = {"translation": np.percentile(sim12[same], qs).round(4).tolist(),
               "unrelated": np.percentile(sim12[diff], qs).round(4).tolist(), "quantiles": qs}
    report["sim_ref"] = sim_ref
    report["train_logs"] = logs
    write_json(EVAL / "map_fidelity.json", report)

    # ------------------------------------------------------------------ browser assets
    q = np.clip(np.round(x * 127 / np.abs(x).max()), -127, 127).astype(np.int8)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "data" / "cloud_emb.i8.bin").write_bytes(q.tobytes())
    r4 = lambda a: np.round(np.asarray(a, dtype=np.float64), 5).tolist()
    write_json(WEB / "data" / "cloud.json", {
        "model": MODEL_ID, "dim": 384, "n": len(pts), "emb_scale": float(np.abs(x).max() / 127),
        "langs": [{"code": c, "name": LANG_NAMES[c]} for c in LANG_LIST], "topics": topics,
        "points": [{k: p[k] for k in ("t", "l", "c", "i")} | {"s": 0 if p["s"] == "train" else 1 if p["s"] == "dev" else 2} for p in pts],
        "heads": {k: {kk: r4(vv) for kk, vv in h.items()} for k, h in heads.items()},
        "metrics": report["spaces"], "layer_lens": lens, "layer_means": r4(layer_means), "sim_ref": sim_ref,
        "head_info": {k: {"tau": v.get("tau"), "dev_trust10": v.get("dev_trust10")} for k, v in logs.items() if isinstance(v, dict)},
    })
    for k, v in report["spaces"].items():
        print(f"{k:26s}", json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'retrieval_p1_en_to'}))
    print("layer lens:", [(d["layer"], round(d["translation_cos"], 2), round(d["random_cos"], 2), round(d["language_probe_acc"], 2)) for d in lens])


if __name__ == "__main__":
    main()
