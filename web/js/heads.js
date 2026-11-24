// The 384-d -> 3-d projection heads (PCA, linear FC, autoencoder), cloud loading, neighbor search.
//       Mirrors build/build_cloud.py apply_head(). Neighbors are ALWAYS searched in 384-d, never in 3-d.
//       The optional t-SNE map has no formula: its cloud coordinates are precomputed (build/build_tsne.py) and a new
//       vector is drawn at the similarity-weighted center of its nearest reference sentences (K_PLACE, TAU as in
//       build/eval_placement.py).
// Verification: tests/check_heads.py (browser coordinates vs Python apply_head)

const H = 384, K_PLACE = 8, TAU = 0.05;

function erf(x) { // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
const gelu = v => 0.5 * v * (1 + erf(v / Math.SQRT2));

function prep(head) {
  const flat = m => Float32Array.from(m.flat());
  const p = { mean: Float32Array.from(head.mean), W: flat(head.W) };
  if (head.W1) Object.assign(p, { W1: flat(head.W1), b1: Float32Array.from(head.b1), W2: flat(head.W2), b2: Float32Array.from(head.b2), hid: head.b1.length });
  return p;
}

/** v: Float32Array(384) -> [x, y, z] */
export function applyHead(p, v, off = 0) {
  const xc = new Float32Array(H);
  for (let k = 0; k < H; k++) xc[k] = v[off + k] - p.mean[k];
  const z = [0, 0, 0];
  for (let d = 0; d < 3; d++) { let s = 0; const o = d * H; for (let k = 0; k < H; k++) s += p.W[o + k] * xc[k]; z[d] = s; }
  if (p.W1) {
    for (let j = 0; j < p.hid; j++) {
      let s = p.b1[j]; const o = j * H;
      for (let k = 0; k < H; k++) s += p.W1[o + k] * xc[k];
      const a = gelu(s);
      for (let d = 0; d < 3; d++) z[d] += p.W2[d * p.hid + j] * a;
    }
    for (let d = 0; d < 3; d++) z[d] += p.b2[d];
  }
  return z;
}

export async function loadCloud(base = './data/', { tsne: wantTsne = true } = {}) {
  const [meta, buf, tsne] = await Promise.all([
    fetch(base + 'cloud.json').then(r => r.json()),
    fetch(base + 'cloud_emb.i8.bin').then(r => r.arrayBuffer()),
    wantTsne ? fetch(base + 'tsne.json').then(r => (r.ok ? r.json() : null)).catch(() => null) : null,   // optional
  ]);
  const q = new Int8Array(buf), n = meta.n;
  const emb = new Float32Array(n * H);
  for (let i = 0; i < n; i++) {           // de-quantize and re-normalize each row to unit length
    let s = 0; const o = i * H;
    for (let k = 0; k < H; k++) { const v = q[o + k] * meta.emb_scale; emb[o + k] = v; s += v * v; }
    s = Math.sqrt(s) || 1;
    for (let k = 0; k < H; k++) emb[o + k] /= s;
  }
  const heads = Object.fromEntries(Object.entries(meta.heads).map(([k, h]) => [k, prep(h)]));
  const layerMeans = meta.layer_means ? meta.layer_means.map(m => Float32Array.from(m)) : null;
  const coordCache = {};
  if (tsne && tsne.coords.length === n * 3) { coordCache.tsne = Float32Array.from(tsne.coords); meta.metrics.tsne = tsne.metrics; }
  const placed = new WeakMap();   // vector -> {layer: [x,y,z]} for neighbor-center placement (one 384-d search each)
  /** 3-d coordinates of every cloud point under one head (cached). */
  function coords(name) {
    if (!coordCache[name]) {
      const c = new Float32Array(n * 3), p = heads[name];
      for (let i = 0; i < n; i++) { const z = applyHead(p, emb, i * H); c[i * 3] = z[0]; c[i * 3 + 1] = z[1]; c[i * 3 + 2] = z[2]; }
      coordCache[name] = c;
    }
    return coordCache[name];
  }
  /** Project a unit vector taken at `layer` (0..12). Earlier layers are re-centered onto the layer-12 map. */
  function project(name, v, layer = 12) {
    let x = v;
    if (layer !== 12 && layerMeans) {
      x = new Float32Array(H);
      for (let k = 0; k < H; k++) x[k] = v[k] - layerMeans[layer][k] + layerMeans[12][k];
    }
    if (!heads[name]) {            // a map with no formula (t-SNE): center of the nearest reference sentences
      let memo = placed.get(v); if (!memo) placed.set(v, memo = {});
      const key = name + ':' + layer;
      return memo[key] || (memo[key] = neighborCenter(name, x));
    }
    return applyHead(heads[name], x);
  }
  function neighborCenter(name, x) {
    let s = 0; for (let k = 0; k < H; k++) s += x[k] * x[k];
    s = Math.sqrt(s) || 1;
    const u = new Float32Array(H); for (let k = 0; k < H; k++) u[k] = x[k] / s;
    const nb = neighbors(u, K_PLACE), c = coords(name), z = [0, 0, 0];
    let wsum = 0;
    for (const d of nb) { const w = Math.exp((d.sim - nb[0].sim) / TAU); wsum += w; for (let a = 0; a < 3; a++) z[a] += w * c[d.index * 3 + a]; }
    return z.map(v => v / wsum);
  }
  /** Top-k cloud sentences by cosine in the full 384-d space. mask: optional Uint8Array(n), 0 = skip that point. */
  function neighbors(v, k = 8, mask = null) {
    const best = [];
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      let s = 0; const o = i * H;
      for (let d = 0; d < H; d++) s += emb[o + d] * v[d];
      if (best.length < k) { best.push([s, i]); best.sort((a, b) => b[0] - a[0]); }
      else if (s > best[k - 1][0]) { best[k - 1] = [s, i]; best.sort((a, b) => b[0] - a[0]); }
    }
    return best.map(([sim, index]) => ({ sim, index }));
  }
  /** How much of a 384-d neighborhood survives on the 3-d map, for one query point (share of top-k also top-k in 3-d). */
  function localOverlap(name, v, pos, k = 10) {
    const hi = new Set(neighbors(v, k).map(d => d.index));
    const c = coords(name), d2 = new Float32Array(n);
    for (let i = 0; i < n; i++) d2[i] = (c[i * 3] - pos[0]) ** 2 + (c[i * 3 + 1] - pos[1]) ** 2 + (c[i * 3 + 2] - pos[2]) ** 2;
    const idx = Array.from(d2.keys()).sort((a, b) => d2[a] - d2[b]).slice(0, k);
    return idx.filter(i => hi.has(i)).length / k;
  }
  // the reference set is parallel: every sentence id exists once per language
  const byId = new Map();
  meta.points.forEach((p, i) => { if (!byId.has(p.i)) byId.set(p.i, []); byId.get(p.i)[p.l] = i; });
  /** Index of the same sentence in another language (or -1). */
  const sameSentence = (index, lang) => byId.get(meta.points[index].i)?.[lang] ?? -1;
  return { meta, n, emb, coords, project, neighbors, localOverlap, sameSentence, tsne, hasMap: name => !!(heads[name] || coordCache[name]), hasLayerMeans: !!layerMeans };
}
