// The explanation methods, all computed in the browser from forward passes only.
//   target S = cosine(mean-pooled sentence A, mean-pooled sentence B)
//   1. pair decomposition  (exact: the word-pair terms sum to S)
//   2. occlusion           (delete a word, re-run, record the drop)       [needs the engine]
//   3. attention rollout   (Abnar & Zuidema 2020)
//   + deletion test        (does removing the "important" words actually move S?)
// The same formulas are implemented in build/eval_attribution.py, which asserts closed-form == autograd.
// Verification: tests/smoke.mjs

import { H, HEADS, LAYERS, dot, meanPool, norm, tokenVecs, unit } from './engine.js';

/** Token-pair matrix M[t][u] = h_t . h_u / (nA nB |mA| |mB|).  sum(M) == cosine exactly. */
export function pairMatrix(HA, HB) {
  const nA = HA.length, nB = HB.length;
  const denom = nA * nB * norm(meanPool(HA)) * norm(meanPool(HB));
  const M = new Float32Array(nA * nB);
  let total = 0;
  for (let t = 0; t < nA; t++) for (let u = 0; u < nB; u++) { const v = dot(HA[t], HB[u]) / denom; M[t * nB + u] = v; total += v; }
  return { M, nA, nB, total };
}

/** Collapse a token-pair matrix to word groups. Returns words x words plus what is left over. */
export function wordMatrix(pm, groupsA, groupsB) {
  const W = groupsA.map(ga => groupsB.map(gb => {
    let s = 0;
    for (const t of ga.tokens) for (const u of gb.tokens) s += pm.M[t * pm.nB + u];
    return s;
  }));
  const rowTok = new Float32Array(pm.nA), colTok = new Float32Array(pm.nB);
  for (let t = 0; t < pm.nA; t++) for (let u = 0; u < pm.nB; u++) { rowTok[t] += pm.M[t * pm.nB + u]; colTok[u] += pm.M[t * pm.nB + u]; }
  const sumTok = (arr, g) => g.tokens.reduce((s, t) => s + arr[t], 0);
  const rows = groupsA.map(g => sumTok(rowTok, g)), cols = groupsB.map(g => sumTok(colTok, g));
  const content = (gs, vals) => gs.reduce((s, g, i) => s + (g.punct ? 0 : vals[i]), 0);
  return { W, rows, cols, total: pm.total, wordsShareA: content(groupsA, rows), wordsShareB: content(groupsB, cols) };
}

/** Attention rollout for sentence b of a run; relevance of each input token to the mean-pooled output. */
export function rollout(r, b) {
  const n = r.lens[b], T = r.T;
  let R = new Float64Array(n * n);
  for (let i = 0; i < n; i++) R[i * n + i] = 1;
  for (let l = 0; l < LAYERS - 1; l++) {
    const A = new Float64Array(n * n);
    for (let h = 0; h < HEADS; h++) {
      const base = (((b * (LAYERS - 1) + l) * HEADS + h) * T) * T;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] += r.attn[base + i * T + j] / HEADS;
    }
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) A[i * n + j] *= 0.5; A[i * n + i] += 0.5; }
    const next = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const a = A[i * n + k]; if (a) for (let j = 0; j < n; j++) next[i * n + j] += a * R[k * n + j]; }
    R = next;
  }
  const score = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) score[j] += R[i * n + j] / n;
  return score;
}

/** Mean attention (over heads) from every token to every token at one layer: n x n nested array. */
export function attentionAt(r, b, layer, head = -1) {
  const n = r.lens[b], T = r.T, out = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    for (let h = 0; h < HEADS; h++) {
      if (head >= 0 && h !== head) continue;
      const base = (((b * (LAYERS - 1) + layer) * HEADS + h) * T) * T;
      for (let j = 0; j < n; j++) row[j] += r.attn[base + i * T + j] / (head >= 0 ? 1 : HEADS);
    }
    out.push(row);
  }
  return out;
}

export const toWords = (tokenScores, groups) => groups.map(g => g.tokens.reduce((s, t) => s + tokenScores[t], 0));
export const without = (ids, positions) => { const drop = new Set(positions); return ids.filter((_, i) => !drop.has(i)); };

/** Occlusion: similarity drop when each (non-punctuation) word is deleted. Returns per-group scores (punct = 0). */
export async function occlusion(engine, ids, groups, eOther, s0) {
  const idx = groups.map((g, i) => (g.punct ? -1 : i)).filter(i => i >= 0);
  const embs = await engine.embedMany(idx.map(i => without(ids, groups[i].tokens)));
  const scores = new Array(groups.length).fill(0);
  idx.forEach((gi, k) => { scores[gi] = s0 - dot(embs[k], eOther); });
  return scores;
}

function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/**
 * Deletion test for one sentence. For each method, delete its top-k words (k = 1..n) and record the similarity S.
 * A faithful ranking makes the curve fall fast. "random" is the mean of 5 seeded shuffles.
 */
export async function deletionTest(engine, ids, groups, eOther, s0, methodScores) {
  const content = groups.map((g, i) => (g.punct ? -1 : i)).filter(i => i >= 0);
  const n = content.length;
  const orders = {};
  for (const [name, sc] of Object.entries(methodScores)) orders[name] = [[...content].sort((a, b) => sc[b] - sc[a])];
  const rnd = mulberry32(20261115);
  orders.random = Array.from({ length: 5 }, () => { const a = [...content]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; });
  const jobs = [], index = [];
  for (const [name, list] of Object.entries(orders)) list.forEach((order, rep) => {
    for (let k = 1; k <= n; k++) { jobs.push(without(ids, order.slice(0, k).flatMap(gi => groups[gi].tokens))); index.push([name, rep, k]); }
  });
  const embs = await engine.embedMany(jobs);
  const curves = {};
  for (const name of Object.keys(orders)) curves[name] = new Array(n + 1).fill(0);
  index.forEach(([name, , k], i) => { curves[name][k] += dot(embs[i], eOther) / orders[name].length; });
  for (const name of Object.keys(orders)) curves[name][0] = s0;
  const area = Object.fromEntries(Object.entries(curves).map(([k, c]) => [k, c.slice(1).reduce((s, v) => s + v, 0) / n]));
  return { curves, area, n, orders: Object.fromEntries(Object.entries(orders).map(([k, v]) => [k, v[0]])) };
}

/** Everything that needs only the first forward pass (both sentences in one batch). */
export function analyzePair(r, groupsA, groupsB) {
  const last = LAYERS - 1;
  const HA = tokenVecs(r, 0, last), HB = tokenVecs(r, 1, last);
  const eA = unit(meanPool(HA)), eB = unit(meanPool(HB));
  const sim = dot(eA, eB);
  const pm = pairMatrix(HA, HB);
  const wm = wordMatrix(pm, groupsA, groupsB);
  const layerSim = [], layerVec = { A: [], B: [] };
  for (let l = 0; l < LAYERS; l++) {
    const a = unit(meanPool(tokenVecs(r, 0, l))), b = unit(meanPool(tokenVecs(r, 1, l)));
    layerVec.A.push(a); layerVec.B.push(b); layerSim.push(dot(a, b));
  }
  const wordVec = (b, groups) => Array.from({ length: LAYERS }, (_, l) => {
    const tv = tokenVecs(r, b, l);
    return groups.map(g => unit(meanPool(g.tokens.map(t => tv[t]))));
  });
  return {
    sim, eA, eB, wm, layerSim, layerVec,
    wordVecA: wordVec(0, groupsA), wordVecB: wordVec(1, groupsB),
    scores: {
      pair: { A: wm.rows, B: wm.cols },
      rollout: { A: toWords(rollout(r, 0), groupsA), B: toWords(rollout(r, 1), groupsB) },
    },
  };
}
