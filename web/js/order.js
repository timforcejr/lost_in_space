// The live word-order test.  Scramble the words of a sentence (token ids are moved, nothing is re-tokenized),
//       re-run the model, and follow two things layer by layer: how far the pooled sentence vector moves, and what a
//       linear probe trained offline (build/eval_order.py) reads from it ("natural order" vs "scrambled").
//       units() and reorder() MUST mirror build/eval_order.py.
// Verification: tests/smoke.mjs

import { H, LAYERS, dot, meanPool, tokenVecs, unit } from './engine.js';
import { wordGroups } from './tokens.js';

/** Movable units: one per display word, as token positions; a stray '▁' piece travels with the word after it. */
export function units(tokens) {
  const groups = wordGroups(tokens), owner = new Map();
  groups.forEach((g, gi) => g.tokens.forEach(p => owner.set(p, gi)));
  const out = groups.map(() => []);
  let pending = [];
  for (let p = 1; p < tokens.length - 1; p++) {
    if (owner.has(p)) { out[owner.get(p)].push(...pending, p); pending = []; } else pending.push(p);
  }
  if (pending.length && out.length) out[out.length - 1].push(...pending);
  return { units: out, movable: groups.map(g => !g.punct) };
}

/** Token positions with the movable units put in the order `perm`; punctuation keeps its slot. */
export function reorder(us, movable, perm) {
  const slots = movable.map((m, i) => (m ? i : -1)).filter(i => i >= 0), order = us.map((_, i) => i);
  slots.forEach((slot, k) => { order[slot] = slots[perm[k]]; });
  return order.flatMap(u => us[u]);
}

function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const sigmoid = z => 1 / (1 + Math.exp(-z));

/**
 * seq: {ids, tokens} of one sentence; selfLayers: its 13 unit pooled vectors; eOther: unit embedding of the other sentence.
 * probes: [{w:number[384], b}] per layer, or null.  Returns null when the sentence has fewer than 3 words.
 */
export async function orderTest(engine, seq, selfLayers, eOther, probes, seed = 1, nShuffles = 5) {
  const { units: us, movable } = units(seq.tokens), n = movable.filter(Boolean).length;
  if (n < 3) return null;
  const rnd = mulberry32(20261115 + seed), identity = Array.from({ length: n }, (_, i) => i), perms = [];
  for (let tries = 0; perms.length < nShuffles && tries < 200; tries++) {
    const p = [...identity];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
    if (p.some((v, i) => v !== i) && !perms.some(q => q.join() === p.join())) perms.push(p);
  }
  const variants = perms.map(p => ({ kind: 'shuffle', pos: reorder(us, movable, p) }));
  variants.push({ kind: 'reverse', pos: reorder(us, movable, [...identity].reverse()) });
  const last = seq.ids.length - 1;
  for (const v of variants) {
    v.ids = [seq.ids[0], ...v.pos.map(p => seq.ids[p]), seq.ids[last]];
    v.text = v.pos.map(p => seq.tokens[p]).join('').replaceAll('▁', ' ').trim();
  }
  const r = await engine.run(variants.map(v => v.ids));
  const prob = (l, vec) => (probes ? sigmoid(dot(probes[l].w, vec) + probes[l].b) : null);
  variants.forEach((v, b) => {
    v.layers = Array.from({ length: LAYERS }, (_, l) => unit(meanPool(tokenVecs(r, b, l))));
    v.cos = v.layers.map((vec, l) => dot(vec, selfLayers[l]));          // similarity to the original sentence, per layer
    v.simOther = dot(v.layers[LAYERS - 1], eOther);                      // final score against the other sentence
    v.probe = v.layers.map((vec, l) => prob(l, vec));
  });
  const sh = variants.filter(v => v.kind === 'shuffle'), mean = f => Array.from({ length: LAYERS }, (_, l) => sh.reduce((s, v) => s + f(v, l), 0) / sh.length);
  return { n, variants, cosShuffle: mean((v, l) => v.cos[l]), cosReverse: variants[variants.length - 1].cos,
    probeNatural: selfLayers.map((vec, l) => prob(l, vec)), probeShuffle: probes ? mean((v, l) => v.probe[l]) : null,
    simOtherShuffle: sh.reduce((s, v) => s + v.simOther, 0) / sh.length, seed };
}
