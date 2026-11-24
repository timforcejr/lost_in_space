// Gradient-based explanations in the browser.  onnxruntime-web cannot backpropagate, so build/export_grad.py
//       exports the encoder's forward AND hand-written backward pass as one ordinary graph (web/model/mgrad).
//       One inference call returns d(similarity)/d(hidden state) at every layer.  From that this file computes
//         - integrated gradients  (layer-0 path from an all-<pad> baseline, 24 midpoint steps),
//         - gradient x input.
//       Same definitions as build/eval_attribution.py.
// Verification: eval/grad_parity.json (graph vs autograd), tests/check_grad.py (browser vs PyTorch)

import { AutoModel, Tensor } from '../vendor/transformers/transformers.min.js';
import { H, LAYERS, locked, currentRun, checkRun } from './engine.js';

export const IG_STEPS = 24;
const CHUNK = 8;                                   // path points per inference call (keeps memory flat and lets the UI show progress)
let model = null, loading = null;

/** Download and start the gradient model (41 MB, once). */
export function loadGrad(onProgress = () => {}, dir = 'mgrad') {
  loading ??= AutoModel.from_pretrained(dir, { dtype: 'fp32', device: 'wasm', progress_callback: onProgress }).then(m => (model = m), e => { loading = null; throw e; });
  return loading;
}

async function call(h0, S, T, eOther) {
  const out = await locked(() => model({ h0: new Tensor('float32', h0, [S, T, H]), e_other: new Tensor('float32', eOther, [H]) }));
  return { sim: out.sim.data, hidden: out.hidden.data, grad: out.grad.data };
}

/**
 * x, base: arrays of T Float32Array(384), the layer-0 states of the sentence and of its blank (<pad>) baseline.
 * eOther: unit embedding of the other sentence.  Returns token-level scores.
 */
export async function explainGrad(x, base, eOther, onStep = () => {}) {
  const T = x.length, TH = T * H, run = currentRun();
  const both = new Float32Array(2 * TH);
  x.forEach((v, t) => both.set(v, t * H)); base.forEach((v, t) => both.set(v, TH + t * H));
  const first = await call(both, 2, T, eOther);                       // path point 0 = the sentence, 1 = the baseline
  const off = (l, t) => (l * T + t) * H;                              // path point 0

  const gradInput = Array.from({ length: T }, (_, t) => { const o = off(0, t); let s = 0; for (let k = 0; k < H; k++) s += first.grad[o + k] * first.hidden[o + k]; return s; });

  // integrated gradients: average the layer-0 gradient along the straight path baseline -> sentence
  const acc = new Float64Array(TH), diff = new Float32Array(TH);
  for (let t = 0; t < T; t++) for (let k = 0; k < H; k++) diff[t * H + k] = x[t][k] - base[t][k];
  for (let k0 = 0; k0 < IG_STEPS; k0 += CHUNK) {
    const S = Math.min(CHUNK, IG_STEPS - k0), path = new Float32Array(S * TH);
    for (let s = 0; s < S; s++) { const a = (k0 + s + 0.5) / IG_STEPS; for (let i = 0; i < TH; i++) path[s * TH + i] = both[TH + i] + a * diff[i]; }
    checkRun(run);
    const out = await call(path, S, T, eOther);
    for (let s = 0; s < S; s++) { const o = s * LAYERS * TH; for (let i = 0; i < TH; i++) acc[i] += out.grad[o + i]; }
    onStep(Math.min(1, (k0 + S) / IG_STEPS));
  }
  const ig = Array.from({ length: T }, (_, t) => { let s = 0; for (let k = 0; k < H; k++) s += diff[t * H + k] * acc[t * H + k]; return s / IG_STEPS; });
  return { sim: first.sim[0], simBase: first.sim[1], gradInput, ig, igSum: ig.reduce((s, v) => s + v, 0), igTarget: first.sim[0] - first.sim[1] };
}
