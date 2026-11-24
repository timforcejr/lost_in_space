// Load the ONNX encoder in the browser (transformers.js / onnxruntime-web, WASM) and run padded
//       batches that return every layer's hidden states and attention maps. Nothing leaves the page.
// Verification: tests/smoke.mjs (parity with build/export_model.py)

import { AutoTokenizer, AutoModel, Tensor, env } from '../vendor/transformers/transformers.min.js';

export const H = 384, LAYERS = 13, HEADS = 12, MAX_TOKENS = 64;

// onnxruntime-web (WASM) runs one inference at a time; every model call on the page goes through this queue.
// After each call the queue hands control back to the browser (a macrotask), so the page can paint progress and take
// input between calls; without that, a long sentence freezes the tab for the whole chain of tests.
let queue = Promise.resolve(), generation = 0;
const breathe = v => new Promise(r => { const c = new MessageChannel(); c.port1.onmessage = () => r(v); c.port2.postMessage(0); });
export function locked(fn) { const p = queue.then(fn, fn).then(breathe); queue = p.catch(() => {}); return p; }
/** A new pair was submitted: multi-call jobs started for the previous pair stop at their next call. */
export const supersede = () => ++generation;
export const currentRun = () => generation;
export const SUPERSEDED = 'superseded';
export function checkRun(g) { if (g !== generation) throw new Error(SUPERSEDED); }

const STALL_MS = 60000;
/** Load tokenizer + model. Rejects if no progress event arrives for 60 s, or an unhandled rejection fires meanwhile. */
export function loadEngine(onProgress = () => {}, dirs = {}) {
  let timer = 0, fail = () => {}, arm = () => {};
  const stalled = new Promise((_, rej) => {
    arm = () => { clearTimeout(timer); timer = setTimeout(() => rej(new Error('model download stalled (no progress for 60 s)')), STALL_MS); };
    const onRej = e => { e.preventDefault(); rej(e.reason instanceof Error ? e.reason : new Error(String(e.reason))); };
    addEventListener('unhandledrejection', onRej);
    fail = () => { clearTimeout(timer); removeEventListener('unhandledrejection', onRej); };
    arm();
  });
  stalled.catch(() => {});
  const work = loadEngineInner(p => { arm(); onProgress(p); }, dirs);
  return Promise.race([work, stalled]).finally(fail);
}

async function loadEngineInner(onProgress, { modelDir = 'mmini', tokenizerDir = 'mmini' } = {}) {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = new URL('../model/', import.meta.url).href;
  env.backends.onnx.wasm.wasmPaths = new URL('../vendor/transformers/', import.meta.url).href;
  // GitHub Pages cannot send the cross-origin-isolation headers that WASM threads need, so plan for 1 thread.
  env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  const tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir);
  // the 96 MB word-embedding table lives in model_quantized.onnx_data so that no file exceeds GitHub's 100 MB limit
  const model = await AutoModel.from_pretrained(modelDir, { dtype: 'q8', device: 'wasm', use_external_data_format: true, progress_callback: onProgress });
  const PAD = tokenizer.pad_token_id ?? 1;

  /** text -> {ids:number[], tokens:string[], truncated:boolean} with <s> ... </s>, cut to MAX_TOKENS. */
  function encode(text) {
    // a typed "<s>", "</s>", "<pad>", "<unk>" or "<mask>" must stay text, not become the model's control token
    let ids = Array.from(tokenizer.encode(text.replace(/<(?=\/?(?:s|pad|unk|mask)>)/gi, '< ')), Number);
    const truncated = ids.length > MAX_TOKENS;
    if (truncated) ids = [...ids.slice(0, MAX_TOKENS - 1), ids[ids.length - 1]];
    return { ids, tokens: tokenizer.model.convert_ids_to_tokens(ids), truncated };
  }

  /** One padded forward pass. Returns flat arrays: hidden [B,13,T,384], attn [B,12,12,T,T]. */
  async function run(idLists) {
    const B = idLists.length, lens = idLists.map(a => a.length), T = Math.max(...lens);
    const ids = new BigInt64Array(B * T).fill(BigInt(PAD));
    const mask = new BigInt64Array(B * T);
    idLists.forEach((seq, b) => seq.forEach((id, t) => { ids[b * T + t] = BigInt(id); mask[b * T + t] = 1n; }));
    const out = await locked(() => model({ input_ids: new Tensor('int64', ids, [B, T]), attention_mask: new Tensor('int64', mask, [B, T]) }));
    return { hidden: out.hidden.data, attn: out.attn.data, B, T, lens };
  }

  /** Final-layer mean-pooled unit embeddings for many sequences (chunked to bound memory). */
  async function embedMany(idLists, chunk = 24) {
    const result = [], g = generation;
    for (let i = 0; i < idLists.length; i += chunk) {
      checkRun(g);
      const r = await run(idLists.slice(i, i + chunk));
      for (let b = 0; b < r.B; b++) result.push(unit(meanPool(tokenVecs(r, b, LAYERS - 1))));
    }
    return result;
  }

  return { encode, run, embedMany, padId: PAD, threads: env.backends.onnx.wasm.numThreads };
}

// ---- small vector helpers shared with xai.js ----
export function tokenVecs(r, b, layer) {
  const out = [];
  for (let t = 0; t < r.lens[b]; t++) {
    const o = ((b * LAYERS + layer) * r.T + t) * H;
    out.push(r.hidden.subarray(o, o + H));
  }
  return out;
}
export function meanPool(vecs) {
  const m = new Float32Array(H);
  for (const v of vecs) for (let k = 0; k < H; k++) m[k] += v[k];
  for (let k = 0; k < H; k++) m[k] /= vecs.length;
  return m;
}
export const dot = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
export const norm = a => Math.sqrt(dot(a, a));
export function unit(a) { const n = norm(a) || 1; const o = new Float32Array(a.length); for (let k = 0; k < a.length; k++) o[k] = a[k] / n; return o; }
