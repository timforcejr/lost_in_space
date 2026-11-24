// Title: models.js, the model registry and the "which model is loaded" choice.
// What it does: lists the models the page can load (up to three), with the files each one reads and a flag per feature, so a card
// whose data does not exist for the loaded model shows an honest label instead of another model's numbers. The choice is read from
// ?model=a|b (wins, and is saved), then localStorage 'lis.model', then the default (A). Picking a model saves it and reloads the page;
// only one model is ever in memory per window. If a non-default model cannot load, the page goes back to A with ?model=a&fallback=<id>.

const KEY = 'lis.model';
export const DEFAULT_ID = 'a';

/** The registry. `features`: what the loaded model has data for. `wording`: per-model text rules (plan section 3). */
export const MODELS = [
  {
    id: 'a', enabled: true,
    label: 'Translation-trained MiniLM (default)', short: 'Translation-trained MiniLM',
    descriptor: 'trained on translations; this is the model on the poster.',
    hf: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    modelDir: 'mmini', tokenizerDir: 'mmini', gradDir: 'mgrad', dataDir: './data/', prefix: '',
    loadingName: 'model', sizeNote: '', titleSuffix: '',
    features: { ig: true, tsne: true, placement: true, orderProbe: true, bench: true, sanity: true },
    wording: { overlapVerdict: false, lensNote: 'default', mapPullNote: false, defaultHead: 'ae' },
    lensCompare: null,
  },
  {
    id: 'b', enabled: true,
    label: 'Masked-word MiniLM', short: 'Masked-word MiniLM',
    descriptor: 'trained on masked words only, never on translations.',
    hf: 'microsoft/Multilingual-MiniLM-L12-H384',
    // assets in web/model/mlm, web/model/mgrad_mlm, web/data/mlm; parity in eval/variants/mlm/); if they are missing, picking B falls back to A with a clear note.
    // The offline benchmark and sanity check were run on the default model only, so those two cards stay labeled; IG on B is live but not benchmarked (igNote).
    modelDir: 'mlm', tokenizerDir: 'mmini', gradDir: 'mgrad_mlm', dataDir: './data/mlm/', prefix: '',
    loadingName: 'masked-word model', sizeNote: '113 MB, first time only', titleSuffix: ' · masked-word model',
    features: { ig: true, tsne: true, placement: true, orderProbe: true, bench: false, sanity: false },
    wording: { overlapVerdict: true, lensNote: 'mlm', mapPullNote: true, defaultHead: 'pca', igNote: 'Computed on this model; not benchmarked (the benchmark below was measured on the default model).' },
    // the default model's layer-12 numbers, for the comparison in B's lens note (from web/data/cloud.json layer_lens[12])
    lensCompare: { probe12: 0.3995, gap12: 0.7812 },
  },
  {
    id: 'c', enabled: false,                       // stretch slot: stays off until its assets pass the same gates
    label: 'E5-small (stretch)', short: 'E5-small',
    descriptor: 'trained to match related text pairs; starts from the masked-word model.',
    hf: 'intfloat/multilingual-e5-small',
    modelDir: 'e5', tokenizerDir: 'e5', gradDir: null, dataDir: './data/e5/', prefix: 'query: ',
    loadingName: 'E5-small model', sizeNote: '118 MB, first time only', titleSuffix: ' · E5-small model',
    features: { ig: false, tsne: false, placement: false, orderProbe: false, bench: false, sanity: false },
    wording: { overlapVerdict: true, lensNote: 'default', mapPullNote: true, defaultHead: 'pca' },
    lensCompare: null,
  },
];

export const byId = id => MODELS.find(m => m.id === id && m.enabled) || null;
export const defaultModel = () => byId(DEFAULT_ID);

function readStore() { try { return localStorage.getItem(KEY); } catch { return null; } }
function writeStore(id) { try { localStorage.setItem(KEY, id); return localStorage.getItem(KEY) === id; } catch { return false; } }

/** The model this page load uses, and why: {model, source: 'url'|'storage'|'default', fallbackFrom, fallbackWhy}. */
export function currentModel(search = location.search) {
  const q = new URLSearchParams(search), want = q.get('model');
  let model = null, source = 'default';
  if (want && byId(want)) { model = byId(want); source = 'url'; if (readStore() !== want) writeStore(want); }
  else { const s = readStore(); if (s && byId(s)) { model = byId(s); source = 'storage'; } }
  model ||= defaultModel();
  const fb = q.get('fallback'), from = fb && fb !== model.id ? (MODELS.find(m => m.id === fb) || null) : null;   // unknown ids are ignored
  return { model, source, fallbackFrom: from, fallbackWhy: from ? (q.get('why') === 'missing' ? 'missing' : 'error') : null };
}

/** Address for a reload into model `id` (keeps ?view= and the #a=..&b=.. hash). The ?model= parameter is kept in step if present, or used when storage fails. */
function urlFor(id, stored, extra = {}) {
  const u = new URL(location.href);
  u.searchParams.delete('fallback'); u.searchParams.delete('why');
  if (u.searchParams.has('model') || !stored) u.searchParams.set('model', id);
  for (const [k, v] of Object.entries(extra)) u.searchParams.set(k, v);
  return u.href;
}
function go(href) { if (href === location.href) location.reload(); else location.replace(href); }

/** Switch to model `id`: save it, tell the other windows (pop-out), reload. broadcast=false when another window asked. */
export function switchModel(id, { broadcast = true } = {}) {
  if (!byId(id)) return false;
  const stored = writeStore(id), href = urlFor(id, stored);
  if (broadcast) window.dispatchEvent(new CustomEvent('lis-model-pick', { detail: { id } }));   // js/popout.js posts {t:'model', id} on its channel
  setTimeout(() => go(href), broadcast ? 150 : 0);                                              // give the channel message time to leave
  return true;
}

/** The loaded non-default model could not start: save A and reload with ?model=a&fallback=<id>&why=missing|error. No retry loop. */
export function fallbackToDefault(failedId, why) {
  window.addEventListener('unhandledrejection', e => e.preventDefault());   // late rejections from the abandoned load: the page is leaving
  writeStore(DEFAULT_ID);
  const u = new URL(location.href);
  u.searchParams.set('model', DEFAULT_ID); u.searchParams.set('fallback', failedId); u.searchParams.set('why', why === 'missing' ? 'missing' : 'error');
  go(u.href);
}

/** Do the model's files exist? (HEAD requests; a static host answers them). Only called for a non-default model. */
export async function assetsPresent(m) {
  const files = [new URL(`../model/${m.modelDir}/config.json`, import.meta.url), new URL(`../model/${m.modelDir}/onnx/model_quantized.onnx`, import.meta.url),
    new URL(`../model/${m.modelDir}/onnx/model_quantized.onnx_data`, import.meta.url),
    new URL(m.dataDir + 'cloud.json', location.href), new URL(m.dataDir + 'cloud_emb.i8.bin', location.href)];
  for (const f of files) {
    try { const r = await fetch(f, { method: 'HEAD', cache: 'no-store' }); if (!r.ok) return false; } catch { return false; }
  }
  return true;
}

/** Header picker, descriptor line, document title, the root data-model attribute and the header band for a non-default model. */
export function mountPicker(cur) {
  const m = cur.model, root = document.documentElement;
  root.dataset.model = m.id;
  root.classList.toggle('model-alt', m.id !== DEFAULT_ID);
  if (m.titleSuffix) document.title = 'Lost in Space' + m.titleSuffix;
  const sel = document.querySelector('#model-pick'), desc = document.querySelector('#model-desc');
  if (sel) {
    sel.replaceChildren(...MODELS.filter(x => x.enabled).map(x => { const o = document.createElement('option'); o.value = x.id; o.textContent = x.label; o.title = x.descriptor; return o; }));
    sel.value = m.id;
    sel.addEventListener('change', () => { if (sel.value !== m.id) { sel.disabled = true; switchModel(sel.value); } });
  }
  if (desc) desc.textContent = cur.fallbackFrom ? fallbackText(cur) : m.descriptor;
  const tr = document.querySelector('#viewer .overlay.tr');   // the pop-out hides the header: a tag on its map names a non-default model
  if (tr && !document.querySelector('#model-tag')) { const t = document.createElement('span'); t.id = 'model-tag'; t.className = 'model-tag'; t.textContent = 'Model: ' + m.short; tr.prepend(t); }
  window.lisModel = { id: m.id, fallbackFrom: cur.fallbackFrom ? cur.fallbackFrom.id : null, switchTo: id => switchModel(id, { broadcast: false }) };
}

export function fallbackText(cur) {
  const f = cur.fallbackFrom;
  return cur.fallbackWhy === 'missing' ? `The ${f.loadingName} files were not found; showing the default model.` : `Could not load the ${f.loadingName}; showing the default model.`;
}
