// Page controller for Lost in Space (working title: Meaning Map): load cloud + model, run a sentence pair, render the 3-d map,
//       the four explanations, the deletion test and the evidence cards.
// Verification: tests/smoke.mjs (headless Chrome)

import { loadEngine, dot, tokenVecs, LAYERS, MAX_TOKENS, supersede, SUPERSEDED } from './engine.js';
import { wordGroups, guessLang } from './tokens.js';
import { analyzePair, occlusion, deletionTest, toWords } from './xai.js';
import { loadGrad, explainGrad, IG_STEPS } from './grad.js';
import { orderTest } from './order.js';
import { loadCloud } from './heads.js';
import { View3D } from './view3d.js';
import { lineChart, barChart, showTip, hideTip, tipRow } from './charts.js';
import { currentModel, mountPicker, assetsPresent, fallbackToDefault, fallbackText, DEFAULT_ID } from './models.js';

const $ = s => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const f2 = v => (+v.toFixed(2) >= 0 ? '' : '−') + Math.abs(v).toFixed(2);
const g3 = v => (+v.toFixed(3) >= 0 ? '' : '−') + Math.abs(v).toFixed(3);
const f3 = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);

const LANG_COLORS = ['--series-1', '--series-2', '--series-3', '--series-4'];          // en es hi zh (+ shape per language)
const TOPIC_COLORS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5', '--series-6', '--series-7'];
const METHODS = {
  occlusion: { name: 'Occlusion', color: '--series-1', signed: true,
    note: 'Delete one word, re-run the model, record how far the score falls. A direct test of the model; the deleted-word sentence can be ungrammatical.' },
  pair: { name: 'Pair split', color: '--series-2', signed: true,
    note: 'An exact split of the score across words at the last layer: the parts add up to the score. Each word vector already holds context from its whole sentence.' },
  ig: { name: 'Integrated gradients', tab: 'Int. gradients', color: '--series-5', signed: true,
    note: 'Fade the sentence in from a blank one (every word replaced by padding) in 24 steps and add up the gradient along the way. The parts add up to the change in score from blank to full. Real backpropagation, run in your browser.' },
  rollout: { name: 'Attention', color: '--series-4', signed: false,
    note: 'Attention rollout: where attention flowed, composed over all 12 layers. It never looks at the other sentence, so it shows routing, not a reason for this score.' },
};
const EXAMPLES = [
  ['EN / ES', 'The cat sleeps on the sofa.', 'El gato duerme en el sofá.'],
  ['EN / HI', 'I went to the hospital to see the doctor.', 'मैं डॉक्टर से मिलने अस्पताल गया।'],
  ['EN / ZH', 'The children are playing football in the park.', '孩子们在公园里踢足球。'],
  ['ZH / HI', '今天的天气很好。', 'आज मौसम बहुत अच्छा है।'],
  ['Word order', 'The dog bit the man.', 'The man bit the dog.'],
  ['Negation', 'I love this movie.', 'I do not love this movie.'],
  ['Two banks', 'She sat on the bank of the river.', 'She deposited money at the bank.'],
  ['Unrelated', 'The stock market fell sharply today.', 'My grandmother bakes bread every Sunday.'],
];

const loadLangs = () => { try { const v = JSON.parse(localStorage.getItem('mm.langs')); if (Array.isArray(v) && v.length === 4 && v.some(Boolean)) return v.map(Boolean); } catch { /* storage unavailable */ } return [true, true, true, true]; };
const CUR = currentModel(), MODEL = CUR.model, ALT = MODEL.id !== DEFAULT_ID, F = MODEL.features;
const DEFAULT_NAME = 'paraphrase-multilingual-MiniLM';
const state = { langs: loadLangs(), head: MODEL.wording.defaultHead || 'ae', colorBy: 'topic', layer: 12, method: 'occlusion', matrix: 'specific', showWords: true, isolate: null, result: null, bench: 'all', orderSeed: 1, busy: false };
let cloud, engine, view, evalData = null, placement = null, orderData = null, sanityData = null;
let bootNotice = CUR.fallbackFrom ? fallbackText(CUR) : null;          // shown in the chip until the first result is replaced

// ------------------------------------------------------------------ boot
init().catch(err => { console.error(err); setStatus('error', 'Could not start: ' + (err.message || err)); });

async function init() {
  setStatus('', 'Loading map…');
  mountPicker(CUR);
  if (ALT) {                                                                // a non-default model: its files must exist, and any load failure falls back to A
    if (!(await assetsPresent(MODEL))) { setStatus('error', `The ${MODEL.loadingName} files were not found; loading the default model…`); fallbackToDefault(MODEL.id, 'missing'); return; }
    try { await bootModel(); } catch (err) { console.warn('model ' + MODEL.id + ' failed to load; falling back to the default model', err); setStatus('error', `Could not load the ${MODEL.loadingName}; loading the default model…`); fallbackToDefault(MODEL.id, 'error'); return; }
  } else await bootModel();
  setStatus('ready', bootNotice || `Ready · ${MODEL.short} · runs in your browser`);
  $('#go').disabled = false;
  $('#why-empty').textContent = 'Type two sentences and press “Map them”, or pick an example.';
  const h = new URLSearchParams(location.hash.slice(1));
  if (!$('#in-a').value.trim() && !$('#in-b').value.trim()) {              // keep anything typed while the model was loading
    $('#in-a').value = h.get('a') || EXAMPLES[0][1];
    $('#in-b').value = h.get('b') || EXAMPLES[0][2];
  }
  submit();
}

/** Cloud, map, cards and the encoder for the loaded model. Rejects on a load failure (init falls back to A for a non-default model). */
async function bootModel() {
  EXAMPLES.forEach(([name, a, b]) => { const btn = el('button', null, name); btn.type = 'button'; btn.title = a + '  |  ' + b; btn.onclick = () => { $('#in-a').value = a; $('#in-b').value = b; submit(); }; $('#examples').append(btn); });
  cloud = await loadCloud(MODEL.dataDir, { tsne: F.tsne });
  if (state.head !== 'ae') [...$('#seg-head').querySelectorAll('button')].forEach(x => x.classList.toggle('on', x.dataset.v === state.head));
  applyModelLabels();
  try { view = new View3D($('#viewer'), { onHover: hover3d, onPick: pick3d }); }
  catch (e) { console.warn('WebGL unavailable', e); $('#nogl').hidden = false; }
  bindControls();
  drawCloud();
  renderMapCard();
  if (F.placement) fetch(MODEL.dataDir + 'placement.json').then(r => (r.ok ? r.json() : null)).then(d => { placement = d; renderMapCard(); }).catch(() => {});
  const optional = (file, then, dir = './data/') => fetch(dir + file).then(r => (r.ok ? r.json() : null)).catch(() => null).then(then).catch(err => console.error(err));
  optional('eval.json', d => { evalData = d; renderBench(); renderStress(); });              // offline benchmark files: measured on the default model (labeled when another is loaded)
  optional('sanity.json', d => { sanityData = d; renderStress(); });
  if (F.orderProbe) optional('order.json', d => { orderData = d; renderOrder(); }, MODEL.dataDir);
  else renderOrder();

  const seen = {};
  engine = await loadEngine(p => {
    if (p.status === 'progress' && p.total) { seen[p.file] = [p.loaded, p.total]; const [l, t] = Object.values(seen).reduce((s, v) => [s[0] + v[0], s[1] + v[1]], [0, 0]); setStatus('', `Loading ${MODEL.loadingName}… ${Math.round(100 * l / t)}%` + (MODEL.sizeNote ? ` (${MODEL.sizeNote})` : '')); }
  }, { modelDir: MODEL.modelDir, tokenizerDir: MODEL.tokenizerDir });
}

/** Labels for the features the loaded model has no data for (nothing changes for the default model). */
function applyModelLabels() {
  if (!ALT) return;
  const flag = (text, id) => { const p = el('p', 'model-flag', text); if (id) p.id = id; return p; };
  if (!F.ig || !MODEL.gradDir) {
    const t = $('#tabs button[data-m="ig"]'); t.disabled = true; t.setAttribute('aria-disabled', 'true');
    t.title = 'Integrated gradients needs a separate gradient file; built for the default model only.';
    $('#tabs').after(flag('Integrated gradients needs a separate gradient file; built for the default model only.', 'ig-off-note'));
  }
  const measured = `Measured on the default model (${DEFAULT_NAME}), not on the loaded model`;
  if (!F.bench) { const h = $('#bench-chart').parentElement.querySelector('h3'); h.before(flag(measured + '.', 'bench-flag')); }
  if (!F.sanity) { const c = $('#sanity-chart').parentElement, h = c.querySelector('h3'); h.before(flag(measured + '.', 'sanity-flag')); }
}

function statusName() { return ALT ? MODEL.short + ' · ' : ''; }

function setStatus(cls, text) { $('#status').className = 'status ' + cls; $('#status-text').textContent = text; }

function bindControls() {
  $('#form').addEventListener('submit', ev => { ev.preventDefault(); submit(); });
  const seg = (id, key, fn) => $(id).addEventListener('click', ev => { const b = ev.target.closest('button'); if (!b) return; state[key] = b.dataset.v; [...$(id).querySelectorAll('button')].forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); }); fn(); });
  seg('#seg-color', 'colorBy', () => { state.isolate = null; drawCloud(); });
  seg('#seg-head', 'head', () => { drawCloud(); drawQuery(true); renderMapCard(); });
  seg('#seg-bench', 'bench', renderBench);
  $('#btn-scramble').onclick = () => { state.orderSeed++; if (state.result) runOrder(state.result).catch(err => { if (err.message !== SUPERSEDED) console.error(err); }); };
  seg('#seg-matrix', 'matrix', () => state.result && renderMatrix());
  $('#tabs').addEventListener('click', ev => { const b = ev.target.closest('button'); if (!b || b.disabled) return; state.method = b.dataset.m; [...$('#tabs').children].forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-selected', x === b); }); renderWhy(); drawQuery(false); });
  document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', x.classList.contains('on')));
  [...$('#tabs').children].forEach(x => x.setAttribute('aria-selected', x.classList.contains('on')));
  if (!cloud.hasMap('tsne')) { const t = $('#seg-head button[data-v="tsne"]'); if (t) t.hidden = true; }
  $('#btn-words').onclick = e => { state.showWords = !state.showWords; e.target.classList.toggle('on', state.showWords); drawQuery(false); };
  $('#btn-reset').onclick = () => view?.resetView();
  $('#layer').addEventListener('input', e => { state.layer = +e.target.value; layerLabel(); drawQuery(false); });
  window.addEventListener('themechange', () => {                    // colors are read from CSS when drawn: draw again with the new tokens (no new analysis run)
    if (!state.result) return;
    drawCloud(); drawQuery(false); renderWhy(); renderMatrix(); renderBench(); renderStress(); renderOrder(); renderLens(); renderMapCard();
    if (state.result.del) drawDeletion(state.result);
  });
  $('#layer-prev').onclick = () => stepLayer(-1);
  $('#layer-next').onclick = () => stepLayer(1);
  $('#btn-play').onclick = playLayers;
}
function layerLabel() {
  $('#layer-label').textContent = state.layer === 12 ? 'Layer 12 · final' : state.layer === 0 ? 'Layer 0 · no context' : `Layer ${state.layer}`;
  $('#layer-prev').disabled = state.layer <= 0; $('#layer-next').disabled = state.layer >= 12;   // arrows stop at both ends
}
/** Arrow buttons: move the slider one layer (clamped 0..12) and fire its own 'input' event, so the map, label, arrows and the pop-out sync run the same code as a drag. */
function stepLayer(d) {
  state.stopPlay = true;                                    // an arrow press ends an auto-play in progress
  const s = $('#layer'), v = Math.max(0, Math.min(12, +s.value + d));
  if (v === +s.value) return;
  s.value = v; s.dispatchEvent(new Event('input', { bubbles: true }));
}
async function playLayers() {
  if (!state.result || state.playing) return;
  state.playing = true; state.stopPlay = false;
  try { for (let l = 0; l <= 12 && !state.stopPlay; l++) { state.layer = l; $('#layer').value = l; layerLabel(); drawQuery(false); await new Promise(r => setTimeout(r, 420)); } }
  finally { state.playing = false; }
}

// ------------------------------------------------------------------ cloud
function categories() {
  const m = cloud.meta;
  return state.colorBy === 'lang' ? m.langs.map((l, i) => ({ name: l.name, color: css(LANG_COLORS[i]), shape: i }))
    : m.topics.map((t, i) => ({ name: t.replace('/', ' / '), color: css(TOPIC_COLORS[i]), shape: 0 }));
}
function drawCloud() {
  const cats = categories(), pts = cloud.meta.points, n = cloud.n;
  const colors = new Float32Array(n * 3), shapes = new Float32Array(n);
  const rgb = cats.map(c => hex2rgb(c.color));
  for (let i = 0; i < n; i++) { const k = state.colorBy === 'lang' ? pts[i].l : pts[i].c; colors.set(rgb[k], i * 3); shapes[i] = pts[i].l; }
  state.baseColors = colors;
  // language switches: always present, and the language legend when color = language (shape always marks language)
  const lt = $('#lang-toggles'); lt.replaceChildren(el('span', 'seg-label', 'Show'));
  cloud.meta.langs.forEach((l, k) => {
    const b = el('button'); b.type = 'button'; b.setAttribute('aria-pressed', String(state.langs[k]));
    b.title = state.langs[k] ? `Hide ${l.name} sentences` : `Show ${l.name} sentences`;
    const mk = el('span', 'mark s' + k); mk.style.background = state.colorBy === 'lang' ? css(LANG_COLORS[k]) : css('--text-2'); b.append(mk, el('span', null, l.name));
    b.classList.toggle('off', !state.langs[k]);
    b.onclick = () => toggleLang(k);
    lt.append(b);
  });
  const lg = $('#legend3d'); lg.replaceChildren();
  if (state.colorBy !== 'lang') cats.forEach((c, k) => {
    const b = el('button'); b.type = 'button'; b.title = 'Click to isolate';
    const mk = el('span', 'mark s0'); mk.style.background = c.color; b.append(mk, el('span', null, c.name));
    b.classList.toggle('off', state.isolate != null && state.isolate !== k);
    b.onclick = () => { state.isolate = state.isolate === k ? null : k; drawCloud(); };
    lg.append(b);
  });
  const m = cloud.meta.metrics[state.head];
  $('#honesty').textContent = state.head === 'tsne'
    ? 't-SNE: only nearby points mean anything; gaps and long distances do not. Shapes mark language.'
    : 'A 3D sketch of 384 dimensions: nearby points are roughly right, exact distances are not. Shapes mark language.' + (MODEL.wording.mapPullNote ? ' The trained maps include a small pull of translations together; PCA has none.' : '');
  if (!view) return;
  view.setCloud(cloud.coords(state.head), colors, shapes);
  applyEmphasis();
}
function applyEmphasis() {
  if (!view) return;
  const n = cloud.n, pts = cloud.meta.points, sizes = new Float32Array(n).fill(0.06), alphas = new Float32Array(n).fill(state.result ? 0.3 : 0.8);
  if (state.isolate != null && state.colorBy !== 'lang') for (let i = 0; i < n; i++) if (pts[i].c !== state.isolate) alphas[i] = 0.03;
  // lit-up neighbors keep their category color (color always means language/topic); A/B identity is carried by the link lines
  if (state.result) for (const d of [...state.result.neighA, ...state.result.neighB]) { sizes[d.index] = 0.15; alphas[d.index] = 1; }
  for (let i = 0; i < n; i++) if (!state.langs[pts[i].l]) { alphas[i] = 0; sizes[i] = 0; }   // switched-off languages: not drawn, not hoverable
  view.setEmphasis(sizes, alphas);
}

// ---- languages the reader wants to see.  Display only: the model and every score are unchanged.
const langMask = () => (state.langs.every(Boolean) ? null : Uint8Array.from(cloud.meta.points, p => +state.langs[p.l]));
const readLang = () => state.langs.findIndex(Boolean);                       // first language still on (English when all are)
/** The same reference sentence in the reader's language, when the point itself is in another one. */
function gloss(i) {
  const p = cloud.meta.points[i], r = readLang(); if (p.l === r) return null;
  const j = cloud.sameSentence(i, r); return j < 0 ? null : `${cloud.meta.langs[r].name}: ${cloud.meta.points[j].t}`;
}
function findNeighbors(R) { const mask = langMask(); R.neighA = cloud.neighbors(R.an.eA, 8, mask); R.neighB = cloud.neighbors(R.an.eB, 8, mask); }
function toggleLang(k) {
  if (state.langs[k] && state.langs.filter(Boolean).length === 1) return;     // always keep one language on
  state.langs[k] = !state.langs[k];
  try { localStorage.setItem('mm.langs', JSON.stringify(state.langs)); } catch { /* storage unavailable */ }
  if (state.result) { findNeighbors(state.result); renderNeighbors(); }
  drawCloud(); drawQuery(false);
}

function hover3d(i, ev) {
  const tip = $('#tip3d');
  if (i < 0) { tip.hidden = true; return; }
  const p = cloud.meta.points[i], r = $('#viewer').getBoundingClientRect();
  tip.replaceChildren(el('div', null, p.t));
  const gl = gloss(i); if (gl) tip.append(el('div', 'gloss', gl));
  let meta = `${cloud.meta.langs[p.l].name} · ${cloud.meta.topics[p.c]}`;
  if (state.result) { const v = cloud.emb.subarray(i * 384, i * 384 + 384); meta += ` · similarity to A ${f2(dot(v, state.result.an.eA))}, to B ${f2(dot(v, state.result.an.eB))}`; }
  tip.append(el('div', 'meta', meta), el('div', 'meta', 'Click: use as B · Shift-click: use as A'));
  tip.hidden = false;
  tip.style.left = Math.min(r.width - 330, ev.clientX - r.left + 14) + 'px'; tip.style.top = Math.max(8, ev.clientY - r.top - 70) + 'px';
}
function pick3d(i, ev) { $(ev.shiftKey ? '#in-a' : '#in-b').value = cloud.meta.points[i].t; if (engine) submit(); }

// ------------------------------------------------------------------ run a pair
async function submit() {
  const a = $('#in-a').value.trim(), b = $('#in-b').value.trim();
  if (!engine) return;
  if (state.busy) { state.again = true; return; }                          // run once more with whatever is in the boxes when this run ends
  if (!a || !b) { setStatus('error', 'Type a sentence in both boxes.'); return; }
  const A = engine.encode(a), B = engine.encode(b);
  const gA = wordGroups(A.tokens), gB = wordGroups(B.tokens);
  if (!gA.some(g => !g.punct) || !gB.some(g => !g.punct)) { setStatus('error', 'Each sentence needs at least one word.'); return; }
  supersede();                                                              // tests still running for the previous pair stop at their next model call
  state.busy = true; state.gradError = null; $('#go').disabled = true; setStatus('', 'Thinking…');
  const t0 = performance.now();
  try {
    const r = await engine.run([A.ids, B.ids]);
    const an = analyzePair(r, gA, gB);
    const [occA, occB] = [await occlusion(engine, A.ids, gA, an.eB, an.sim), await occlusion(engine, B.ids, gB, an.eA, an.sim)];
    an.scores.occlusion = { A: occA, B: occB };
    const R = { a, b, A, B, gA, gB, an, run: r, ms: performance.now() - t0, del: null, grad: null, order: null };
    findNeighbors(R); state.result = R;
    history.replaceState(null, '', '#' + new URLSearchParams({ a, b }).toString());
    $('#why-empty').hidden = true; $('#why-body').hidden = false;
    window.__mm = { state, cloud };
    renderWhy(); renderNeighbors(); renderLens(); renderMapCard(); applyEmphasis(); drawQuery(true); renderOrder();
    const note = bootNotice; bootNotice = null;
    setStatus('ready', note || `Ready · ${statusName()}${(state.result.ms / 1000).toFixed(1)} s in your browser`);
    runDeletion().then(() => runGradients(R)).then(() => runOrder(R)).catch(err => { if (err.message !== SUPERSEDED) console.error(err); });
  } catch (err) { if (err.message !== SUPERSEDED) { console.error(err); setStatus('error', 'Something went wrong: ' + (err.message || err)); } }
  finally { state.busy = false; $('#go').disabled = false; if (state.again) { state.again = false; submit(); } }
}

function drawQuery(refocus) {
  const R = state.result; if (!view || !R) return;
  const L = state.layer, col = { A: css('--a'), B: css('--b') }, sc = R.an.scores[state.method] || R.an.scores.occlusion;
  const build = (key, groups, wordVecs, layerVecs, neigh) => {
    const vals = sc[key], mx = Math.max(1e-9, ...vals.map(Math.abs));
    return {
      key, color: col[key], pos: cloud.project(state.head, layerVecs[L], L),
      traj: layerVecs.map((v, l) => cloud.project(state.head, v, l)),
      links: L === 12 ? neigh.map(d => Array.from(cloud.coords(state.head).subarray(d.index * 3, d.index * 3 + 3))) : [],
      words: groups.map((g, i) => ({ g, i })).filter(o => !o.g.punct).map(({ g, i }) => ({ text: g.text, pos: cloud.project(state.head, wordVecs[L][i], L), weight: Math.abs(vals[i]) / mx })),
    };
  };
  const S = [build('A', R.gA, R.an.wordVecA, R.an.layerVec.A, R.neighA), build('B', R.gB, R.an.wordVecB, R.an.layerVec.B, R.neighB)];
  view.setQuery(S, { showWords: state.showWords, showTraj: L !== 12, layer: L });
  if (refocus) view.focus([S[0].pos, S[1].pos, ...S.flatMap(s => s.links), ...(state.showWords ? S.flatMap(s => s.words.map(w => w.pos)) : [])]);
}

// ------------------------------------------------------------------ explanation panel
function mixColor(v, signed) { // v in [-1, 1] -> css color; diverging blue / red around a neutral, or one-hue for unsigned
  const t = Math.min(1, Math.abs(v));
  const base = signed && v < 0 ? css('--neg') : css('--pos');
  return `color-mix(in oklab, ${base} ${Math.round(t * 55)}%, ${css('--mid')})`;
}
function renderWhy() {
  const R = state.result; if (!R) return;
  const { an } = R, ref = cloud.meta.sim_ref, M = METHODS[state.method];
  $('#sim').textContent = an.sim.toFixed(2);
  const langs = [guessLang(R.a), guessLang(R.b)].map(c => (c ? cloud.meta.langs.find(l => l.code === c).name : null));
  const who = !langs[0] || !langs[1] ? 'These two sentences' : langs[0] === langs[1] ? 'Both ' + langs[0] : langs[0] + ' and ' + langs[1];
  const cut = [R.A.truncated && 'A', R.B.truncated && 'B'].filter(Boolean);
  let verdict = 'between the two reference groups';
  const overlap = !!ref && MODEL.wording.overlapVerdict && ref.translation[0] <= ref.unrelated[4];   // this model's translation and unrelated bands overlap
  if (ref && !overlap) verdict = an.sim >= ref.translation[0] ? 'as close as a typical translation pair' : an.sim <= ref.unrelated[4] ? 'about as far apart as two unrelated sentences' : 'closer than unrelated sentences, but short of a translation pair';
  const line = overlap ? (an.sim >= ref.translation[2] ? `${who}: about as close as a typical translation pair, but on this model unrelated pairs score almost as high.` : `${who}: this model barely separates translations from unrelated sentences here.`)
    : `${who}: the model places these ${verdict}.`;
  $('#sim-text').textContent = line + (cut.length ? ` Sentence ${cut.join(' and ')} was longer than the ${MAX_TOKENS} word pieces this page handles; only the beginning was used.` : '');
  renderStrip(an.sim, ref);
  $('#method-note').textContent = M.note + (state.method === 'ig' && MODEL.wording.igNote ? ' ' + MODEL.wording.igNote : '');

  const sc = an.scores[state.method];
  const pending = $('#method-pending'); pending.hidden = !!sc;
  $('#chips-a').parentElement.hidden = $('#chips-b').parentElement.hidden = $('#scale').hidden = !sc;
  if (!sc) { pending.textContent = state.gradError ? 'The gradient step failed: ' + state.gradError : (state.gradStatus || 'Waiting for the gradient model…'); renderMatrix(); return; }
  if (state.method === 'ig' && R.grad) $('#method-note').textContent = (MODEL.wording.igNote ? M.note + ' ' + MODEL.wording.igNote : M.note) + ` Check: A’s word parts add up to ${f2(R.grad.A.igSum)}, and the real rise from blank to full was ${f2(R.grad.A.igTarget)}; they should match.`;
  const mx = Math.max(1e-9, ...sc.A.map(Math.abs), ...sc.B.map(Math.abs));
  const chips = (host, groups, vals) => {
    host.replaceChildren();
    groups.forEach((g, i) => {
      const c = el('span', 'chip' + (g.punct ? ' punct' : ''));
      c.append(el('span', 'w', g.text));
      if (!g.punct) {
        c.style.background = mixColor(vals[i] / mx, M.signed);
        c.append(el('span', 'v', M.signed ? f3(vals[i]) : vals[i].toFixed(3)));
        c.addEventListener('pointermove', ev => showTip(ev, t => { t.append(el('div', 'tip-title', g.text)); Object.entries(METHODS).filter(([k]) => an.scores[k]).forEach(([k, m]) => { const s = an.scores[k][host.id === 'chips-a' ? 'A' : 'B'][i]; tipRow(t, css(m.color), m.name, m.signed ? f3(s) : s.toFixed(3)); }); }));
        c.addEventListener('pointerleave', hideTip);
      }
      host.append(c);
    });
  };
  chips($('#chips-a'), R.gA, sc.A); chips($('#chips-b'), R.gB, sc.B);
  const scale = $('#scale'); scale.replaceChildren();
  const bar = el('span', 'bar');
  bar.style.background = M.signed ? `linear-gradient(90deg, ${mixColor(-1, true)}, ${css('--mid')}, ${mixColor(1, true)})` : `linear-gradient(90deg, ${css('--mid')}, ${mixColor(1, false)})`;
  scale.append(el('span', null, M.signed ? 'lowers the score' : 'little'), bar, el('span', null, M.signed ? 'raises the score' : 'much'));
  renderMatrix();
}

function renderStrip(sim, ref) {
  const host = $('#sim-strip'); host.replaceChildren(); if (!ref) return;
  const NS = 'http://www.w3.org/2000/svg', W = 420, X = v => 8 + (Math.max(-0.2, Math.min(1, v)) + 0.2) / 1.2 * (W - 16);
  const s = document.createElementNS(NS, 'svg'); s.setAttribute('viewBox', `0 0 ${W} 46`); s.setAttribute('class', 'chart'); s.setAttribute('role', 'img');
  s.setAttribute('aria-label', `Similarity ${sim.toFixed(2)} compared with unrelated sentences and translation pairs`);
  const add = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text) e.textContent = text; s.append(e); return e; };
  add('line', { x1: 8, x2: W - 8, y1: 16, y2: 16, class: 'axis' });
  add('rect', { x: X(ref.unrelated[0]), y: 11, width: X(ref.unrelated[4]) - X(ref.unrelated[0]), height: 10, rx: 4, fill: css('--muted-mark') });
  add('rect', { x: X(ref.translation[0]), y: 11, width: X(ref.translation[4]) - X(ref.translation[0]), height: 10, rx: 4, fill: css('--series-3') });
  // band labels, centered under each band's median. If they collide (this model's bands overlap) the right-hand one drops to a second row;
  // either label is shifted to stay inside the chart. Text is a fixed screen size, so measure with the same scale (--k) layout.js will set.
  const hw = host.getBoundingClientRect().width; if (hw) s.style.setProperty('--k', (hw / W).toFixed(4));
  const lu = add('text', { x: X(ref.unrelated[2]), y: 38, class: 'band-label', 'text-anchor': 'middle' }, 'unrelated sentences');
  const lt = add('text', { x: X(ref.translation[2]), y: 38, class: 'band-label', 'text-anchor': 'middle' }, 'translations');
  add('circle', { cx: X(sim), cy: 16, r: 7, fill: css('--text'), stroke: css('--surface'), 'stroke-width': 2 });
  host.append(s);
  const fit = t => { const bb = t.getBBox(), dx = bb.x < 2 ? 2 - bb.x : bb.x + bb.width > W - 2 ? W - 2 - bb.x - bb.width : 0; if (dx) t.setAttribute('x', +t.getAttribute('x') + dx); return t.getBBox(); };
  let bu = fit(lu), bt = fit(lt);
  if (bu.x < bt.x + bt.width + 4 && bt.x < bu.x + bu.width + 4) {
    const [lo, hi, bl, bh] = +lu.getAttribute('x') <= +lt.getAttribute('x') ? [lu, lt, bu, bt] : [lt, lu, bt, bu];
    const row = bl.height + 2; hi.setAttribute('y', 38 + row); s.setAttribute('viewBox', `0 0 ${W} ${Math.ceil(46 + row)}`); fit(hi);
  }
}

function renderMatrix() {
  const R = state.result, { wm } = R.an, host = $('#matrix'); host.replaceChildren();
  const ia = R.gA.map((g, i) => (g.punct ? -1 : i)).filter(i => i >= 0), ib = R.gB.map((g, i) => (g.punct ? -1 : i)).filter(i => i >= 0);
  let shown = 0; for (const i of ia) for (const j of ib) shown += wm.W[i][j];
  // "pair-specific" = double-centered: remove each word's overall level so what is left is who pairs with whom
  const specific = state.matrix === 'specific';
  const rowMean = Object.fromEntries(ia.map(i => [i, ib.reduce((s, j) => s + wm.W[i][j], 0) / ib.length]));
  const colMean = Object.fromEntries(ib.map(j => [j, ia.reduce((s, i) => s + wm.W[i][j], 0) / ia.length]));
  const grand = shown / (ia.length * ib.length);
  const val = (i, j) => (specific ? wm.W[i][j] - rowMean[i] - colMean[j] + grand : wm.W[i][j]);
  let mx = 1e-9; for (const i of ia) for (const j of ib) mx = Math.max(mx, Math.abs(val(i, j)));
  const t = el('table'), th = el('thead'), hr = el('tr'); hr.append(el('th'));
  ib.forEach(j => { const h = el('th', null, R.gB[j].text); h.title = R.gB[j].text; hr.append(h); }); th.append(hr); t.append(th);
  const tb = el('tbody');
  ia.forEach(i => {
    const tr = el('tr'); tr.append(el('th', null, R.gA[i].text));
    const best = ib.reduce((b, j) => (val(i, j) > val(i, b) ? j : b), ib[0]);
    ib.forEach(j => {
      const td = el('td'), v = wm.W[i][j], shownV = val(i, j); td.style.background = mixColor(shownV / mx, true); if (j === best && shownV > 0) td.className = 'best';
      td.addEventListener('pointermove', ev => showTip(ev, tp => { tp.append(el('div', 'tip-title', `${R.gA[i].text}  ×  ${R.gB[j].text}`)); tipRow(tp, null, 'share of the score', f3(v)); tipRow(tp, null, 'pair-specific part', f3(wm.W[i][j] - rowMean[i] - colMean[j] + grand)); }));
      td.addEventListener('pointerleave', hideTip); tr.append(td);
    });
    tb.append(tr);
  });
  t.append(tb); host.append(t);
  $('#matrix-note').textContent = (specific
    ? 'Blue = these two words pair up more than you would expect from each word on its own. '
    : `Exact last-layer split of the ${wm.total.toFixed(2)}: these word pairs account for ${shown.toFixed(2)}; punctuation and start/end tokens carry the other ${(wm.total - shown).toFixed(2)}. `)
    + 'Outlined cell = each row’s strongest partner.';
}

// ------------------------------------------------------------------ cards
function renderNeighbors() {
  const R = state.result, host = $('#neighbors'); host.replaceChildren();
  const on = cloud.meta.langs.filter((_, k) => state.langs[k]).map(l => l.name);
  $('#neigh-note').textContent = `Nearest reference sentences in the full 384-dimension space (not the 3D picture)${on.length < 4 ? ', among the languages switched on (' + on.join(', ') + ')' : ''}. Click one to load it as sentence B.`;
  for (const [key, list] of [['A', R.neighA], ['B', R.neighB]]) {
    const g = el('div', 'neigh-group'), head = el('div', 'neigh-head');
    head.append(el('span', 'tag ' + key.toLowerCase(), key), el('span', null, `nearest to sentence ${key}`)); g.append(head);
    list.slice(0, 5).forEach(d => {
      const p = cloud.meta.points[d.index], b = el('button', 'neigh-item'); b.type = 'button';
      const right = el('span'), gl = gloss(d.index); right.append(el('span', 'txt', p.t)); if (gl) right.append(el('span', 'gloss', gl));
      right.append(el('span', 'meta', `${cloud.meta.langs[p.l].name} · ${cloud.meta.topics[p.c]}`));
      b.append(el('span', 'cos', d.sim.toFixed(2)), right);
      b.onclick = () => { $('#in-b').value = p.t; submit(); };
      g.append(b);
    });
    host.append(g);
  }
}

async function runDeletion() {
  const R = state.result, host = $('#del-chart');
  const nWords = R.gA.filter(g => !g.punct).length;
  if (nWords < 3) { host.replaceChildren(el('p', 'empty-small', 'Sentence A needs at least three words for a deletion test.')); $('#del-summary').textContent = ''; window.__mm.deletionDone = true; return; }
  $('#del-summary').textContent = '';
  host.replaceChildren(el('p', 'empty-small', 'Running the deletion test…'));
  const scores = Object.fromEntries(Object.keys(METHODS).filter(k => R.an.scores[k]).map(k => [k, R.an.scores[k].A]));
  const del = await deletionTest(engine, R.A.ids, R.gA, R.an.eB, R.an.sim, scores);
  if (state.result !== R) return;
  R.del = del; drawDeletion(R);
  window.__mm.deletionDone = true;
}
function drawDeletion(R) {
  const del = R.del, host = $('#del-chart');
  const WIDTH = { occlusion: 6, pair: 6, ig: 2.5, rollout: 2.5 };   // thick lines drawn first, thin ones on top, so lines that coincide stay visible
  const GLYPH = { occlusion: 'circle', pair: 'square', ig: 'diamond', rollout: 'triangle' };
  const series = Object.entries(METHODS).filter(([k]) => del.curves[k]).map(([k, m]) => ({ name: m.name, color: css('--del-' + k), values: del.curves[k], width: WIDTH[k], glyph: GLYPH[k] }));
  series.push({ name: 'Random order (baseline)', color: css('--muted'), values: del.curves.random, dash: true, width: 2.5, glyph: 'none' });
  const all = series.flatMap(s => s.values), lo = Math.min(0, Math.floor((Math.min(...all) - 0.03) * 10) / 10), hi = 1;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  lineChart(host, series, Array.from({ length: del.n + 1 }, (_, i) => String(i)), { yDomain: [lo, hi], yTicks: ticks, height: 250, xTitle: 'words deleted from sentence A', fmt: v => v.toFixed(2), xTip: v => `${v} word${v === '1' ? '' : 's'} deleted · cosine similarity to B`, aria: 'Deletion test curves' });
  const rank = Object.entries(del.area).sort((a, b) => a[1] - b[1]);
  const s = $('#del-summary');
  if (R.an.sim < 0.3) { s.textContent = 'The score starts near zero, so there is little for a deletion test to remove here. Try a pair the model calls similar.'; return; }
  s.replaceChildren(el('span', null, 'Score left on average as words are removed (lower = better ranking): '));
  rank.forEach(([k, v], i) => { const b = el('b', null, `${k === 'random' ? 'random' : METHODS[k].name} ${v.toFixed(2)}`); s.append(b, document.createTextNode(i < rank.length - 1 ? ' · ' : '')); });
}

// ------------------------------------------------------------------ gradients (second model, loaded after the first result)
async function runGradients(R) {
  if (state.result !== R) return;
  if (!F.ig || !MODEL.gradDir) { window.__mm.gradDone = true; return; }         // no gradient file for this model: the tab is disabled with a note
  const tabBtn = $('#tabs button[data-m="ig"]'), busy = t => { if (state.result !== R) return; state.gradStatus = t; tabBtn.classList.toggle('busy', !!t); if (!R.grad && state.method === 'ig') renderWhy(); };
  try {
    const seen = {};
    busy('Loading the gradient model (43 MB, once)…');
    await loadGrad(p => { if (p.status === 'progress' && p.total) { seen[p.file] = [p.loaded, p.total]; const [l, t] = Object.values(seen).reduce((a, v) => [a[0] + v[0], a[1] + v[1]], [0, 0]); busy(`Loading the gradient model… ${Math.round(100 * l / t)}%`); } }, MODEL.gradDir);
    if (state.result !== R) return;
    const t0 = performance.now();
    const blank = seq => seq.ids.map((id, i) => (i === 0 || i === seq.ids.length - 1 ? id : engine.padId));
    const rb = await engine.run([blank(R.A), blank(R.B)]);                      // layer-0 states of the blank baselines
    busy('Integrating gradients… 0%');
    const gA = await explainGrad(tokenVecs(R.run, 0, 0), tokenVecs(rb, 0, 0), R.an.eB, f => busy(`Integrating gradients… ${Math.round(50 * f)}%`));
    if (state.result !== R) return;
    const gB = await explainGrad(tokenVecs(R.run, 1, 0), tokenVecs(rb, 1, 0), R.an.eA, f => busy(`Integrating gradients… ${Math.round(50 + 50 * f)}%`));
    if (state.result !== R) return;
    R.grad = { A: gA, B: gB, ms: performance.now() - t0, steps: IG_STEPS };
    R.an.scores.ig = { A: toWords(gA.ig, R.gA), B: toWords(gB.ig, R.gB) };
    busy(''); renderWhy(); drawQuery(false);
    setStatus('ready', `Ready · ${statusName()}${(R.ms / 1000).toFixed(1)} s · gradients ${(R.grad.ms / 1000).toFixed(1)} s, all in your browser`);
    if (R.del) {                                                                // add integrated gradients to the live deletion test
      const del = await deletionTest(engine, R.A.ids, R.gA, R.an.eB, R.an.sim, { ig: R.an.scores.ig.A });
      if (state.result !== R) return;
      R.del.curves.ig = del.curves.ig; R.del.area.ig = del.area.ig; R.del.orders.ig = del.orders.ig; drawDeletion(R);
    }
  } catch (err) { if (err.message === SUPERSEDED) return; console.error(err); if (state.result === R) { state.gradError = String(err.message || err); busy(''); renderWhy(); } }
  finally { if (state.result === R) window.__mm.gradDone = true; }
}

const BENCH_NAMES = { occlusion: 'Occlusion', integrated_gradients: 'Integrated gradients', pair_decomposition: 'Pair split',
  attention_rollout: 'Attention rollout', token_count: 'Word length only', random_word: 'Random order' };
const BASELINES = ['random_word', 'token_count'];
function renderBench() {
  const host = $('#bench-chart');
  if (!evalData) { host.replaceChildren(el('p', 'empty-small', 'Benchmark file not built yet (run build/eval_attribution.py).')); return; }
  const d = state.bench === 'all' ? evalData.all : evalData.by_explained_language[state.bench];
  const rows = Object.keys(BENCH_NAMES).filter(k => d[k]).map(k => ({ name: BENCH_NAMES[k], value: d[k].gap, lo: d[k].gap_ci95[0], hi: d[k].gap_ci95[1], muted: BASELINES.includes(k), note: `most-important words removed first: ${d[k].morf_auc.toFixed(2)} of the score remains; least-important first: ${d[k].lerf_auc.toFixed(2)}` })).sort((a, b) => b.value - a.value);
  barChart(host, rows, { max: Math.max(0.6, ...rows.map(r => r.hi)) * 1.02, ticks: [0, 0.2, 0.4, 0.6], xTitle: 'faithfulness gap (higher = better)', valueName: 'gap', aria: 'Faithfulness gap by method' });
  const n = state.bench === 'all' ? evalData.n_explanations : evalData.by_language_n[state.bench];
  const note = `Each method ranks the words of one sentence; we remove them in that order and re-run the model. Gap = how much more of the score is lost when the top-ranked words go first than when the bottom-ranked go first (higher = the method found words the model really uses). ${n.toLocaleString()} explanations; whiskers are 95% intervals. Grey bars ignore the model: just ranking words by length already beats random.`;
  $('#bench-note').textContent = note;
}

// ------------------------------------------------------------------ stress tests of the evidence
const pct0 = v => Math.round(v * 100) + '%';
function renderStress() {
  if (sanityData) {
    const pick = { occlusion: ['Occlusion', '--series-1'], integrated_gradients: ['Integrated gradients', '--series-5'], pair_decomposition: ['Pair split', '--series-2'], attention_rollout: ['Attention', '--series-4'], token_count: ['Word length (dashed baseline; ignores the model)', '--muted'] };
    const series = Object.entries(pick).map(([k, [name, c]]) => ({ name, short: name.split(' (')[0], color: css(c), values: [1, ...sanityData.rank_corr[k]], dash: k === 'token_count' }));
    const lo = Math.min(0, ...series.flatMap(x => x.values)) < -0.02 ? -0.25 : 0;
    lineChart($('#sanity-chart'), series, ['trained', ...sanityData.depths.map(d => (d === 12 ? 'all 12' : 'top ' + d))], { yDomain: [lo, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], xEvery: 1, xTitle: 'encoder layers replaced by random weights', xTip: v => `${v} · how much the word ranking stays the same (1 = unchanged)`, aria: 'Model randomization sanity check' });
    const r = sanityData.rank_corr, last = k => r[k][r[k].length - 1];
    { const i = r.attention_rollout.length - 2; $('#evid-takeaway').textContent = `Attention rollout barely changes when most of the model is scrambled (agreement ${f2(r.attention_rollout[i])} with ${sanityData.depths[i]} of 12 layers random), so read it as where attention flowed, not as a reason for the score.`; }
    $('#sanity-note').textContent = `If an explanation describes the model, it should change when we swap the model’s layers for random ones. Each line shows how much a method’s word ranking stays the same (1 = unchanged; ${sanityData.n_explanations} explanations). With all 12 layers random, occlusion and integrated gradients fall to ${f2(last('occlusion'))} and ${f2(last('integrated_gradients'))}. Attention rollout still agrees ${f2(r.attention_rollout[r.attention_rollout.length - 2])} with ${sanityData.depths[r.attention_rollout.length - 2]} layers random: the warning sign. The dashed line ignores the model on purpose, and the word table stays trained, so no line has to reach 0.`;
  }
}

// ------------------------------------------------------------------ word order: live scramble test + offline probe
async function runOrder(R) {
  if (state.result !== R || !engine) return;
  const res = await orderTest(engine, R.A, R.an.layerVec.A, R.an.eB, orderData && orderData.probes, state.orderSeed);
  if (state.result !== R) return;
  R.order = res; window.__mm.orderDone = true; renderOrder();
}
function renderOrder() {
  const R = state.result, o = R && R.order, d = orderData;
  const layers = Array.from({ length: LAYERS }, (_, l) => String(l));
  $('#btn-scramble').hidden = !o;
  if (R && !o) {                                              // nothing for this pair (yet): never leave the previous pair's charts up
    $('#order-live').textContent = window.__mm?.orderDone ? 'Sentence A needs at least three words for the scramble test.' : 'Runs after the other tests.';
    for (const id of ['#order-chart', '#order-summary']) $(id).replaceChildren();
  }
  if (o) {
    $('#order-live').replaceChildren(el('span', null, 'A with its words scrambled: '), el('b', null, '“' + o.variants[0].text + '”'));
    const series = [{ name: 'Your A vs scrambled A (mean of 5)', short: 'your A', color: css('--series-1'), values: o.cosShuffle, emph: true }, { name: 'Your A vs A reversed', short: 'reversed', color: css('--series-4'), values: o.cosReverse }];
    if (d) series.push({ name: `Typical sentence, scrambled (median of ${d.scramble.n_test})`, short: 'typical', color: css('--muted'), values: d.scramble.cos_band_p10_p50_p90.shuffle[1], dash: true });
    const lo = Math.max(0, Math.floor(Math.min(...series.flatMap(x => x.values)) * 10 - 0.5) / 10), ticks = []; for (let t = lo; t <= 1.0001; t += (1 - lo) > 0.45 ? 0.2 : 0.1) ticks.push(+t.toFixed(1));
    lineChart($('#order-chart'), series, layers, { yDomain: [lo, 1], yTicks: ticks, xTitle: 'transformer layer', xEvery: 2, xTip: v => `Layer ${v} · cosine similarity to the original sentence`, aria: 'Similarity between a sentence and its scrambled versions, by layer' });
    $('#order-summary').replaceChildren(el('span', null, 'After scrambling, A still scores '), el('b', null, f2(o.cosShuffle[12])), el('span', null, ' against its own original, and '), el('b', null, f2(o.simOtherShuffle)), el('span', null, ` against B (in order: ${f2(R.an.sim)}).`));
  }
  if (!d) {
    if (!F.orderProbe) {                                      // the probe and the scramble benchmark were fitted on the default model's vectors
      $('#order-note').textContent = 'The scramble benchmark (PAWS-X) was measured on the default model only; the live test above runs on the loaded model.';
    }
    return;
  }
  const w = d.paws, band = d.scramble.cos_band_p10_p50_p90;
  $('#order-note').textContent = `Benchmark: scrambling every word of a typical sentence still leaves ${f2(band.shuffle[1][12])} similarity to the original. On PAWS-X, sentence pairs that share their words but do not always mean the same thing, the score barely tells real paraphrases from re-ordered look-alikes (${f2(w.cosine_auc.all[12])}, where 0.50 is a coin flip). PAWS-X has no Hindi.`;
}

function renderMapCard() {
  const pct = v => (v * 100).toFixed(1) + '%';
  const m = cloud.meta.metrics, host = $('#map-table'); host.replaceChildren();
  const t = el('table', 'metrics'), head = el('tr');
  ['Map', 'Trust@10 (1 = perfect)', 'Neighbors kept', 'Translation found', 'Language guessable (chance 25%)'].forEach(h => head.append(el('th', null, h)));
  const ts = cloud.tsne;
  const ov = k => (k === 'tsne' ? pct(ts.placement.heldout_overlap10_neighbour_centre) + '*' : placement ? pct(placement.heads[k].heldout_overlap10_head) : '–');
  const th = el('thead'); th.append(head); t.append(th);
  const tb = el('tbody');
  [['pca', 'PCA'], ['fc', 'Linear'], ['ae', 'Autoencoder'], ...(ts ? [['tsne', 't-SNE*']] : [])].forEach(([k, name]) => {
    const tr = el('tr', state.head === k ? 'on' : ''); [name, m[k].trust10.toFixed(2), ov(k), pct(m[k].retrieval_p1_mean), pct(m[k].language_probe_acc)].forEach(v => tr.append(el('td', null, v))); tb.append(tr);
  });
  const tr = el('tr', 'ref'); ['Full 384-d', '1.00', '100.0%', pct(m.full_384d.retrieval_p1_mean), pct(m.full_384d.language_probe_acc)].forEach(v => tr.append(el('td', null, v))); tb.append(tr);
  t.append(tb); host.append(t);
  const note = el('p', 'note');
  note.textContent = `Trust@10: a forgiving score for whether map neighbors are real neighbors (1 is perfect). Neighbors kept: of a sentence’s 10 real nearest neighbors, how many stay nearest on the map. It is low, so never read fine distances off the picture. Translation found: an English sentence’s nearest other-language point is its own translation. Language guessable: how often a simple model can name the language from the position (25% is chance; the full space scores ${Math.round(m.full_384d.language_probe_acc * 100)}%).`;
  if (ts) note.textContent += ' *t-SNE saw every sentence when it was fitted, so its row is not a fair comparison, and only its nearby points mean anything.';
  if (state.result) { const R = state.result; const oa = cloud.localOverlap(state.head, R.an.eA, cloud.project(state.head, R.an.eA)), ob = cloud.localOverlap(state.head, R.an.eB, cloud.project(state.head, R.an.eB)); note.textContent += ` For your sentences, ${Math.round(oa * 10)} of A’s and ${Math.round(ob * 10)} of B’s 10 true nearest neighbors are also their nearest on this map.`; }
  host.append(note);
  if (!state.result) renderLens();
}

function renderLens() {
  const lens = cloud.meta.layer_lens, R = state.result;
  const series = [
    { name: 'Translation pairs', short: 'translations', color: css('--series-3'), values: lens.map(d => d.translation_cos) },
    { name: 'Unrelated pairs', short: 'unrelated', color: css('--muted'), values: lens.map(d => d.random_cos) },
  ];
  if (R) series.push({ name: 'Your A and B', short: 'A and B', color: css('--text'), values: R.an.layerSim, emph: true });
  lineChart($('#lens-chart'), series, lens.map(d => String(d.layer)), { yDomain: [Math.min(0, ...series.flatMap(s => s.values)) < -0.01 ? -0.25 : 0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], xTitle: 'transformer layer', xEvery: 2, endLabels: true, xTip: v => `Layer ${v} · cosine similarity`, aria: 'Similarity by layer' });
  const drop = lens.find(d => d.language_probe_acc < 0.995), lastFull = drop ? drop.layer - 1 : 12, p = l => Math.round(lens[l].language_probe_acc * 100);
  if (MODEL.wording.lensNote === 'mlm') {                    // per-model note: read from this model's lens; the default model's numbers from the registry
    const L = lens[lens.length - 1], gap = L.translation_cos - L.random_cos, A = MODEL.lensCompare;
    $('#lens-note').textContent = 'Average cosine across 816 held-out sentences. Every pair scores high on this model because all its vectors point the same way; compare the two lines, not their height. '
      + `This model was only trained to fill in masked words. A linear probe names the language with ${p(L.layer)}% accuracy even at layer ${L.layer} (chance 25%${A ? `; ${Math.round(A.probe12 * 100)}% for the default model` : ''}), `
      + `and translations score only ${f2(gap)} above unrelated pairs${A ? ` (default model: ${f2(A.gap12)})` : ''}. The languages stay apart. This shows language identity, not whether meaning differs across languages.`;
    return;
  }
  $('#lens-note').textContent = `Lines: average similarity of translation pairs and unrelated pairs at each layer (816 test sentences). A simple language guesser is right 100% of the time up to layer ${lastFull} and ${p(12)}% at layer 12 (chance 25%): the last layers hide most of the language and push unrelated sentences apart.`;
}
