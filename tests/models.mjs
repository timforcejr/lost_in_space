// Title: models.mjs, the model selector test (registry, switch, fallback, storage that throws, pop-out follows).
// What it checks:
//   0. registry (web/js/models.js in Node): ids a / b / c, C disabled, B's folders, every feature flag a boolean, A = the shipped folders;
//   1. against the given server (real assets only): A by default, picker shows A, title "Lost in Space", no request for any mlm/ file;
//   2. same server, B's files made absent (every mlm/ request answered 404 by interception): picking B falls back to A within 70 s (?model=a&fallback=b&why=missing, chip + descriptor say so, storage 'a'), no page error;
//   3. B with its real assets (web/model/mlm, web/model/mgrad_mlm, web/data/mlm): pick B -> reload -> B loaded: chip, title, band, integrated
//      gradients computed on B (parts add up to the score change) with the "not benchmarked" note, "measured on the default model" labels on the benchmark and
//      sanity cards, B's own order probe, t-SNE tab and placement, PCA default, lens note per model (no "weakened in the last layers"), overlap-aware verdict on
//      EN/ES, Unrelated and Word order; no request for model A's weights, A's gradient model or A's data files; reload keeps B; ?model=a returns A (no mlm/ request);
//   4. B's weights fail to download (request aborted): falls back with why=error within 70 s;
//   5. localStorage that throws: A boots, picking B carries ?model=b in the address and loads B;
//   6. pop-out follows: main (A) pops out the map, then picks B: both windows reload into B; the pop-out shows the model tag.
// Screenshots: model_header_A.png, model_header_B_fallback.png, model_header_B.png (B loaded), model_B_full.png, model_B_map.png, model_B_lens.png, model_B_verdict.png, model_popout_B.png.
// Run:  node tests/models.mjs http://127.0.0.1:8000/ [outDir]      (exit 0 = pass; run from the repo root)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { MODELS, byId } from '../web/js/models.js';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const out = process.argv[3] || 'tests/out_popout';
fs.mkdirSync(out, { recursive: true });
const problems = [];
const chk = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) problems.push(m); };
const wait = ms => new Promise(r => setTimeout(r, ms));
const MLM = /\/(model|data)\/mlm\//;

// ---------------------------------------------------------------- 0. registry
{
  const ids = MODELS.map(m => m.id);
  chk(ids.join() === 'a,b,c' && MODELS.length <= 3, 'registry has a, b, c (at most three): ' + ids.join());
  chk(!byId('c') && byId('a') && byId('b'), 'C is disabled, A and B enabled');
  const A = byId('a'), B = byId('b');
  chk(A.modelDir === 'mmini' && A.tokenizerDir === 'mmini' && A.gradDir === 'mgrad' && A.dataDir === './data/', 'A points at the shipped folders');
  chk(B.modelDir === 'mlm' && B.tokenizerDir === 'mmini' && B.dataDir === './data/mlm/', 'B points at model/mlm, shares A\'s tokenizer, data/mlm');
  const keys = ['ig', 'tsne', 'placement', 'orderProbe', 'bench', 'sanity'];
  chk(MODELS.every(m => keys.every(k => typeof m.features[k] === 'boolean')), 'every model has a boolean flag for ' + keys.join(', '));
  chk(Object.values(A.features).every(Boolean) && !A.wording.overlapVerdict && A.wording.lensNote === 'default' && A.wording.defaultHead === 'ae', 'A: every feature on, default wording, autoencoder map');
  chk(B.wording.defaultHead === 'pca' && B.wording.overlapVerdict && B.wording.lensNote === 'mlm' && B.lensCompare && B.lensCompare.probe12 > 0.3 && B.lensCompare.probe12 < 0.5, 'B: PCA default, overlap verdict, per-model lens note, lensCompare from A');
}

const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', protocolTimeout: 900000,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });

/** A page in its own browser context (its own localStorage). allow: RegExps for console / http lines that are expected in this section. */
async function open(ctx, base, q = '', { allow = [], init = null, intercept = null, name = 'page' } = {}) {
  const page = await ctx.newPage();
  await page.setViewport({ width: 1500, height: 1000, deviceScaleFactor: 1 });
  watch(page, name, allow);
  if (init) await page.evaluateOnNewDocument(init);
  if (intercept) { await page.setRequestInterception(true); page.on('request', r => intercept(r) || r.continue()); }
  await page.goto(base + q, { waitUntil: 'domcontentloaded' });
  return page;
}
function watch(page, name, allow = []) {
  page.reqs = [];
  const ok = s => allow.some(re => re.test(s));
  page.on('request', r => page.reqs.push(r.method() + ' ' + r.url()));
  page.on('console', m => { const s = `[${name} ${m.type()}] ${m.text().slice(0, 300)}`; if (['error', 'warning'].includes(m.type()) && !ok(s)) problems.push(s); });
  page.on('pageerror', e => problems.push(`[${name} pageerror] ${String(e).slice(0, 300)}`));
  page.on('response', r => { const s = `[${name} http ${r.status()}] ${r.url()}`; if (r.status() >= 400 && !ok(s)) problems.push(s); });
}
/** Wait (across reloads) until the page has a first result and the chain (deletion, gradients, order) is done. */
async function ready(page, timeout = 300000) {
  const t0 = Date.now();
  for (;;) {
    try { if (await page.evaluate(() => !!(window.__mm && window.__mm.state.result && !window.__mm.state.busy && window.__mm.orderDone))) return true; } catch { /* navigating */ }
    if (Date.now() - t0 > timeout) return false;
    await wait(300);
  }
}
async function until(page, fn, timeout = 70000) {
  const t0 = Date.now();
  for (;;) { try { if (await page.evaluate(fn)) return Date.now() - t0; } catch { /* navigating */ } if (Date.now() - t0 > timeout) return -1; await wait(200); }
}
const info = page => page.evaluate(() => ({ model: document.documentElement.dataset.model, alt: document.documentElement.classList.contains('model-alt'), title: document.title,
  sel: document.querySelector('#model-pick').value, opts: [...document.querySelectorAll('#model-pick option')].map(o => o.value).join(), desc: document.querySelector('#model-desc').textContent,
  chip: document.querySelector('#status-text').textContent, search: location.search, store: (() => { try { return localStorage.getItem('lis.model'); } catch { return 'THROWS'; } })(),
  head: window.__mm?.state.head, igDisabled: document.querySelector('#tabs button[data-m="ig"]').disabled }));
const shotHeader = async (page, file) => { const h = await page.$('header.top'); await h.screenshot({ path: `${out}/${file}` }); };
const pick = (page, id) => page.evaluate(id => { const s = document.querySelector('#model-pick'); s.value = id; s.dispatchEvent(new Event('change', { bubbles: true })); }, id);
const example = async (page, name) => { await page.evaluate(n => { window.__mm.orderDone = false; [...document.querySelectorAll('#examples button')].find(b => b.textContent === n).click(); }, name); await wait(300); return ready(page); };
const OVERLAP = /about as close as a typical translation pair, but on this model unrelated pairs score almost as high\.|this model barely separates translations from unrelated sentences here\./;
const A_VERDICT = /the model places these/;

// ---------------------------------------------------------------- 1. A by default (real server)
{
  const ctx = await browser.createBrowserContext();
  const p = await open(ctx, url, '', { name: 'A' });
  chk(await ready(p), 'A boots and runs the first pair');
  const i = await info(p);
  chk(i.model === 'a' && !i.alt && i.sel === 'a' && i.opts === 'a,b' && i.title === 'Lost in Space', 'A loaded by default; picker lists a,b with A selected; title unchanged: ' + JSON.stringify(i));
  chk(i.desc === 'trained on translations; this is the model on the poster.', 'A descriptor line');
  chk(!i.igDisabled && !(await p.$('#bench-flag, #sanity-flag, #ig-off-note')) && await p.$eval('#model-tag', e => e.offsetParent === null && /default|MiniLM/i.test(e.textContent)).catch(() => true), 'A: IG tab enabled, no model labels (the pop-out tag exists but is hidden in the main window)');
  chk(i.head === 'ae', 'A: autoencoder map by default (as before)');
  chk(!p.reqs.some(r => MLM.test(r)), 'A: no request for any mlm/ file');
  await shotHeader(p, 'model_header_A.png');
  await p.close();   // every open window follows a model switch (BroadcastChannel), so section 1's tab must not see section 2's pick

  // ---------------------------------------------------------------- 2. B picked, B's files absent -> fallback to A
  // B's files are simulated as absent (404 for every mlm/ request), whether or not they exist on this server
  const allow = [/http 404\] .*\/(model|data)\/mlm\//, /Failed to load resource.*404/];
  const q = await open(ctx, url, '', { name: 'B-missing', allow, intercept: r => (MLM.test(r.url()) ? (r.respond({ status: 404, contentType: 'text/plain', body: 'absent (test)' }), true) : false) });
  await ready(q);
  const t0 = Date.now();
  await pick(q, 'b');
  const ms = await until(q, () => location.search.includes('fallback=b') && document.documentElement.dataset.model === 'a', 70000);
  chk(ms >= 0 && Date.now() - t0 < 70000, `picking B without its files falls back to A within 70 s (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  await ready(q);
  const j = await info(q);
  chk(/model=a/.test(j.search) && /fallback=b/.test(j.search) && /why=missing/.test(j.search) && j.store === 'a' && j.sel === 'a', 'fallback address and storage: ' + JSON.stringify(j));
  chk(/masked-word model files were not found; showing the default model/.test(j.desc), 'descriptor says B was not found: ' + j.desc);
  const chipNow = await q.evaluate(() => document.querySelector('#status-text').textContent);
  chk(/not found; showing the default model|Ready/.test(chipNow), 'chip after the fallback: ' + chipNow);
  await shotHeader(q, 'model_header_B_fallback.png');
  const sims = await q.evaluate(() => window.__mm.state.result.an.sim);
  chk(Math.abs(sims - 0.972) < 0.0015, 'after the fallback, A gives the same EN/ES score as before (0.972): ' + sims.toFixed(3));
  await ctx.close();
}

// ---------------------------------------------------------------- 3. B with its real assets
{
  const ctx = await browser.createBrowserContext();
  const p = await open(ctx, url, '', { name: 'B' });
  chk(await ready(p), 'A boots');
  const before = p.reqs.length;
  await pick(p, 'b');
  await until(p, () => document.documentElement.dataset.model === 'b', 30000);
  chk(await ready(p), 'B loads after the reload and runs the first pair (gradients and order included)');
  let i = await info(p);
  chk(i.model === 'b' && i.alt && i.sel === 'b' && i.title === 'Lost in Space · masked-word model' && i.store === 'b', 'B loaded: data-model, band, picker, title, storage: ' + JSON.stringify(i));
  chk(i.desc === 'trained on masked words only, never on translations.', 'B descriptor line');
  chk(/Masked-word MiniLM/.test(i.chip), 'chip names the masked-word model: ' + i.chip);
  const g = await p.evaluate(() => { const R = window.__mm.state.result; return R.grad && { sim: R.an.sim, gsim: R.grad.A.sim, sum: R.grad.A.igSum, target: R.grad.A.igTarget, err: window.__mm.state.gradError }; });
  chk(!i.igDisabled && g && Math.abs(g.gsim - g.sim) < 0.03 && Math.abs(g.sum - g.target) < 0.05, 'B integrated gradients run on B: ' + JSON.stringify(g));
  await p.bringToFront(); await p.click('#tabs button[data-m="ig"]'); await wait(300);
  chk(/Computed on this model; not benchmarked/.test(await p.$eval('#method-note', e => e.textContent)), 'IG tab note: computed on this model, not benchmarked');
  await p.click('#tabs button[data-m="occlusion"]');
  const lab = await p.evaluate(() => ({ bench: document.querySelector('#bench-flag')?.textContent, sanity: document.querySelector('#sanity-flag')?.textContent,
    probeGone: !document.querySelector('#order-probe-chart') && !document.querySelector('#order-probe-note'),
    lens: document.querySelector('#lens-note').textContent, honesty: document.querySelector('#honesty').textContent, tsneHidden: document.querySelector('#seg-head button[data-v="tsne"]').hidden,
    pcaOn: document.querySelector('#seg-head button[data-v="pca"]').classList.contains('on'), aeOn: document.querySelector('#seg-head button[data-v="ae"]').classList.contains('on'),
    verdict: document.querySelector('#sim-text').textContent, poster: /as on the poster/i.test(document.body.innerText), mapRows: [...document.querySelectorAll('#map-table tbody tr')].map(r => r.cells[0].textContent + ':' + r.cells[2].textContent).join(' '),
    cloudModel: window.__mm.cloud.meta.model }));
  const measured = 'Measured on the default model (paraphrase-multilingual-MiniLM), not on the loaded model.';
  chk(lab.cloudModel === 'microsoft/Multilingual-MiniLM-L12-H384', 'cloud data is B\'s own: ' + lab.cloudModel);
  chk(lab.bench === measured && lab.sanity === measured, 'benchmark and sanity cards labeled "measured on the default model"');
  const vis = await p.evaluate(() => ['#bench-flag', '#sanity-flag'].map(sel => { const e = document.querySelector(sel); if (!e) return sel + ' missing'; const r = e.getBoundingClientRect(), c = getComputedStyle(e); return (c.display === 'none' || c.visibility === 'hidden' || r.width < 20 || r.height < 8 || e.offsetParent === null) ? sel + ' hidden ' + JSON.stringify([c.display, r.width, r.height]) : ''; }).filter(Boolean));
  chk(vis.length === 0, 'B: the "measured on the default model" labels are visible (non-zero size, in the page flow): ' + vis.join('; '));
  chk(lab.probeGone, 'word-order probe chart and note are not on the page');
  chk(!/weakened in the last layers/.test(lab.lens) && /compare the two lines, not their height/.test(lab.lens) && /100% accuracy even at layer 12/.test(lab.lens) && /40% for the default model/.test(lab.lens) && /only 0\.05 above unrelated pairs \(default model: 0\.78\)/.test(lab.lens), 'B lens note (from the data, no "weakened"): ' + lab.lens);
  chk(!/semantic drift|meaning drift|misunderstand/i.test(lab.lens + lab.verdict), 'no drift-of-meaning wording');
  chk(/PCA has none/.test(lab.honesty) && lab.pcaOn && !lab.aeOn && !lab.tsneHidden, 'B: PCA default, map note clause, t-SNE tab available');
  chk(/t-SNE\*:/.test(lab.mapRows) && !/: –/.test(lab.mapRows.replace(/Full 384-d:[^ ]*/, '')), 'map card: B\'s t-SNE row and held-out overlap column filled: ' + lab.mapRows);
  chk(OVERLAP.test(lab.verdict) && !A_VERDICT.test(lab.verdict), 'EN/ES verdict is overlap-aware: ' + lab.verdict);
  chk(!lab.poster, 'no "as on the poster"');
  const after = p.reqs.slice(before);
  chk(!after.some(r => /\/model\/(mgrad|mmini\/onnx)\//.test(r)), 'B: model A\'s weights and A\'s gradient model are not requested (one model per window)');
  chk(!after.some(r => /\/data\/(cloud\.json|cloud_emb|tsne\.json|placement\.json|order\.json)/.test(r)), 'B: none of A\'s per-model data files requested');
  chk(['cloud.json', 'tsne.json', 'placement.json', 'order.json'].every(f => after.some(r => r.includes('/data/mlm/' + f))) && after.some(r => r.includes('/model/mgrad_mlm/')), 'B: its own cloud, t-SNE, placement, order and gradient files are used');
  await shotHeader(p, 'model_header_B.png');
  await p.screenshot({ path: `${out}/model_B_full.png`, fullPage: true });
  await (await p.$('.blk[data-blk=map]')).screenshot({ path: `${out}/model_B_map.png` });
  await (await p.$('.blk[data-blk=trustm]')).screenshot({ path: `${out}/model_B_lens.png` });
  await (await p.$('.why .hero')).screenshot({ path: `${out}/model_B_verdict.png` });
  for (const n of ['Unrelated', 'Word order']) {
    chk(await example(p, n), 'B runs the ' + n + ' example');
    const v = await p.$eval('#sim-text', e => e.textContent);
    chk(OVERLAP.test(v), `${n}: overlap-aware verdict: ${v}`);
  }
  // reload keeps B; ?model=a brings A back
  await p.reload({ waitUntil: 'domcontentloaded' });
  chk(await ready(p) && (await info(p)).model === 'b', 'reload keeps B (saved per device)');
  const mark = p.reqs.length;
  await p.goto(url + '?model=a', { waitUntil: 'domcontentloaded' });
  chk(await ready(p), '?model=a: A boots');
  i = await info(p);
  chk(i.model === 'a' && !i.alt && i.store === 'a' && i.title === 'Lost in Space' && !i.igDisabled, '?model=a returns A and resets storage: ' + JSON.stringify(i));
  chk(!p.reqs.slice(mark).some(r => MLM.test(r)), 'A after B: no request for any mlm/ file');
  await ctx.close();
}

// ---------------------------------------------------------------- 4. B's weights fail to download -> fallback (why=error)
{
  const ctx = await browser.createBrowserContext();
  const allow = [/model .* failed to load|falling back/i, /Failed to load resource|net::ERR_FAILED|ERR_FAILED/, /http 4\d\d/, /onnx|Error|error/];
  const p = await open(ctx, url, '?model=b', { name: 'B-error', allow, intercept: r => (r.method() === 'GET' && /\/model\/mlm\/onnx\//.test(r.url()) ? (r.abort('failed'), true) : false) });
  const t0 = Date.now();
  const ms = await until(p, () => location.search.includes('fallback=b') && location.search.includes('why=error'), 70000);
  chk(ms >= 0, `B download failure falls back to A within 70 s (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  chk(await ready(p), 'A runs after the failure fallback');
  const i = await info(p);
  chk(i.model === 'a' && /Could not load the masked-word model; showing the default model\./.test(i.desc) && i.store === 'a', 'error fallback text and storage: ' + JSON.stringify(i));
  await ctx.close();
}

// ---------------------------------------------------------------- 5. localStorage throws
{
  const ctx = await browser.createBrowserContext();
  const init = () => { Object.defineProperty(window, 'localStorage', { configurable: true, get() { throw new DOMException('denied', 'SecurityError'); } }); };
  const p = await open(ctx, url, '', { name: 'nostore', init });
  chk(await ready(p) && (await info(p)).model === 'a', 'storage that throws: A boots');
  await pick(p, 'b');
  await until(p, () => document.documentElement.dataset.model === 'b', 30000);
  chk(await ready(p), 'storage that throws: B loads');
  const i = await info(p);
  chk(i.model === 'b' && /model=b/.test(i.search) && i.store === 'THROWS', 'storage that throws: the choice rides in ?model=b: ' + i.search);
  await ctx.close();
}

// ---------------------------------------------------------------- 6. pop-out follows the main window's choice
{
  const ctx = await browser.createBrowserContext();
  const main = await open(ctx, url, '', { name: 'main' });
  chk(await ready(main), 'pop-out test: main (A) ready');
  await main.bringToFront();
  const popTarget = ctx.waitForTarget(t => t.url().includes('view=graphic'), { timeout: 30000 });
  await main.click('#btn-popout');
  const pop = await (await popTarget).page();
  watch(pop, 'pop');
  chk(await ready(pop) && (await info(pop)).model === 'a', 'pop-out opened with A');
  await pick(main, 'b');
  const tm = await until(main, () => document.documentElement.dataset.model === 'b', 60000);
  const tp = await until(pop, () => document.documentElement.dataset.model === 'b', 60000);
  chk(tm >= 0 && tp >= 0, 'picking B in main reloads both windows into B');
  chk(await ready(main) && await ready(pop), 'both windows run B');
  const ip = await info(pop);
  chk(/view=graphic/.test(ip.search) && ip.model === 'b', 'pop-out kept ?view=graphic: ' + ip.search);
  chk(await pop.$eval('#model-tag', e => getComputedStyle(e).display !== 'none' && e.textContent === 'Model: Masked-word MiniLM'), 'pop-out shows the model tag');
  const same = await Promise.all([main, pop].map(x => x.evaluate(() => window.__mm.state.result.an.sim.toFixed(3))));
  chk(same[0] === same[1], 'both windows show the same score: ' + same.join(' / '));
  await pop.screenshot({ path: `${out}/model_popout_B.png` });
  await ctx.close();
}

await browser.close();
console.log(problems.length ? `\n${problems.length} problem(s):\n` + problems.join('\n') : '\nall model-selector checks passed');
process.exit(problems.length ? 1 : 0);
