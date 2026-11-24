// Headless-Chrome smoke test: load the page, run a pair, check numbers and console, take screenshots.
// Run:  node tests/smoke.mjs http://127.0.0.1:8000/ [outDir]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const out = process.argv[3] || 'tests/out';
fs.mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100'] });
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1100, deviceScaleFactor: 1 });
const problems = [], offsite = [], mlmReqs = [];
page.on('console', m => { if (['error', 'warning'].includes(m.type())) problems.push(`[${m.type()}] ${m.text().slice(0, 300)}`); });
page.on('pageerror', e => { console.error('[pageerror] ' + String(e).slice(0, 400)); process.exit(1); });   // a script error stops the page: fail at once instead of timing out
page.on('request', r => { if (/\/(model|data)\/mlm\//.test(r.url())) mlmReqs.push(r.url()); });
page.on('request', r => { const u = new URL(r.url()); if (u.hostname !== new URL(url).hostname && u.protocol !== 'data:' && u.protocol !== 'blob:') offsite.push(r.url()); });
page.on('response', r => { if (r.status() >= 400) problems.push(`[http ${r.status()}] ${r.url()}`); });

const run = async (a, b, name, extra) => {
  await page.evaluate((a, b) => { window.__mm && (window.__mm.deletionDone = false); document.querySelector('#in-a').value = a; document.querySelector('#in-b').value = b; document.querySelector('#form').requestSubmit(); }, a, b);
  // the chain is: first result -> deletion test -> gradient model (integrated gradients) -> word-order test
  await page.waitForFunction(a => window.__mm && window.__mm.state.result && window.__mm.state.result.a === a && window.__mm.orderDone, { timeout: 240000 }, a);
  if (extra) await extra();
  await new Promise(r => setTimeout(r, 1800));
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
  return page.evaluate(() => { const R = window.__mm.state.result; return { sim: R.an.sim, total: R.an.wm.total, ms: Math.round(R.ms), wordsA: R.gA.map(g => g.text), wordsB: R.gB.map(g => g.text),
    occA: R.an.scores.occlusion.A.map(v => +v.toFixed(3)), pairA: R.an.scores.pair.A.map(v => +v.toFixed(3)), rollA: R.an.scores.rollout.A.map(v => +v.toFixed(3)),
    layerSim: R.an.layerSim.map(v => +v.toFixed(3)),
    grad: R.grad && { ms: Math.round(R.grad.ms), sim: R.grad.A.sim, igSum: R.grad.A.igSum, igTarget: R.grad.A.igTarget, igA: R.an.scores.ig.A, igB: R.an.scores.ig.B,
      tokensA: R.A.tokens, idsA: R.A.ids, idsB: R.B.ids }, gradError: window.__mm.state.gradError || null,
    order: R.order && { text: R.order.variants[0].text, cos12: R.order.cosShuffle[12], cos0: R.order.cosShuffle[0], simOther: R.order.simOtherShuffle, probeNatural: R.order.probeNatural.map(v => v == null ? null : +v.toFixed(3)), probeShuffle: R.order.probeShuffle && R.order.probeShuffle.map(v => +v.toFixed(3)) },
    neighA: R.neighA.slice(0, 2).map(d => [+d.sim.toFixed(2), window.__mm.cloud.meta.points[d.index].t.slice(0, 60)]), delArea: R.del && R.del.area }; });
};

const t0 = Date.now();
await page.goto(url, { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => window.__mm && window.__mm.orderDone, { timeout: 300000 });
console.log('first result after', ((Date.now() - t0) / 1000).toFixed(1), 's');
const results = {};
const sleep0 = ms => new Promise(r => setTimeout(r, ms));
results.en_es = await run('The cat sleeps on the sofa.', 'El gato duerme en el sofá.', '01_en_es');
{ // color default is Topic; Language stays visible as point shapes + the language switches
  const d = await page.evaluate(() => ({ cb: window.__mm.state.colorBy, on: [...document.querySelectorAll('#seg-color button.on')].map(b => b.dataset.v), tog: document.querySelectorAll('#lang-toggles button').length, leg: document.querySelectorAll('#legend3d button').length, pressed: document.querySelector('#seg-color button[data-v="topic"]').getAttribute('aria-pressed') }));
  const chk = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) problems.push('[topic default] ' + m); };
  chk(d.cb === 'topic' && d.on.join() === 'topic' && d.pressed === 'true', 'color-by defaults to topic: ' + JSON.stringify(d));
  chk(d.tog === 4 && d.leg > 1, 'language switches and topic legend present on load');
}
{ // model selector: the default model is the one loaded
  const m = await page.evaluate(() => ({ id: document.documentElement.dataset.model, alt: document.documentElement.classList.contains('model-alt'), title: document.title, sel: document.querySelector('#model-pick').value, ig: !document.querySelector('#tabs button[data-m="ig"]').disabled, labels: document.querySelectorAll('.model-flag').length }));
  const chk = (c, msg) => { console.log((c ? 'PASS ' : 'FAIL ') + msg); if (!c) problems.push('[model] ' + msg); };
  chk(m.id === 'a' && !m.alt && m.title === 'Lost in Space' && m.sel === 'a' && m.ig && m.labels === 0, 'default model A loaded, unchanged page: ' + JSON.stringify(m));
}
results.en_hi = await run('I went to the hospital to see the doctor.', 'मैं डॉक्टर से मिलने अस्पताल गया।', '02_en_hi');
results.zh_hi = await run('今天的天气很好。', 'आज मौसम बहुत अच्छा है।', '03_zh_hi');
results.order = await run('The dog bit the man.', 'The man bit the dog.', '04_word_order', async () => { await page.click('#tabs button[data-m="ig"]'); });
results.negation = await run('I love this movie.', 'I do not love this movie.', '05_negation', async () => { await page.click('#tabs button[data-m="pair"]'); });
results.unrelated = await run('The stock market fell sharply today.', 'My grandmother bakes bread every Sunday.', '06_unrelated', async () => { await page.click('#seg-color button[data-v="topic"]'); await page.click('#seg-head button[data-v="pca"]'); });
const v = await page.$('#viewer'); await v.screenshot({ path: `${out}/07_viewer_topic_pca.png` });
// four methods only (Grad-CAM was removed 2026-11-19): four tabs, no Grad-CAM tab, slider or depth table, no Grad-CAM text in the page
const tabs = await page.evaluate(() => ({ m: [...document.querySelectorAll('#tabs button')].map(b => b.dataset.m), gone: !document.querySelector('#cam-layer, #cam-box, #depth, #camlayer-chart'), text: /grad-?cam/i.test(document.body.innerText) }));
if (tabs.m.join() !== 'occlusion,pair,ig,rollout' || !tabs.gone || tabs.text) problems.push('[methods] expected four method tabs and no Grad-CAM: ' + JSON.stringify(tabs));
await page.click('#tabs button[data-m="occlusion"]');
for (const [name, r] of Object.entries(results)) {
  if (r.gradError || !r.grad) { problems.push(`[grad] ${name}: gradient model failed: ${r.gradError}`); continue; }
  if (Math.abs(r.grad.sim - r.sim) > 0.03) problems.push(`[grad] ${name}: gradient model score ${r.grad.sim.toFixed(3)} vs main model ${r.sim.toFixed(3)}`);
  if (Math.abs(r.grad.igSum - r.grad.igTarget) > 0.05) problems.push(`[grad] ${name}: integrated gradients sum ${r.grad.igSum.toFixed(3)} vs target ${r.grad.igTarget.toFixed(3)}`);
  if (r.wordsA.filter(w => /[\p{L}\p{N}]/u.test(w)).length >= 3 && !r.order) problems.push(`[order] ${name}: no word-order result`);
}
{ // layer arrows (Tim 2026-11-18): step, clamp, disabled at the ends, label + slider + state in sync
  const chk = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) problems.push('[layer arrows] ' + m); };
  const S = () => page.evaluate(() => ({ v: document.querySelector('#layer').value, st: window.__mm.state.layer, lab: document.querySelector('#layer-label').textContent, pd: document.querySelector('#layer-prev').disabled, nd: document.querySelector('#layer-next').disabled,
    pa: document.querySelector('#layer-prev').getAttribute('aria-label'), na: document.querySelector('#layer-next').getAttribute('aria-label') }));
  let a = await S();
  chk(a.v === '12' && a.nd && !a.pd && a.pa === 'Previous layer' && a.na === 'Next layer', 'start at 12: next disabled, aria-labels ' + JSON.stringify(a));
  await page.click('#layer-prev'); a = await S();
  chk(a.v === '11' && a.st === 11 && a.lab === 'Layer 11' && !a.nd, 'prev -> 11, label synced ' + a.lab);
  await page.click('#layer-next'); a = await S();
  chk(a.v === '12' && a.st === 12 && a.lab.startsWith('Layer 12') && a.nd, 'next -> 12 and next disabled');
  await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 1; s.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.click('#layer-prev'); a = await S();
  chk(a.v === '0' && a.st === 0 && a.lab.startsWith('Layer 0') && a.pd && !a.nd, 'prev -> 0 and prev disabled: ' + a.lab);
  await page.evaluate(() => document.querySelector('#layer-prev').click()); a = await S();
  chk(a.v === '0' && a.st === 0, 'clamp at 0');
  await page.focus('#layer-next'); await page.keyboard.press('Enter'); a = await S();
  chk(a.v === '1' && a.st === 1, 'keyboard Enter on focused Next steps to 1');
  await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 12; s.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.evaluate(() => document.querySelector('#layer-next').click()); a = await S();
  chk(a.v === '12' && a.st === 12, 'clamp at 12');
  await page.click('#btn-play'); await sleep0(1000); await page.click('#layer-prev'); await sleep0(1500); a = await S();
  const l1 = a.st; await sleep0(1200); a = await S();
  chk(a.st === l1, 'arrow press stops Play (layer stays ' + l1 + ')');
  await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 12; s.dispatchEvent(new Event('input', { bubbles: true })); });
}
await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 3; s.dispatchEvent(new Event('input')); });
await new Promise(r => setTimeout(r, 800)); await v.screenshot({ path: `${out}/08_viewer_layer3.png` });
// t-SNE map + language switches
const sleep = ms => new Promise(r => setTimeout(r, ms));
await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 12; s.dispatchEvent(new Event('input')); });
await run('I went to the hospital to see the doctor.', 'मैं डॉक्टर से मिलने अस्पताल गया।', '10_tsne_en_hi', async () => { await page.click('#seg-color button[data-v="lang"]'); await page.click('#seg-head button[data-v="tsne"]'); });
await sleep(1500); await v.screenshot({ path: `${out}/11_viewer_tsne.png` });
const langCheck = async label => page.evaluate(label => { const { state, cloud } = window.__mm, R = state.result, pts = cloud.meta.points;
  const neigh = [...R.neighA, ...R.neighB].map(d => pts[d.index].l);
  return { label, langs: state.langs.slice(), neighborLangs: [...new Set(neigh)].sort(), nNeigh: neigh.length, leak: neigh.some(l => !state.langs[l]),
    listed: document.querySelectorAll('#neighbors .neigh-item').length, glosses: document.querySelectorAll('#neighbors .gloss').length }; }, label);
const toggles = [];
toggles.push(await langCheck('all on'));
await page.click('#lang-toggles button:nth-of-type(3)');                       // Hindi off
await sleep(900); toggles.push(await langCheck('hindi off')); await v.screenshot({ path: `${out}/12_viewer_tsne_no_hindi.png` });
await page.click('#lang-toggles button:nth-of-type(1)'); await page.click('#lang-toggles button:nth-of-type(4)'); // English, Chinese off
await sleep(600); toggles.push(await langCheck('spanish only'));
await page.click('#lang-toggles button:nth-of-type(2)');                       // last one: must refuse
await sleep(300); toggles.push(await langCheck('tried to switch the last one off'));
await page.screenshot({ path: `${out}/13_spanish_only.png`, fullPage: true });
for (const k of [1, 3, 4]) await page.click(`#lang-toggles button:nth-of-type(${k})`);
await sleep(600); toggles.push(await langCheck('all back on'));
for (const t of toggles) { if (t.leak) problems.push(`[langs] ${t.label}: a switched-off language is in the neighbor list`); if (t.nNeigh !== 16) problems.push(`[langs] ${t.label}: ${t.nNeigh} neighbors, expected 16`); if (!t.langs.some(Boolean)) problems.push(`[langs] ${t.label}: every language is off`); }
if (toggles[3].langs.join() !== 'false,true,false,false') problems.push('[langs] the last language could be switched off');
await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 4; s.dispatchEvent(new Event('input')); });
await sleep(800); await v.screenshot({ path: `${out}/14_viewer_tsne_layer4.png` });
await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 12; s.dispatchEvent(new Event('input')); });
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
await new Promise(r => setTimeout(r, 1200));
await page.screenshot({ path: `${out}/09_phone.png`, fullPage: true });
const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
const headCoords = await page.evaluate(() => Object.fromEntries(['pca', 'fc', 'ae'].map(h => [h, Array.from(window.__mm.cloud.coords(h).slice(0, 60))])));
fs.writeFileSync(`${out}/results.json`, JSON.stringify({ results, toggles, problems, offsite, phoneScrollWidth: scrollW, headCoords }, null, 1));
const brief = Object.fromEntries(Object.entries(results).map(([k, r]) => [k, { sim: +r.sim.toFixed(3), ms: r.ms, gradMs: r.grad && r.grad.ms, gradSim: r.grad && +r.grad.sim.toFixed(3), ig: r.grad && [+r.grad.igSum.toFixed(3), +r.grad.igTarget.toFixed(3)], scrambled: r.order && [r.order.text, +r.order.cos12.toFixed(3), +r.order.simOther.toFixed(3)], delArea: r.delArea }]));
console.log(JSON.stringify({ brief, toggles, problems, offsite, phoneScrollWidth: scrollW }, null, 1));
await browser.close();
if (mlmReqs.length) { problems.push('[model] requests for model B files while A is loaded: ' + mlmReqs.slice(0, 3).join(' ')); console.log('FAIL ' + problems[problems.length - 1]); } else console.log('PASS no request for model B files while A is loaded');
if (scrollW > 391) problems.push(`[layout] horizontal overflow at 390 px: scrollWidth ${scrollW}`);
process.exit(problems.length || offsite.length ? 1 : 0);
