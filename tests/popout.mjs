// Two-screen test. Page 1 ?view=map and page 2 ?view=explain in one Chrome; an example clicked in page 1
// must show the same pair and score in page 2; a method tab, a map head, layer 5 and a language switch must sync both
// ways; a third window must adopt the current state; the Pop out button must open ?view=explain and Dock must undo it.
// Run:  node tests/popout.mjs http://127.0.0.1:8000/ [outDir]      (exit 0 = pass)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const out = process.argv[3] || 'tests/out_popout';
fs.mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  protocolTimeout: 900000,   // several windows each load and run the model; a page can stay busy for minutes on a slow machine
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
const problems = [], log = [];
const say = s => { log.push(s); console.log(s); };
const fail = s => { problems.push(s); console.log('FAIL ' + s); };
const T = 300000;

async function open(path, name) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 1 });
  watchPage(page, name);
  await page.goto(new URL(path, url).href, { waitUntil: 'domcontentloaded' });
  return page;
}
function watchPage(page, name) {
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) problems.push(`[${name} ${m.type()}] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', e => problems.push(`[${name} pageerror] ${String(e).slice(0, 300)}`));
  page.on('response', r => { if (r.status() >= 400) problems.push(`[${name} http ${r.status()}] ${r.url()}`); });
}
const ready = p => p.waitForFunction(() => window.__mm && window.__mm.state.result && !window.__mm.state.busy, { timeout: T });
const st = p => p.evaluate(() => { const s = window.__mm.state; return { a: s.result.a, b: s.result.b, method: s.method, head: s.head, layer: s.layer, langs: s.langs.join(''), sim: document.querySelector('#sim').textContent, busy: s.busy }; });
const until = (p, fn, arg, what) => p.waitForFunction(fn, { timeout: 60000, polling: 100 }, arg).then(() => true, () => { fail('timed out waiting for ' + what); return false; });
const shown = (p, sel) => p.$eval(sel, e => getComputedStyle(e).display !== 'none' && e.offsetHeight > 0);
const click = async (p, sel) => { await p.bringToFront(); await p.click(sel); };   // headless Chrome sends mouse input only to the front tab
const setLayer = (p, v) => p.evaluate(v => { const e = document.querySelector('#layer'); e.value = String(v); e.dispatchEvent(new Event('input', { bubbles: true })); }, v);

const t0 = Date.now();
const p1 = await open('?view=map', 'p1');
await ready(p1); say(`page 1 ready after ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const p2 = await open('?view=explain', 'p2');
await ready(p2);
say(`both pages ready after ${((Date.now() - t0) / 1000).toFixed(1)} s`);

// layout of each mode
const lay = { p1: { viewer: await shown(p1, '#viewer'), composer: await shown(p1, '.composer'), why: await shown(p1, '.why'), cards: await shown(p1, '.cards') },
  p2: { viewer: await shown(p2, '#viewer'), composer: await shown(p2, '.composer'), why: await shown(p2, '.why'), cards: await shown(p2, '.cards') } };
say('layout ' + JSON.stringify(lay));
if (!(lay.p1.viewer && lay.p1.composer && !lay.p1.why && !lay.p1.cards)) fail('?view=map layout wrong');
if (!(!lay.p2.viewer && !lay.p2.composer && lay.p2.why && lay.p2.cards)) fail('?view=explain layout wrong');
for (const [n, p] of [['p1', p1], ['p2', p2]]) { const h = await p.$$eval('.help-btn', b => b.length); say(`${n} help buttons: ${h}`); }
const view1 = await p1.evaluate(() => location.search), view2 = await p2.evaluate(() => location.search);
say(`addresses keep the mode: p1 ${view1} p2 ${view2}`);

// 1. an example clicked in page 1 shows the same pair in page 2
await p1.evaluate(() => [...document.querySelectorAll('#examples button')].find(b => b.textContent === 'Word order').click());
const A = 'The dog bit the man.';
await until(p2, a => window.__mm.state.result.a === a && !window.__mm.state.busy, A, 'page 2 to take the example');
await until(p1, a => window.__mm.state.result.a === a && !window.__mm.state.busy, A, 'page 1 to run the example');
let s1 = await st(p1), s2 = await st(p2);
say(`example: p1 ${JSON.stringify([s1.a, s1.b, s1.sim])} p2 ${JSON.stringify([s2.a, s2.b, s2.sim])}`);
if (s1.a !== s2.a || s1.b !== s2.b) fail('pair differs between pages');
if (s1.sim !== s2.sim || !s1.sim || s1.sim === '–') fail(`#sim differs: ${s1.sim} vs ${s2.sim}`);
const search1 = await p1.evaluate(() => location.search + location.hash);
say(`p1 address after the run: ${search1.slice(0, 80)}`);
if (!search1.startsWith('?view=map#a=')) fail('page 1 lost ?view=map after a run');

// 2. tab in page 2 -> page 1; map head in page 1 -> page 2
await click(p2, '#tabs button[data-m="ig"]');
if (await until(p1, () => window.__mm.state.method === 'ig', null, 'page 1 method = ig')) say('tab click in p2 -> p1 method ig');
await click(p1, '#seg-head button[data-v="pca"]');
if (await until(p2, () => window.__mm.state.head === 'pca', null, 'page 2 head = pca')) say('Map click in p1 -> p2 head pca');

// 3. layer 5 both ways
await setLayer(p1, 5);
if (await until(p2, () => window.__mm.state.layer === 5, null, 'page 2 layer 5')) say('layer 5 p1 -> p2');
await setLayer(p2, 12);
if (await until(p1, () => window.__mm.state.layer === 12, null, 'page 1 layer 12')) say('layer 12 p2 -> p1');
await setLayer(p2, 5);
if (await until(p1, () => window.__mm.state.layer === 5, null, 'page 1 layer 5')) say('layer 5 p2 -> p1');

// 3b. layer arrows in page 1 sync to page 2 (the explain view hides the map, so its arrows are not shown; its state follows)
await click(p1, '#layer-next');
if (await until(p2, () => window.__mm.state.layer === 6, null, 'page 2 layer 6 after Next arrow')) say('Next arrow p1 -> p2 layer 6');
await click(p1, '#layer-prev'); await click(p1, '#layer-prev');
if (await until(p2, () => window.__mm.state.layer === 4, null, 'page 2 layer 4 after two Prev arrows')) say('Prev arrows p1 -> p2 layer 4');
await setLayer(p2, 12);
if (await until(p1, () => document.querySelector('#layer-next').disabled && document.querySelector('#layer-label').textContent.startsWith('Layer 12'), null, 'page 1 arrows follow remote layer 12')) say('remote layer 12 -> p1 Next disabled');
await setLayer(p2, 5); await until(p1, () => window.__mm.state.layer === 5, null, 'page 1 layer 5');
for (const [n, p] of [['p1', p1], ['p2', p2]]) { const cb = await p.evaluate(() => window.__mm.state.colorBy); if (cb !== 'topic') fail(`${n} color default is ${cb}, expected topic`); }

// 4. language switch (Hindi off) in page 1 -> page 2
await p1.evaluate(() => document.querySelectorAll('#lang-toggles button')[2].click());
if (await until(p2, () => window.__mm.state.langs.join('') === 'truetruefalsetrue', null, 'page 2 Hindi off')) say('language switch p1 -> p2');
await p1.evaluate(() => document.querySelectorAll('#lang-toggles button')[2].click());
await until(p2, () => window.__mm.state.langs.every(Boolean), null, 'page 2 Hindi back on');

// 5. a pair typed in page 2's... (page 2 has no boxes) - a neighbor click in page 2 changes the pair in page 1
const nb = await p2.evaluate(() => { const b = document.querySelector('#neighbors button, #neighbors [role="button"], #neighbors .item'); if (!b) return null; b.click(); return true; });
if (nb) {
  await until(p1, () => window.__mm.state.result.b !== 'The man bit the dog.' && !window.__mm.state.busy, null, 'page 1 to take page 2 neighbor click');
  await until(p2, () => !window.__mm.state.busy, null, 'page 2 idle');
  s1 = await st(p1); s2 = await st(p2);
  say(`neighbor click in p2: p1 b=${JSON.stringify(s1.b.slice(0, 50))} p2 b=${JSON.stringify(s2.b.slice(0, 50))} sim ${s1.sim}/${s2.sim}`);
  if (s1.b !== s2.b || s1.sim !== s2.sim) fail('neighbor click did not sync');
} else say('no clickable neighbor found in page 2 (skipped)');

// 6. a third window adopts the current state
await new Promise(r => setTimeout(r, 1500));
s1 = await st(p1);
const p3 = await open('', 'p3');
await ready(p3);
await until(p3, s => { const x = window.__mm.state; return x.result.a === s.a && x.result.b === s.b && x.method === s.method && x.head === s.head && x.layer === s.layer && !x.busy; }, s1, 'page 3 to adopt the snapshot');
const s3 = await st(p3);
say(`third window: ${JSON.stringify({ a: s3.a, b: s3.b.slice(0, 40), method: s3.method, head: s3.head, layer: s3.layer, sim: s3.sim })} (p1 sim ${s1.sim})`);
if (s3.sim !== s1.sim) fail('third window score differs');

// 7. Pop out (graphic only) / dock back, on page 3
const wait = ms => new Promise(r => setTimeout(r, ms));
const popOutNow = async (page) => {
  const t = new Promise(r => browser.once('targetcreated', r));
  await click(page, '#btn-popout');
  const tg = await Promise.race([t, wait(10000).then(() => null)]);
  const w = tg && await tg.page();
  if (w) watchPage(w, 'popwin');
  return w;
};
const full = p => p.evaluate(() => [document.documentElement.className, document.querySelector('#btn-popout').textContent, location.search]);
const isFull = f => !f[0].includes('pv-') && f[1] === 'Pop out' && f[2] === '';

const metrics = p => p.evaluate(() => { const bot = c => { const k = [...document.querySelectorAll(`.col[data-col="${c}"] > .blk`)].filter(b => b.offsetHeight > 0); return k.length ? Math.round(k[k.length - 1].getBoundingClientRect().bottom + scrollY) : 0; };
  return { page: document.documentElement.scrollHeight, c0: bot(0), c1: bot(1), n0: [...document.querySelectorAll('.col[data-col="0"] > .blk')].filter(b => b.offsetHeight > 0).length, n1: [...document.querySelectorAll('.col[data-col="1"] > .blk')].filter(b => b.offsetHeight > 0).length, whyW: Math.round(document.querySelector('.why').getBoundingClientRect().width) }; });
const geo = p => p.evaluate(() => { const m = document.querySelector('.blk[data-blk=map]'), c0 = document.querySelector('.col[data-col="0"]').getBoundingClientRect(), c1 = document.querySelector('.col[data-col="1"]').getBoundingClientRect(); return { mapDisplay: getComputedStyle(m).display, mapH: Math.round(m.getBoundingClientRect().height), mapW: Math.round(m.getBoundingClientRect().width), top0: Math.round(c0.top + scrollY), top1: Math.round(c1.top + scrollY), w0: c0.width, w1: c1.width }; });
const docked = await metrics(p3); say('docked metrics ' + JSON.stringify(docked));
const gDocked = await geo(p3); say('docked geometry ' + JSON.stringify(gDocked));
if (gDocked.mapH < 400 || gDocked.mapW < 1300 || Math.abs(gDocked.w0 - gDocked.w1) > 2 || gDocked.top0 < gDocked.mapH) fail('docked page: map row / equal columns wrong ' + JSON.stringify(gDocked));
await p3.screenshot({ path: `${out}/before_docked_full.png`, fullPage: true });
let p4 = await popOutNow(p3);
if (!p4) fail('Pop out did not open a window');
else {
  await p4.waitForFunction(() => location.search === '?view=graphic', { timeout: 20000 }).catch(() => fail('pop-up address is not ?view=graphic'));
  const cls = await full(p3);
  say(`opener after Pop out: ${JSON.stringify(cls)}`);
  if (!cls[0].includes('pv-main') || cls[1] !== 'Dock' || cls[2] !== '?view=main') fail('opener did not switch to ?view=main');
  await ready(p4);
  await until(p4, s => { const x = window.__mm.state; return x.result.a === s.a && x.result.b === s.b && x.method === s.method && !x.busy; }, s1, 'pop-up to adopt the state');
  { const mm = await Promise.all([p3, p4].map(p => p.evaluate(() => [document.documentElement.dataset.model, document.querySelector('#model-tag')?.textContent || ''])));
    say('loaded model (opener, pop-out): ' + JSON.stringify(mm));
    if (mm[0][0] !== 'a' || mm[1][0] !== 'a' || mm[1][1] !== 'Model: Translation-trained MiniLM') fail('opener and pop-out should both load the default model, and the pop-out tag names it: ' + JSON.stringify(mm)); }
  // layout: pop-out = only the graphic, filling the window; opener keeps boxes, settings, cards
  const L4 = { viewer: await shown(p4, '#viewer'), composer: await shown(p4, '.composer'), why: await shown(p4, '.why'), cards: await shown(p4, '.cards'), header: await shown(p4, '.top'),
    tl: await shown(p4, '.overlay.tl'), canvas: await shown(p4, '#viewer canvas'), dock: await shown(p4, '#btn-dock'), legend: await shown(p4, '#legend3d') };
  const L3 = { blk: await p3.$eval('.blk[data-blk="map"]', e => getComputedStyle(e).display), viewer: await shown(p3, '#viewer'), composer: await shown(p3, '.composer'), why: await shown(p3, '.why'), cards: await shown(p3, '.cards'),
    note: await shown(p3, '#popout-note'), dock: await p3.$eval('#btn-popout', e => e.textContent), tl: await shown(p3, '.overlay.tl').catch(() => false) };
  const C4 = {}; for (const sel of ['#legend3d', '#seg-color', '#seg-head', '#btn-words', '#lang-toggles', '#layer', '#btn-play', '#layer-prev', '#layer-next', '#layer-label', '#btn-reset']) C4[sel] = await shown(p4, sel);
  say('pop-out layout ' + JSON.stringify(L4)); say('pop-out controls ' + JSON.stringify(C4)); say('opener layout ' + JSON.stringify(L3));
  if (!(L4.viewer && L4.canvas && L4.dock && L4.legend && L4.tl && !L4.composer && !L4.why && !L4.cards && !L4.header)) fail('pop-out is not the whole map block');
  if (Object.entries(C4).some(([k, v]) => !v)) fail('pop-out is missing a control: ' + JSON.stringify(C4));
  if (!(L3.blk === 'none' && !L3.viewer && L3.composer && L3.why && L3.cards && L3.note && L3.dock === 'Dock')) fail('opener still has the map block in its layout / lacks the Dock chip');
  await wait(5500); const m3 = await metrics(p3); say('popped-out metrics ' + JSON.stringify(m3));
  const gOut = await geo(p3); say('popped-out geometry ' + JSON.stringify(gOut));
  if (gOut.mapDisplay !== 'none' || gOut.mapH !== 0) fail('map row still in the layout while popped out: ' + JSON.stringify(gOut));
  if (!(gOut.top0 < gDocked.top0 - gDocked.mapH + 120 && Math.abs(gOut.top0 - gOut.top1) < 2 && Math.abs(gOut.w0 - gOut.w1) <= 2)) fail('columns did not move up into the map row space / not equal: ' + JSON.stringify({ gDocked, gOut }));
  if (!(m3.page < docked.page - 400 && m3.n0 >= 2 && m3.n1 >= 2 && Math.abs(m3.c0 - m3.c1) < 500)) fail('no real space saving / columns not re-flowed: ' + JSON.stringify({ docked, m3 }));
  const nb = await p3.evaluate(() => [...document.querySelectorAll('.blk')].map(b => b.dataset.blk + ':' + Math.round(b.getBoundingClientRect().height)).join(' ')); say('blocks while popped out: ' + nb);
  const fit = async () => p4.evaluate(() => { const r = document.querySelector('#viewer').getBoundingClientRect(), c = document.querySelector('#viewer canvas'); return [Math.round(r.width) - innerWidth, Math.round(r.height) - innerHeight, c.clientWidth - innerWidth, c.clientHeight - innerHeight]; });
  say('pop-out fill (expect zeros): ' + JSON.stringify(await fit()));
  if ((await fit()).some(d => Math.abs(d) > 2)) fail('pop-out map does not fill its window');
  await p4.setViewport({ width: 900, height: 700 }); await wait(500);
  say('after resize (expect zeros): ' + JSON.stringify(await fit()));
  if ((await fit()).some(d => Math.abs(d) > 2)) fail('pop-out map does not follow a resize');
  // the pop-out's own controls drive its graphic and the opener's hidden controls
  await setLayer(p4, 3);
  if (await until(p3, () => window.__mm.state.layer === 3, null, 'opener layer 3 from the pop-out')) say('layer 3 pop-out -> opener');
  await click(p4, '#seg-color button[data-v="lang"]');
  if (await until(p3, () => window.__mm.state.colorBy === 'lang', null, 'opener color = language from the pop-out')) say('color pop-out -> opener');
  await p4.evaluate(() => document.querySelectorAll('#lang-toggles button')[2].click());
  await until(p3, () => window.__mm.state.langs.join('') === 'truetruefalsetrue', null, 'opener Hindi off from the pop-out');
  await click(p4, '#seg-color button[data-v="topic"]'); await setLayer(p4, 4);
  await until(p3, () => window.__mm.state.colorBy === 'topic' && window.__mm.state.layer === 4, null, 'opener topic / layer 4');
  // the opener drives the pop-out too
  await setLayer(p3, 3);
  if (await until(p4, () => window.__mm.state.layer === 3 && document.querySelector('#layer-label').textContent.startsWith('Layer 3'), null, 'pop-out layer 3')) say('layer 3 opener -> pop-out');
  await p3.evaluate(() => document.querySelector('#seg-color button[data-v="lang"]').click());   // the opener's controls are hidden while popped out
  if (await until(p4, () => window.__mm.state.colorBy === 'lang', null, 'pop-out color = language')) say('color opener -> pop-out');
  await p3.evaluate(() => document.querySelector('#seg-color button[data-v="topic"]').click());   // the opener's controls are hidden while popped out
  await until(p4, () => window.__mm.state.colorBy === 'topic', null, 'pop-out color = topic');
  await p4.screenshot({ path: `${out}/popout_graphic.png` }); await p3.bringToFront().catch(() => {}); await p3.screenshot({ path: `${out}/opener_main.png` });
  await p3.screenshot({ path: `${out}/after_popped_reflow.png`, fullPage: true }); await p3.screenshot({ path: `${out}/wide_popped_out.png`, fullPage: true });

  // 7a. closing the pop-out (window close) docks the map back
  await p4.close();
  if (await until(p3, () => !document.documentElement.className.includes('pv-') && document.querySelector('#btn-popout').textContent === 'Pop out', null, 'opener to dock after the pop-out closed')) say('closing the pop-out -> docked');
  const f1 = await full(p3); if (!isFull(f1)) fail('after closing the pop-out the opener is not the full page: ' + JSON.stringify(f1));
  if (!(await shown(p3, '#viewer canvas'))) fail('map canvas is not back after dock');
  await wait(800);
  const rest = await p3.evaluate(() => { const v = document.querySelector('#viewer'), c = v.querySelector('canvas'); return [Math.round(v.getBoundingClientRect().height), c.clientWidth - v.clientWidth, c.clientHeight - v.clientHeight, c.width > 100 && c.height > 100]; });
  say('opener map panel after dock (height, canvas diffs, canvas size ok): ' + JSON.stringify(rest));
  const kept = await p3.evaluate(() => { const s = window.__mm.state; return [s.colorBy, s.layer, s.langs.join('')]; }); say('settings kept after dock: ' + JSON.stringify(kept));
  if (kept[0] !== 'topic' || kept[1] !== 3 || kept[2] !== 'truetruefalsetrue') fail('settings did not carry across the dock: ' + JSON.stringify(kept));
  await wait(1500); const mAfter = await metrics(p3);   // the default arrangement is balanced by measured heights, so the unit counts per column may differ from before: it must be level again
   if (!(m3.page < mAfter.page - 100)) fail('popped-out page is not shorter than the docked page: ' + JSON.stringify({ m3, mAfter }));
  if (mAfter.whyW !== docked.whyW || Math.abs(mAfter.c0 - mAfter.c1) > Math.max(250, 0.15 * Math.max(mAfter.c0, mAfter.c1)) || mAfter.n0 < 1 || mAfter.n1 < 1) fail('layout not restored after dock: ' + JSON.stringify({ docked, mAfter }));
  if (rest[0] < 400 || Math.abs(rest[1]) > 2 || Math.abs(rest[2]) > 2 || !rest[3]) fail('map panel not restored / canvas not resized after dock: ' + JSON.stringify(rest));
  const gBack = await geo(p3); say('docked-again geometry ' + JSON.stringify(gBack));
  if (gBack.mapDisplay === 'none' || gBack.mapH < 400 || Math.abs((gBack.top0 - gBack.mapH) - (gDocked.top0 - gDocked.mapH)) > 120 || gBack.top0 - gBack.mapH < 150 || Math.abs(gBack.mapW - gDocked.mapW) > 2) fail('dock did not restore the map row / columns: ' + JSON.stringify({ gDocked, gBack }));
  await p3.screenshot({ path: `${out}/after_docked.png` });

  // 7b. the Dock button in the pop-out
  p4 = await popOutNow(p3); if (!p4) fail('second Pop out did not open'); else {
    await ready(p4);
    const closed = new Promise(r => p4.once('close', r));
    await p4.evaluate(() => document.querySelector('#btn-dock').click());
    await Promise.race([closed, wait(5000)]);
    if (await until(p3, () => !document.documentElement.className.includes('pv-'), null, 'opener to dock after the pop-out Dock button')) say('pop-out Dock button -> docked, window closed: ' + p4.isClosed());
    if (!p4.isClosed()) fail('Dock button did not close the pop-out');
  }

  // 7c. the Dock button in the opener closes the pop-out
  p4 = await popOutNow(p3); if (!p4) fail('third Pop out did not open'); else {
    await ready(p4);
    const closed = new Promise(r => p4.once('close', r));
    await click(p3, '#btn-popout');
    await Promise.race([closed, wait(5000)]);
    const f = await full(p3); say('after opener Dock: ' + JSON.stringify(f) + ' pop-out closed: ' + p4.isClosed());
    if (!isFull(f)) fail('opener Dock did not restore the page'); if (!p4.isClosed()) fail('opener Dock did not close the pop-out');
  }

  // 7d. reload of the opener while popped out: handle lost, heartbeat keeps it in ?view=main; closing the pop-out then docks it
  p4 = await popOutNow(p3); if (!p4) fail('fourth Pop out did not open'); else {
    await ready(p4);
    await p3.reload({ waitUntil: 'domcontentloaded' }); await ready(p3);
    await wait(5000);   // longer than the 3 s heartbeat limit: the pop-out is alive, so the opener must stay in main
    const f = await full(p3); say('opener after reload, pop-out alive: ' + JSON.stringify(f));
    if (!f[0].includes('pv-main') || f[2] !== '?view=main') fail('reloaded opener left ?view=main although the pop-out is alive');
    await p4.close();
    if (await until(p3, () => !document.documentElement.className.includes('pv-'), null, 'reloaded opener to dock (no handle)')) say('reloaded opener docked after the pop-out closed (message / heartbeat)');
    if (!isFull(await full(p3))) fail('reloaded opener not full page after dock');
  }

  // 7e. reload of the pop-out keeps ?view=graphic and does not leave the opener docked
  p4 = await popOutNow(p3); if (!p4) fail('fifth Pop out did not open'); else {
    await ready(p4);
    await p4.reload({ waitUntil: 'domcontentloaded' }); await ready(p4);
    await wait(4000);
    const f = await full(p3), a = await p4.evaluate(() => location.search);
    say(`after pop-out reload: opener ${JSON.stringify(f)} pop-out ${a}`);
    if (a !== '?view=graphic') fail('pop-out lost its mode on reload');
    if (!f[0].includes('pv-main')) fail('opener docked on a mere reload of the pop-out');
    await p4.close();
    await until(p3, () => !document.documentElement.className.includes('pv-'), null, 'opener to dock at the end');
  }
}

await new Promise(r => setTimeout(r, 1000));
await p1.screenshot({ path: `${out}/p1_map.png` });
await p2.screenshot({ path: `${out}/p2_explain.png`, fullPage: true });
fs.writeFileSync(`${out}/results.json`, JSON.stringify({ log, problems }, null, 1));
console.log(JSON.stringify({ problems }, null, 1));
await browser.close();
process.exit(problems.length ? 1 : 0);
