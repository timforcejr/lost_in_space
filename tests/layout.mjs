// Title: layout.mjs, two-column unit layout test. Drag and drop is driven with synthetic DragEvents (headless Chrome has no real drag session).
// Checks: default columns and aria labels; drag a card into the other column and below the explanation; drag to the end of a column; order / column / folded set kept in
// localStorage ("lis-layout-v2-d") across a reload; fold / expand (aria-expanded, header bar only) for a card and for the map (canvas re-measured, pop-out view unaffected);
// "Reset layout"; keyboard (arrows on the grip, buttons) with focus kept; a storage that throws; an old v1 value and a corrupt value are ignored; tablet width (one column, flattened order);
// phone width (no drag, no column buttons, own storage key). Screenshots in the out dir.
// Run:  node tests/layout.mjs http://127.0.0.1:8000/ [outDir]      (exit 0 = pass)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const out = process.argv[3] || 'tests/out_popout';
fs.mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const problems = [];
const fail = s => { problems.push(s); console.log('FAIL ' + s); };
const say = s => console.log(s);
const wait = ms => new Promise(r => setTimeout(r, ms));
const DEF = 'near,order|why,trustx,trustm,evid';                       // the arrangement the tests pin (the live default is balanced by measured heights, so it varies)
const PIN = () => { if (!sessionStorage.getItem('pin')) { sessionStorage.setItem('pin', '1'); localStorage.setItem('lis-layout-v2-' + (matchMedia('(max-width: 640px)').matches ? 'p' : 'd'), JSON.stringify({ v: 2, cols: [['near', 'order'], ['why', 'trustx', 'trustm', 'evid']], folded: [] })); } };
const CARDS = ['near', 'order', 'trustx', 'trustm', 'evid'];
const hsel = id => `.blk[data-blk=${id}] h2.blk-handle`;   // the drag handle and keyboard focus: every unit's title heading, inside its panel
const fsel = hsel;
async function open(w = 1400, init, q = '', h = 900) {
  const p = await browser.newPage(); await p.setViewport({ width: w, height: h });
  p.on('pageerror', e => problems.push('[pageerror] ' + String(e).slice(0, 200)));
  if (init) await p.evaluateOnNewDocument(init);
  await p.goto(url + q, { waitUntil: 'domcontentloaded' });
  await p.waitForSelector('.blk-head', { timeout: 20000 }); await wait(600);
  return p;
}
const lay = p => p.evaluate(() => [0, 1].map(c => [...document.querySelectorAll(`.col[data-col="${c}"] > .blk`)].map(b => b.dataset.blk).join(',')).join('|'));
const stored = (p, k = 'lis-layout-v2-d') => p.evaluate(k => { try { return localStorage.getItem(k); } catch { return 'ERR'; } }, k);
const rect = (p, id) => p.evaluate(id => { const r = document.querySelector(`.blk[data-blk=${id}]`).getBoundingClientRect(); return { l: Math.round(r.left), t: Math.round(r.top + scrollY), b: Math.round(r.bottom + scrollY), h: Math.round(r.height), w: Math.round(r.width) }; }, id);
const synth = (p, from, pt) => p.evaluate(({ from, pt, hs }) => {
  const dt = new DataTransfer(), g = document.querySelector(hs);
  const t = pt.to ? document.querySelector(`.blk[data-blk=${pt.to}]`) : document.querySelector(`.col[data-col="${pt.col}"]`), r = t.getBoundingClientRect();
  const x = r.left + r.width / 2, y = pt.to ? (pt.pos === 'b' ? r.bottom - 8 : r.top + 8) : r.bottom - 8;
  const ev = (type, el) => el.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, dataTransfer: dt }));
  ev('dragstart', g); ev('dragover', t); const marked = document.querySelectorAll('.drop-t, .drop-b, .drop-into').length; ev('drop', t); ev('dragend', g); return marked;
}, { from, pt, hs: hsel(from) });
const dragBlk = (p, from, to, pos) => synth(p, from, { to, pos });       // pos t / b: upper / lower half of the target unit
const dragCol = (p, from, col) => synth(p, from, { col });               // below the last unit of a column
const key = async (p, sel, k) => { await p.focus(sel); await p.keyboard.press(k); };
const grip = fsel;
const reload = async p => { await p.reload({ waitUntil: 'domcontentloaded' }); await p.waitForSelector('.blk-head'); await wait(600); };
const min = id => `.blk[data-blk=${id}] .blk-min`;
const expanded = (p, id) => p.$eval(min(id), e => e.getAttribute('aria-expanded'));

// the default (nothing saved) is balanced: both columns end about level once the cards have loaded; checked at three desktop widths
const bottoms = p => p.evaluate(() => [0, 1].map(c => { const k = document.querySelector(`.col[data-col="${c}"]`).lastElementChild.getBoundingClientRect(); return Math.round(k.bottom + scrollY); }));
let o;
for (const w of [1280, 1440, 1920]) {
  const d = await open(w, null, '', w === 1920 ? 1080 : 900);
  if (await d.$eval('#btn-reset-layout', e => !e.hidden)) fail('Reset layout shown at the default (' + w + ')');
  const first = await bottoms(d); let prev = '', same = 0;
  for (let k = 0; k < 40 && same < 3; k++) { await wait(700); const b = JSON.stringify(await bottoms(d)) + await lay(d); same = b === prev ? same + 1 : 0; prev = b; }   // stable for 3 polls = loaded
  const bt = await bottoms(d), gap = Math.abs(bt[0] - bt[1]), tall = Math.max(...bt), tol = Math.max(250, 0.15 * tall);
  say(`default ${w}: ${await lay(d)}  initial bottoms ${first} (gap ${Math.abs(first[0] - first[1])}), loaded bottoms ${bt} gap ${gap} of ${tall} (${(100 * gap / tall).toFixed(1)}%, tolerance ${Math.round(tol)})`);
  if (gap > tol) fail(`default columns not level at ${w}: bottoms ${bt}`);
  if (await stored(d) !== null) fail('default layout was stored (balancing must not save): ' + await stored(d));
  // the map is the full-width hero row over two equal columns
  const g = await d.evaluate(() => { const R = e => { const r = e.getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top + scrollY, b: r.bottom + scrollY, w: r.width }; };
    const m = R(document.querySelector('.blk[data-blk=map]')), v = R(document.querySelector('#viewer')), c0 = R(document.querySelector('.col[data-col="0"]')), c1 = R(document.querySelector('.col[data-col="1"]')), cv = document.querySelector('#viewer canvas');
    const out = [...document.querySelectorAll('#viewer .overlay')].filter(o => o.offsetParent).map(o => { const r = o.getBoundingClientRect(); return r.left < v.l - 1 || r.right > v.r + 1 || r.top + scrollY < v.t - 1 || r.bottom + scrollY > v.b + 1 ? o.className : null; }).filter(Boolean);
    const ov = [...document.querySelectorAll('#viewer .overlay')].filter(o => o.offsetParent).map(o => o.getBoundingClientRect()); let hit = 0; for (let i = 0; i < ov.length; i++) for (let j = i + 1; j < ov.length; j++) { const a = ov[i], b = ov[j]; if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) hit++; }
    return { m, v, c0, c1, inner: [innerWidth, innerHeight], out, hit, asp: [cv.width / cv.height, cv.clientWidth / cv.clientHeight], cvd: [cv.clientWidth - document.querySelector('#viewer').clientWidth, cv.clientHeight - document.querySelector('#viewer').clientHeight] }; });
  say(`map row ${w}: x ${Math.round(g.m.l)}..${Math.round(g.m.r)} (cols ${Math.round(g.c0.l)}..${Math.round(g.c1.r)}), height ${Math.round(g.v.b - g.v.t)}, bottom ${Math.round(g.v.b)} of ${g.inner[1]}, columns ${Math.round(g.c0.w)} / ${Math.round(g.c1.w)}, canvas aspect ${g.asp.map(x => x.toFixed(3))}`);
  if (Math.abs(g.m.l - g.c0.l) > 2 || Math.abs(g.m.r - g.c1.r) > 2) fail(`map row is not the full width of the two columns at ${w}`);
  if (Math.abs(g.c0.w - g.c1.w) > 2) fail(`columns not equal at ${w}: ${g.c0.w} / ${g.c1.w}`);
  if (g.m.r - g.m.l < w - 60) fail(`map row not full page width at ${w}: ${g.m.r - g.m.l}`);
  if (g.v.t >= g.c0.t || g.v.b > g.c0.t + 1) fail('map row is not above the columns at ' + w);
  if (g.v.b > g.inner[1] - 8) fail(`map block not fully visible without scrolling at ${w} x ${g.inner[1]}: bottom ${g.v.b}`);
  if (g.out.length) fail('map controls outside the map at ' + w + ': ' + g.out); if (g.hit) fail('map controls overlap at ' + w + ': ' + g.hit);
  { const hd = await d.evaluate(() => { const t = document.querySelector('header.top').getBoundingClientRect(), m = document.querySelector('#model-box').getBoundingClientRect(), s = document.querySelector('#status').getBoundingClientRect(), b = document.querySelector('.brand').getBoundingClientRect();
      return { h: Math.round(t.height), inside: m.left >= t.left && m.right <= t.right + 1 && m.top >= t.top && m.bottom <= t.bottom + 1, row: (s.top + s.bottom) / 2 > b.top && (s.top + s.bottom) / 2 < b.bottom, bh: Math.round(b.height), sel: document.querySelector('#model-pick').value }; });
    say(`header ${w}: ${JSON.stringify(hd)}`);
    if (!hd.inside || hd.bh > 84 || (hd.row && hd.h > 110) || hd.sel !== 'a') fail(`model picker / header at ${w}: ${JSON.stringify(hd)}`); }
  if (Math.abs(g.asp[0] - g.asp[1]) > 0.02 || g.cvd.some(x => Math.abs(x) > 2)) fail('canvas stretched / not sized to the map at ' + w + ': ' + JSON.stringify([g.asp, g.cvd]));
  if (w !== 1280) await d.screenshot({ path: `${out}/wide_${w}x${w === 1920 ? 1080 : 900}.png` });
  await d.screenshot({ path: `${out}/layout_default${w === 1440 ? '' : '_' + w}.png`, fullPage: true });
  if ((await lay(d)).split('|').some(x => !x)) fail('an empty column at the default');
  await d.close();
}
const p = await open(1400, PIN);
o = await lay(p); say('pinned ' + o);
if (o !== DEF) fail('pinned layout (a saved arrangement) was rebalanced / not honored: ' + o);
const hs = await p.$$eval('.blk', bs => bs.map(b => { const h = b.querySelector('.blk-handle'), f = h; return [h.draggable, f.tabIndex, f.getAttribute('aria-label'), !!b.querySelector('.blk-grip, .blk-left, .blk-right')]; }));
say('handles: ' + JSON.stringify(hs.map(x => x.slice(0, 3)))); if (hs.length !== 7 || hs.some((x, k) => (k === 0 ? x[0] : !x[0]) || x[1] !== 0 || !/^Move .+: use arrow keys$/.test(x[2]) || x[3])) fail('header handles wrong (draggable / tabindex / aria-label / no grip or column buttons): ' + JSON.stringify(hs));
const vis = await p.$$eval('.blk-head button', bs => bs.filter(b => getComputedStyle(b).display !== 'none').map(b => b.className));
say('visible header buttons at 1400: ' + vis.length); if (vis.length !== 7 || vis.some(c => !c.includes('blk-min'))) fail('desktop shows more than the "-" buttons: ' + vis);
if (await p.$$eval('.blk-min', els => els.some(e => e.getAttribute('aria-expanded') !== 'true' || e.textContent !== '−' || !e.getAttribute('aria-controls')))) fail('minimize buttons not in the expanded state / lack aria-controls');
const hdr = await p.$$eval('.blk', bs => bs.map(b => { const h = b.querySelector('h2.blk-handle'), pn = h.closest('.card, .why, .viewer'), m = b.querySelector('.blk-min'), hr = h.getBoundingClientRect(), pr = pn.getBoundingClientRect(), mr = m.getBoundingClientRect(), cs = getComputedStyle(h);
  return { id: b.dataset.blk, fs: cs.fontSize, fw: cs.fontWeight, ml: Math.round(hr.left - pr.left), mt: Math.round(hr.top - pr.top), cur: cs.cursor, pad: cs.padding, mr: Math.round(pr.right - mr.right), mtop: Math.round(mr.top - pr.top), inside: pn.contains(m) }; }));
say('headers: ' + JSON.stringify(hdr.map(x => [x.id, x.fs, x.fw, x.ml, x.mt, x.mr, x.mtop])));
for (const k of ['fs', 'fw', 'ml', 'mt', 'pad', 'mr', 'mtop']) if (new Set(hdr.map(x => x[k])).size !== 1) fail(`unit headers differ in ${k}: ` + hdr.map(x => x.id + '=' + x[k]));
if (hdr.some(x => !x.inside || x.fs !== '15px')) fail('a unit header is not inside its panel at 15px');
const cur = await p.$$eval('.blk-handle', els => els.map(e => getComputedStyle(e).cursor)); if (cur.some((c, k) => c !== (k === 0 ? 'default' : 'grab')) || cur.length !== 7) fail('header cursor is not grab: ' + cur);
if (await p.evaluate(() => !!document.querySelector('#viewer').closest('[draggable=true]') || !!document.querySelector('#viewer canvas').closest('[draggable=true]'))) fail('the map graphic sits inside a draggable element (camera drag would break)');
if (await p.evaluate(() => { const e = new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: new DataTransfer() }); document.querySelector('.blk[data-blk=near] #neighbors').dispatchEvent(e); return !!document.querySelector('.blk.dragging'); })) fail('a drag started from the body of a unit');
await p.hover(hsel('near')); await wait(150);
if (await p.$eval(hsel('near'), e => getComputedStyle(e).backgroundColor) === 'rgba(0, 0, 0, 0)') fail('no hover highlight on the header');
const dm = await p.evaluate(() => { const a = document.querySelector('.col[data-col="0"]').getBoundingClientRect(), b = document.querySelector('.col[data-col="1"]').getBoundingClientRect(); return [a.width, b.width]; });
say('column widths ' + dm); if (Math.abs(dm[0] - dm[1]) > 2) fail('columns are not equal: ' + dm);
await p.screenshot({ path: `${out}/layout_pinned.png`, fullPage: true });

// a card (near) dropped on the lower half of the explanation: it sits below the explanation, in the other column
const marks = await dragBlk(p, 'near', 'why', 'b'); say('drop markers during drag: ' + marks); if (marks !== 1) fail('expected one drop marker, saw ' + marks);
o = await lay(p); say('near below explanation ' + o); if (o !== 'order|why,near,trustx,trustm,evid') fail('card not dropped below the explanation: ' + o);
const rw = await rect(p, 'why'), rn = await rect(p, 'near');
if (!(rn.t >= rw.b && rn.l === rw.l)) fail('card is not stacked under the explanation: ' + JSON.stringify({ rw, rn }));
if (await stored(p) === null || !(await stored(p)).includes('"near"')) fail('layout not stored: ' + await stored(p));
if (await p.$eval('#btn-reset-layout', e => e.hidden)) fail('Reset layout not shown after a change');
// a card to the top of the left column, and one to the end of the right column
await dragBlk(p, 'evid', 'order', 't'); o = await lay(p); if (o !== 'evid,order|why,near,trustx,trustm') fail('card to the top of the left column: ' + o);
await dragCol(p, 'order', 1); o = await lay(p); say('order to the end of the right column ' + o); if (o !== 'evid|why,near,trustx,trustm,order') fail('drag to the end of a column: ' + o);
// the map row is fixed: dropping it anywhere changes nothing, and it is not draggable
const mr0 = await rect(p, 'map'); await dragBlk(p, 'map', 'trustx', 't'); await dragCol(p, 'map', 0); o = await lay(p); if (o !== 'evid|why,near,trustx,trustm,order') fail('the map moved into a column: ' + o);
if ((await rect(p, 'map')).t !== mr0.t || (await rect(p, 'map')).w !== mr0.w) fail('map row moved');
await dragBlk(p, 'trustx', 'evid', 'b'); await dragBlk(p, 'order', 'near', 'b'); o = await lay(p); say('custom ' + o); if (o !== 'evid,trustx|why,near,order,trustm') fail('custom arrangement: ' + o);
await p.screenshot({ path: `${out}/layout_custom.png`, fullPage: true }); await p.screenshot({ path: `${out}/wide_custom.png`, fullPage: true });

// fold a few cards: only the header bar stays
await p.click(min('trustx')); await p.click(min('trustm')); await p.click(min('order'));
const fx = await rect(p, 'trustx'); say('folded card height ' + fx.h);
if (await expanded(p, 'trustx') !== 'false' || (await p.$eval(min('trustx'), e => e.textContent)) !== '+') fail('fold did not flip aria-expanded / label');
if (fx.h > 90) fail('folded card is not just its header bar: ' + fx.h);
if (!(await p.$eval(min('trustx'), e => e.getAttribute('aria-label').startsWith('Expand')))) fail('fold label not "Expand ..."');
if (await p.$eval('.blk[data-blk=trustx] .card h2', e => getComputedStyle(e).display === 'none')) fail('folded card lost its title');
if (await p.$eval('.blk[data-blk=trustx] #del-chart', e => getComputedStyle(e).display !== 'none')) fail('folded card body still visible');
await p.screenshot({ path: `${out}/layout_collapsed.png`, fullPage: true });
// persists over a reload (columns, order, folded)
const before = await stored(p); say('stored ' + before);
await reload(p); o = await lay(p); say('after reload ' + o);
if (o !== 'evid,trustx|why,near,order,trustm') fail('layout not kept over reload: ' + o);
if (await expanded(p, 'trustx') !== 'false' || await expanded(p, 'order') !== 'false' || await expanded(p, 'near') !== 'true') fail('folded set not kept over reload');
await p.click(min('trustx')); if (await expanded(p, 'trustx') !== 'true' || (await rect(p, 'trustx')).h < 200) fail('expand did not restore the card');
if (JSON.parse(await stored(p)).folded.includes('trustx')) fail('expanded card still in the stored folded set');
// the map: fold, restore, canvas re-measured; folded map must not break the pop-out view
const sizeOk = () => p.evaluate(() => { const v = document.querySelector('#viewer'), c = v.querySelector('canvas'); return [Math.round(v.getBoundingClientRect().height), c.clientWidth - v.clientWidth, c.clientHeight - v.clientHeight, c.width > 100]; });
const s0 = await sizeOk();
await p.click(min('map')); await wait(300);
if (await p.$eval('#viewer canvas', e => getComputedStyle(e).display) !== 'none') fail('folded map still shows its graphic');
if ((await rect(p, 'map')).h > 90) fail('folded map is not just its header bar');
await reload(p); if (await expanded(p, 'map') !== 'false') fail('folded map not kept over reload');
await p.click(min('map')); await wait(700);
const s1 = await sizeOk(); say('map size before fold / after restore ' + JSON.stringify([s0, s1]));
if (s1[0] < 400 || Math.abs(s1[1]) > 2 || Math.abs(s1[2]) > 2 || !s1[3] || Math.abs(s1[0] - s0[0]) > 3) fail('map canvas not re-measured after restore: ' + JSON.stringify([s0, s1]));
await p.click(min('map'));                                                 // fold again, then open the pop-out window's address: it must still show the whole graphic
const g = await open(1100, null, '?view=graphic');
const gv = await g.evaluate(() => { const v = document.querySelector('#viewer'); return [getComputedStyle(v).display, Math.round(v.getBoundingClientRect().height)]; });
say('?view=graphic with a folded map ' + JSON.stringify(gv)); if (gv[0] === 'none' || gv[1] < 400) fail('folded map broke the pop-out view');
await g.close();
// reset: default columns, everything expanded, storage cleared
await p.click('#btn-reset-layout'); o = await lay(p);
say('after reset ' + o + ' stored ' + await stored(p)); if (await stored(p) !== null || o.split('|').some(x => !x)) fail('reset failed: ' + o + ' ' + await stored(p));
if (await p.$eval('#btn-reset-layout', e => !e.hidden)) fail('Reset layout still shown after reset');
if (await p.$$eval('.blk.collapsed', e => e.length)) fail('reset left a unit folded');

// keyboard: arrows on the focused header; focus stays on the moved unit (the reset gave the balanced default again: pin the known arrangement first)
await p.evaluate(() => localStorage.setItem('lis-layout-v2-d', JSON.stringify({ v: 2, cols: [['near', 'order'], ['why', 'trustx', 'trustm', 'evid']], folded: [] }))); await reload(p);
await key(p, grip('near'), 'ArrowRight'); o = await lay(p);
let foc = await p.evaluate(() => document.activeElement.tagName + '.' + (document.activeElement.closest('.blk')?.dataset.blk));
say('ArrowRight on near header -> ' + o + ' focus ' + foc); if (o !== 'order|near,why,trustx,trustm,evid') fail('ArrowRight: ' + o); if (foc !== 'H2.near') fail('focus lost after ArrowRight: ' + foc);
if (!(await p.$eval('p[role=status]', e => /moved to column 2/.test(e.textContent)).catch(() => false))) fail('move not announced in the live region');
await key(p, grip('near'), 'ArrowDown'); o = await lay(p); if (o !== 'order|why,near,trustx,trustm,evid') fail('ArrowDown: ' + o);
await key(p, grip('near'), 'ArrowUp'); await key(p, grip('near'), 'ArrowLeft'); o = await lay(p); if (o !== 'near,order|why,trustx,trustm,evid') fail('ArrowUp + ArrowLeft: ' + o);
await key(p, grip('why'), 'ArrowDown'); o = await lay(p); say('why header ArrowDown -> ' + o); if (o !== 'near,order|trustx,why,trustm,evid') fail('ArrowDown on the explanation header: ' + o);
foc = await p.evaluate(() => document.activeElement.className); if (!foc.includes('blk-handle')) fail('focus lost on the explanation header: ' + foc);
for (const k of ['ArrowRight', 'ArrowDown', 'ArrowLeft']) { await key(p, grip('map'), k); o = await lay(p); if (o !== 'near,order|trustx,why,trustm,evid') fail('map ' + k + ' moved something: ' + o); }
await key(p, min('evid'), 'Enter'); if (await expanded(p, 'evid') !== 'false') fail('Enter on the minimize button did not fold');
await p.screenshot({ path: `${out}/layout_keyboard.png`, fullPage: true });
await p.click('#btn-reset-layout');
await p.close();

// stale / bad values: old v1 key dropped, corrupt v2 ignored
const bad = await open(1400, () => { localStorage.setItem('lis-layout-v1', '["cards","why","map"]'); localStorage.setItem('lis-layout-v2-d', '{"v":2,"cols":"x"'); });
o = await lay(bad); say('old v1 + corrupt v2 -> ' + o + ', v1 left: ' + await stored(bad, 'lis-layout-v1')); if (o.split('|').some(x => !x) || await stored(bad) !== '{"v":2,"cols":"x"') fail('bad stored values not ignored: ' + o); if (await stored(bad, 'lis-layout-v1') !== null) fail('old v1 key not removed');
await bad.close();
const part = await open(1400, () => { if (!sessionStorage.getItem('x')) { sessionStorage.setItem('x', '1'); localStorage.setItem('lis-layout-v2-d', JSON.stringify({ v: 2, cols: [['why', 'bogus'], ['near']], folded: ['near', 'nope'] })); } });
o = await lay(part); say('partial stored value completed -> ' + o); if (o !== 'why,trustx,trustm,evid,order|near') fail('partial value not completed: ' + o);
await part.evaluate(() => localStorage.clear()); await part.close();

// storage that throws: moves and folding still work, nothing breaks
const q = await open(1400, () => { Storage.prototype.setItem = () => { throw new Error('denied'); }; Storage.prototype.getItem = () => { throw new Error('denied'); }; Storage.prototype.removeItem = () => { throw new Error('denied'); }; });
await wait(1500); const b4 = await lay(q); await q.focus(fsel('near')); await q.keyboard.press(b4.split('|')[0].includes('near') ? 'ArrowRight' : 'ArrowLeft'); o = await lay(q); say('storage denied, arrow key -> ' + o); if (o === b4) fail('no move with storage denied: ' + o);
await q.click(min('trustm')); if (await expanded(q, 'trustm') !== 'false') fail('no fold with storage denied');
await q.close();

// tablet width: one column, no drag, saved arrangement still applies in the flattened order
const tb = await open(900, PIN);
const tops = await tb.evaluate(() => [...document.querySelectorAll('#blocks .blk')].map(b => [b.dataset.blk, Math.round(b.getBoundingClientRect().top + scrollY), Math.round(b.getBoundingClientRect().left)]).sort((a, b) => a[1] - b[1]));
say('900 px: ' + tops.map(x => x[0]).join(',')); if (tops.map(x => x[0]).join() !== 'map,near,why,order,trustx,trustm,evid') fail('900 px flattened order: ' + tops.map(x => x[0])); if (new Set(tops.map(x => x[2])).size !== 1) fail('900 px is not one column');
await tb.click('.blk[data-blk=why] .blk-up'); await wait(100);
const t2 = await tb.evaluate(() => [...document.querySelectorAll('#blocks .blk')].map(b => [b.dataset.blk, b.getBoundingClientRect().top]).sort((a, b) => a[1] - b[1]).map(x => x[0]).join()); say('900 px after why up: ' + t2); if (t2 !== 'map,why,near,order,trustx,trustm,evid') fail('900 px up button: ' + t2);
if (await tb.$eval(hsel('near'), e => getComputedStyle(e).cursor === 'grab')) fail('grab cursor in one column');
const visT = await tb.$$eval('.blk-head button', bs => bs.filter(b => getComputedStyle(b).display !== 'none').length); say('900 px visible header buttons ' + visT); if (visT !== 19) fail('900 px header buttons (6 units x up, down, - and the map\'s -): ' + visT);
await tb.evaluate(() => localStorage.clear()); await tb.close();

// phone: no drag, own storage key, buttons and folding work
const ph = await open(390, PIN);
const gripShown = await ph.$eval(hsel('why'), e => getComputedStyle(e).cursor === 'grab');
const prevented = await ph.evaluate(() => { const e = new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: new DataTransfer() }); document.querySelector('.blk-handle').dispatchEvent(e); return e.defaultPrevented; });
say(`phone: grab cursor ${gripShown}, dragstart prevented ${prevented}`); if (gripShown || !prevented) fail('phone still offers drag');
await ph.click('.blk[data-blk=why] .blk-up'); await ph.click(min('near'));
const pk = [await stored(ph, 'lis-layout-v2-p'), await stored(ph, 'lis-layout-v2-d')]; say('phone storage p/d: ' + pk.map(x => x && x.slice(0, 60))); if (!pk[0] || pk[1] !== null) fail('phone did not use its own key: ' + pk);
const pt = await ph.evaluate(() => [...document.querySelectorAll('#blocks .blk')].map(b => [b.dataset.blk, b.getBoundingClientRect().top]).sort((a, b) => a[1] - b[1]).map(x => x[0]).join()); if (pt !== 'map,why,near,order,trustx,trustm,evid') fail('phone order after up: ' + pt);
await ph.screenshot({ path: `${out}/layout_phone.png` }); await ph.screenshot({ path: `${out}/phone.png` });
await ph.evaluate(() => localStorage.clear());
await ph.close();

// one type scale: a unit looks the same (heading, body text, chart text) in either column, at three widths, and after being moved to the other column
const typeSig = p => p.evaluate(() => {
  const r = {}, fs = e => e ? Math.round(parseFloat(getComputedStyle(e).fontSize) * 10) / 10 : null;
  for (const b of document.querySelectorAll('.blk')) {
    const id = b.dataset.blk, h = b.querySelector('.card > h2, .why h2'), body = b.querySelector('.note, .hero-text, .summary, table.metrics td, ul.plain li, .empty-small, .empty');
    const ch = [...b.querySelectorAll('svg.chart')].filter(s => s.querySelector('text')).map(s => { const k = s.getBoundingClientRect().width / s.viewBox.baseVal.width; return [...new Set([...s.querySelectorAll('text')].map(t => Math.round(fs(t) * k * 2) / 2))].sort(); });
    r[id] = { col: b.closest('.col')?.dataset.col, h: fs(h), body: fs(body), chart: [...new Set(ch.flat())].join('/') };
  }
  return r;
});
for (const w of [1280, 1440, 1920]) {
  const d = await open(w, PIN);
  let prev = '', same = 0; for (let k = 0; k < 40 && same < 3; k++) { await wait(700); const t = await typeSig(d), b = JSON.stringify(t), ready = ['order', 'trustx', 'trustm', 'evid'].every(i => t[i].chart); same = ready && b === prev ? same + 1 : 0; prev = b; }   // loaded = every chart unit has drawn, and nothing changed for 3 polls
  const a = await typeSig(d), ids = Object.keys(a);
  const vals = f => [...new Set(ids.map(i => a[i][f]).filter(v => v != null && v !== ''))];
  say(`type ${w}: ` + ids.map(i => `${i}[c${a[i].col}] h${a[i].h} b${a[i].body} chart ${a[i].chart || '-'}`).join('; '));
  if (vals('h').length !== 1) fail(`unit headings differ in size at ${w}: ${vals('h')}`);
  const bodies = ids.filter(i => a[i].body != null && a[i].col != null);   // body text: notes are 12.5, the table 13: compare per kind, not across kinds
  for (const c of ['0', '1']) for (const i of ids) if (a[i].col === c && a[i].chart && !/^(11|11\.5)(\/11(\.5)?)?$/.test(a[i].chart)) fail(`chart text of ${i} (column ${c}) is ${a[i].chart}px at ${w}, not 11 / 11.5`);
  const noteKinds = new Set(ids.filter(i => ['near', 'order', 'trustx', 'trustm', 'evid'].includes(i)).map(i => a[i].body)); if (noteKinds.size > 2) fail(`card body text sizes vary at ${w}: ${[...noteKinds]}`);
  // move a unit from each column to the other: its sizes must not change
  await dragBlk(d, 'order', 'trustm', 'b'); await dragBlk(d, 'trustx', 'near', 't'); await wait(1200);
  const m = await typeSig(d);
  for (const i of ['order', 'trustx']) { if (m[i].col === a[i].col) fail(`${i} did not change column at ${w}`); if (m[i].h !== a[i].h || m[i].body !== a[i].body || m[i].chart !== a[i].chart) fail(`${i} changed type size after moving column at ${w}: ${JSON.stringify(a[i])} -> ${JSON.stringify(m[i])}`); }
  if (w === 1280) await d.screenshot({ path: `${out}/fix_text_moved.png`, fullPage: true });
  await d.evaluate(() => localStorage.clear()); await d.close();
}

// ---- B pass: under the masked-word model the unrelated / translation bands overlap, so the two strip labels must not collide or leave the chart
for (const w of [1400, 700]) {
  const b = await browser.newPage(); await b.setViewport({ width: w, height: 900 });
  b.on('pageerror', e => problems.push('[pageerror B] ' + String(e).slice(0, 200)));
  await b.goto(url + '?model=b', { waitUntil: 'domcontentloaded' });
  let ok = false; for (let k = 0; k < 400 && !ok; k++) { await wait(500); try { ok = await b.evaluate(() => document.documentElement.dataset.model === 'b' && !!window.__mm?.state.result && !!document.querySelector('#sim-strip svg text')); } catch { /* reloading */ } }
  if (!ok) { fail('B did not load for the strip check at ' + w); await b.close(); continue; }
  await wait(1200);
  const r = await b.evaluate(() => { const svg = document.querySelector('#sim-strip svg'), vb = svg.viewBox.baseVal, ts = [...svg.querySelectorAll('text.band-label')].map(t => { const x = t.getBBox(); return { t: t.textContent, l: x.x, r: x.x + x.width, tp: x.y, bt: x.y + x.height }; });
    const [a, c] = ts, hit = a.l < c.r && c.l < a.r && a.tp < c.bt && c.tp < a.bt; return { vbW: vb.width, vbH: vb.height, ts, hit, inside: ts.every(t => t.l >= 0 && t.r <= vb.width && t.bt <= vb.height + 1) }; });
  if (r.hit || !r.inside || r.ts.length !== 2) fail(`B similarity strip labels at ${w}: ${JSON.stringify(r)}`); else say(`B strip ${w}: labels apart and inside the chart (${r.ts.map(t => `${t.t} ${Math.round(t.l)}..${Math.round(t.r)}`).join(', ')})`);
  if (w === 1400) { const el = await b.$('#sim-strip'); await el.screenshot({ path: `${out}/strip_B_after.png` }); }
  await b.close();
}

fs.writeFileSync(`${out}/layout_results.json`, JSON.stringify({ problems }, null, 1));
console.log(JSON.stringify({ problems }, null, 1));
await browser.close();
process.exit(problems.length ? 1 : 0);
