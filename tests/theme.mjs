// Title: theme and map-only test.
// What it proves: the page follows an emulated prefers-color-scheme (light and dark) when nothing is chosen; the System / Light / Dark
// switch sets data-theme, is keyboard operable, and the choice survives a reload; a throwing localStorage does not break the page;
// a theme change redraws the map, legend and heat cells with the new colors without a new analysis run;
// ?maponly=1 hides the filter rows, header and panels, keeps the topic legend and a viewport-filling map, and legend=0 hides the legend.
// Run:  node tests/theme.mjs http://127.0.0.1:8000/      (exit 0 = pass)
import puppeteer from 'puppeteer-core';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100'] });
const problems = [];
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) problems.push(m); };
const RGB = { light: 'rgb(247, 246, 242)', dark: 'rgb(13, 13, 13)' };

async function open(path, { scheme, noStorage } = {}) {
  await new Promise(r => setTimeout(r, 800));   // let a just-closed window's theme broadcast drain before a new window opens
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  page.on('pageerror', e => problems.push('[pageerror] ' + String(e).slice(0, 300)));
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) problems.push('[console] ' + m.text().slice(0, 300)); });
  if (scheme) await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
  if (noStorage) await page.evaluateOnNewDocument(() => { const t = () => { throw new DOMException('denied', 'SecurityError'); }; Storage.prototype.getItem = t; Storage.prototype.setItem = t; Storage.prototype.removeItem = t; });
  await page.goto(new URL(path, url).href, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.theme-switch button');
  return page;
}
const bg = p => p.evaluate(() => getComputedStyle(document.body).backgroundColor);
const attr = p => p.evaluate(() => document.documentElement.getAttribute('data-theme'));
const vis = (p, sel) => p.evaluate(s => { const e = document.querySelector(s); return !!e && getComputedStyle(e).display !== 'none' && e.getClientRects().length > 0; }, sel);

for (const scheme of ['light', 'dark']) {
  const p = await open('', { scheme });
  ok(await attr(p) === null, `${scheme}: no data-theme by default`);
  ok(await bg(p) === RGB[scheme], `${scheme}: page background follows the system (${await bg(p)})`);
  ok(await p.evaluate(() => document.querySelector('.theme-switch button[data-theme=system]').getAttribute('aria-pressed')) === 'true', `${scheme}: System is the pressed button`);
  await p.close();
}

{ // switch: set, persist, reload, back to system; keyboard
  const p = await open('', { scheme: 'dark' });
  await p.$eval('.theme-switch button[data-theme=light]', e => e.click());
  ok(await attr(p) === 'light' && await bg(p) === RGB.light, 'Light button sets data-theme=light and turns the page light under a dark system');
  ok(await p.evaluate(() => localStorage.getItem('lis.theme')) === 'light', 'choice stored in localStorage');
  await p.reload({ waitUntil: 'domcontentloaded' }); await p.waitForSelector('.theme-switch button');
  ok(await attr(p) === 'light' && await bg(p) === RGB.light, 'choice survives a reload');
  ok(await p.evaluate(() => document.querySelector('.theme-switch button[data-theme=light]').getAttribute('aria-pressed')) === 'true', 'Light shows as pressed after reload');
  await p.focus('.theme-switch button[data-theme=dark]'); await p.keyboard.press('Enter');
  ok(await attr(p) === 'dark', 'Dark button works from the keyboard (Enter)');
  await p.focus('.theme-switch button[data-theme=system]'); await p.keyboard.press('Space');
  ok(await attr(p) === null && await p.evaluate(() => localStorage.getItem('lis.theme')) === null && await bg(p) === RGB.dark, 'System removes data-theme and the stored choice');
  const m = await p.evaluate(() => [...document.querySelectorAll('.theme-switch button')].every(b => b.getAttribute('aria-label')) && document.querySelector('.theme-switch').getAttribute('role') === 'group');
  ok(m, 'switch buttons have aria-labels inside a labeled group');
  await p.close();
}

{ // storage that throws
  const p = await open('', { scheme: 'light', noStorage: true });
  ok(await attr(p) === null && await bg(p) === RGB.light, 'storage throws: page still loads and follows the system');
  await p.$eval('.theme-switch button[data-theme=dark]', e => e.click());
  { const at = await attr(p), c = await bg(p); ok(at === 'dark' && c === RGB.dark, `storage throws: the switch still works for this page view (${at}, ${c})`); }
  await p.close();
}

{ // switching theme redraws with the new colors and does not start a new analysis run
  const p = await open('', { scheme: 'dark' });
  await p.waitForFunction(() => window.__mm && window.__mm.state.result && window.__mm.orderDone, { timeout: 300000 });
  await p.evaluate(() => { window.__r = window.__mm.state.result; window.__n = 0; document.querySelector('#form').addEventListener('submit', () => window.__n++); });
  const snap = () => p.evaluate(() => ({ mark: getComputedStyle(document.querySelector('#legend3d .mark')).backgroundColor, tagA: getComputedStyle(document.querySelector('.field.a .tag')).backgroundColor,
    cell: document.querySelector('#chips-a .chip')?.style.background || '', first: [...window.__mm.state.baseColors.slice(0, 3)].map(v => +v.toFixed(3)).join(','), same: window.__mm.state.result === window.__r, runs: window.__n }));
  const d = await snap();
  await p.$eval('.theme-switch button[data-theme=light]', e => e.click()); await new Promise(r => setTimeout(r, 600));
  const l = await snap();
  ok(l.same && l.runs === 0, 'theme change starts no new analysis run (same result object, no submit)');
  ok(l.mark !== d.mark, `legend color changes with the theme (${d.mark} -> ${l.mark})`);
  ok(l.first !== d.first, `map point colors are redrawn (${d.first} -> ${l.first})`);
  ok(l.cell !== d.cell, 'heat cells are redrawn');
  await p.$eval('.theme-switch button[data-theme=dark]', e => e.click()); await new Promise(r => setTimeout(r, 600));
  const d2 = await snap();
  ok(d2.mark === d.mark && d2.first === d.first && d2.runs === 0, 'back to dark restores the dark colors, still no run');
  await p.close();
}

{ // map only
  const p = await open('?maponly=1');
  ok(await p.evaluate(() => document.documentElement.classList.contains('map-only')), 'maponly=1 sets class map-only');
  for (const s of ['.top', '.composer', '.why', '.cards', 'footer', '.overlay.tl', '#seg-color', '#seg-head', '.overlay.tr', '.overlay.br', '#layer', '.honesty']) ok(!(await vis(p, s)), `map only: ${s} hidden`);
  ok(await vis(p, '.overlay.bl'), 'map only: legend area shown by default');
  const r = await p.evaluate(() => { const b = document.querySelector('#viewer').getBoundingClientRect(); return [b.left, b.top, b.width, b.height, innerWidth, innerHeight]; });
  ok(r[0] === 0 && r[1] === 0 && r[2] === r[4] && r[3] === r[5], `map fills the window (${r.join(',')})`);
  await p.keyboard.press('Escape');
  ok(!(await p.evaluate(() => document.documentElement.classList.contains('map-only'))) && await vis(p, '#seg-color'), 'Escape returns to the full page, filters visible again');
  await p.$eval('#btn-maponly', e => e.click());
  ok(await p.evaluate(() => document.documentElement.classList.contains('map-only')) && !(await vis(p, '#seg-color')), 'the Map only button hides the filter rows');
  ok(new URL(p.url()).searchParams.get('maponly') === '1', 'address carries maponly=1');
  await p.close();
  const q = await open('?maponly=1&legend=0');
  ok(!(await vis(q, '.overlay.bl')), 'legend=0 hides the legend');
  await q.close();
  const v = await open('?view=map&maponly=1');
  ok(await v.evaluate(() => document.documentElement.classList.contains('pv-map') && document.documentElement.classList.contains('map-only')) && !(await vis(v, '.composer')), 'works together with the two-screen ?view=map window');
  await v.close();
}

await browser.close();
if (problems.length) { console.log('\nPROBLEMS:\n' + problems.join('\n')); process.exit(1); }
console.log('\nall theme checks passed');
