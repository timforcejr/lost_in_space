// Title: jump.mjs, regression check for "the page jumps when I switch the benchmark language, or the map's Show-language buttons".
// Checks: after each toggle (and after the 350 ms column-balance debounce plus 1.5 s settle) window.scrollY is unchanged and every block ABOVE the toggled one (and the toggled
// block's own top) keeps its page position, tolerance 2 px. Run at 1440x900 and 1920x1080, light theme, default (auto-balanced) layout and a saved (dragged) layout, with the
// page scrolled so the toggled control sits in view. Prints the measured numbers.
// Run:  node tests/jump.mjs http://127.0.0.1:8000/      (exit 0 = pass)
import puppeteer from 'puppeteer-core';

const url = process.argv[2] || 'http://127.0.0.1:8000/';
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const problems = []; const wait = ms => new Promise(r => setTimeout(r, ms));
const TOL = 2;
const SAVED = JSON.stringify({ v: 2, cols: [['near', 'trustx', 'evid'], ['why', 'order', 'trustm']], folded: [] });
const TOGGLES = [
  ['bench Hindi', '#seg-bench button[data-v=hi]', 'trustx'], ['bench Chinese', '#seg-bench button[data-v=zh]', 'trustx'], ['bench All', '#seg-bench button[data-v=all]', 'trustx'],
  ['Show Hindi off', '#lang-toggles button:nth-of-type(3)', 'map'], ['Show Hindi on', '#lang-toggles button:nth-of-type(3)', 'map'],
];
const snap = p => p.evaluate(() => ({ y: Math.round(scrollY), dh: document.documentElement.scrollHeight, bl: Object.fromEntries([...document.querySelectorAll('.blk')].map(b => { const r = b.getBoundingClientRect(); return [b.dataset.blk, [Math.round(r.top + scrollY), Math.round(r.height), b.parentElement.dataset.col || '']]; })), ch: Math.round(document.querySelector('#bench-chart').getBoundingClientRect().height) }));
for (const [w, h] of [[1440, 900], [1920, 1080]]) for (const saved of [false, true]) {
  const p = await browser.newPage(); await p.setViewport({ width: w, height: h });
  p.on('pageerror', e => problems.push('[pageerror] ' + String(e).slice(0, 200)));
  await p.evaluateOnNewDocument((saved, SAVED) => { try { localStorage.clear(); localStorage.setItem('theme', 'light'); if (saved) localStorage.setItem('lis-layout-v2-d', SAVED); } catch {} }, saved, SAVED);
  await p.goto(url, { waitUntil: 'domcontentloaded' });
  await p.waitForSelector('#bench-chart svg', { timeout: 30000 }); await p.waitForSelector('#lang-toggles button', { timeout: 30000 }); await wait(7000);   // let the 1.5 s / 5 s re-balances finish
  for (const [name, sel, host, sel2] of TOGGLES) {
    await p.evaluate(sel => { const b = document.querySelector(sel); const r = b.getBoundingClientRect(); scrollTo(0, Math.max(0, r.top + scrollY - innerHeight / 2)); }, sel); await wait(400);
    const a = await snap(p);
    await p.evaluate(sel => document.querySelector(sel).click(), sel); if (sel2) await p.evaluate(s => document.querySelector(s).click(), sel2);
    const trace = [];
    for (const ms of [0, 100, 450, 1800]) { await wait(ms ? ms - (trace.length ? trace[trace.length - 1].ms : 0) : 0); const s = await snap(p); trace.push({ ms, s }); }
    for (const { ms, s } of trace) {
      const moved = []; const top = a.bl[host][0];
      for (const id of Object.keys(a.bl)) { if (a.bl[id][0] <= top && a.bl[id][2] === a.bl[host][2] || id === host) { if (Math.abs(s.bl[id][0] - a.bl[id][0]) > TOL) moved.push(`${id} ${a.bl[id][0]}->${s.bl[id][0]}`); } if (s.bl[id][2] !== a.bl[id][2]) moved.push(`${id} col ${a.bl[id][2]}->${s.bl[id][2]}`); }
      const dy = s.y - a.y;
      const tag = `${w}x${h} ${saved ? 'saved' : 'default'} "${name}" @${ms}ms`;
      if (ms === 1800 || Math.abs(dy) > TOL || moved.length) console.log(`${tag}: dScrollY=${dy} dDocH=${s.dh - a.dh} chartH ${a.ch}->${s.ch} ${moved.length ? 'MOVED ' + moved.join('; ') : ''}`);
      if (Math.abs(dy) > TOL) problems.push(`${tag}: scrollY moved ${dy}px`);
      if (moved.length) problems.push(`${tag}: blocks moved ${moved.join('; ')}`);
    }
  }
  await p.close();
}
await browser.close();
if (problems.length) { console.log('FAIL (' + problems.length + ')'); [...new Set(problems)].slice(0, 25).forEach(s => console.log(' - ' + s)); process.exit(1); }
console.log('jump: PASS');
