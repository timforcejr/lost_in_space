// Checks: (a) every p.note is visible or reachable via a "?" tip; (b) model download blocked -> "Could not start" and #go disabled; (c) topic is the default color; (d) layer arrows (step, clamp, disabled ends, label).
// Run: node tests/web_fixes.mjs http://127.0.0.1:8141/ tests/out_fixes
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const url = process.argv[2] || 'http://127.0.0.1:8137/', out = process.argv[3] || 'tests/out_fixes';
fs.mkdirSync(out, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100'] });
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
try {
  // (a)
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1100 });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__mm && window.__mm.orderDone, { timeout: 300000 });
  await page.evaluate(() => [...document.querySelectorAll('#seg-head button')].find(b => b.dataset.v === 'pca').click());
  await new Promise(r => setTimeout(r, 1500));
  const n = await page.evaluate(() => document.querySelectorAll('p.note').length);
  let seen = 0;
  for (let i = 0; i < n; i++) {
    const info = await page.evaluate(i => {
      const p = document.querySelectorAll('p.note')[i];
      const txt = p.textContent.trim();
      const vis = getComputedStyle(p).display !== 'none';
      let h = null;
      for (let e = p.previousElementSibling; e && !h; e = e.previousElementSibling) h = e.matches('h2, h3') ? e : e.matches('.matrix-head') ? e.querySelector('h2') : null;
      h = h || (p.closest('.card, section, article') || document).querySelector('h2, h3');
      const btns = [...document.querySelectorAll('.help-btn')].filter(b => !b.hidden);
      const idx = btns.findIndex(b => b.parentElement === h || h.contains(b));
      return { txt, vis, idx, id: p.id };
    }, i);
    if (!info.txt) continue;
    seen++;
    if (info.vis) { ok(true, `note ${i} (${info.id || 'anon'}) visible`); continue; }
    if (info.idx < 0) { ok(false, `note ${i} (${info.id || 'anon'}) hidden with no ? button`); continue; }
    const btns = await page.$$('.help-btn:not([hidden])');
    await btns[info.idx].click();
    const tip = await page.evaluate(() => document.querySelector('#help-pop').textContent);
    ok(tip.includes(info.txt.slice(0, 40)), `note ${i} (${info.id || 'anon'}) text in ? tip`);
    await page.keyboard.press('Escape');
  }
  ok(seen > 0, `${seen} non-empty notes checked`);
  await page.screenshot({ path: `${out}/notes.png`, fullPage: true });
  // (c) color default
  const cd = await page.evaluate(() => ({ cb: window.__mm.state.colorBy, on: [...document.querySelectorAll('#seg-color button.on')].map(b => b.dataset.v).join() }));
  ok(cd.cb === 'topic' && cd.on === 'topic', 'color default is Topic ' + JSON.stringify(cd));
  // (d) layer arrows
  const L = () => page.evaluate(() => ({ v: document.querySelector('#layer').value, lab: document.querySelector('#layer-label').textContent, pd: document.querySelector('#layer-prev').disabled, nd: document.querySelector('#layer-next').disabled }));
  let l = await L(); ok(l.v === '12' && l.nd && !l.pd, 'arrows at 12: next disabled ' + JSON.stringify(l));
  await page.click('#layer-prev'); l = await L(); ok(l.v === '11' && l.lab === 'Layer 11' && !l.nd, 'prev steps to 11, label synced');
  await page.evaluate(() => { const s = document.querySelector('#layer'); s.value = 1; s.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.click('#layer-prev'); l = await L(); ok(l.v === '0' && l.pd && l.lab.startsWith('Layer 0'), 'prev steps to 0, prev disabled');
  await page.evaluate(() => document.querySelector('#layer-prev').click()); l = await L(); ok(l.v === '0', 'clamped at 0');
  await page.click('#layer-next'); l = await L(); ok(l.v === '1' && !l.pd && l.lab === 'Layer 1', 'next steps to 1');
  await page.close();
  // (b)
  const ctx = await browser.createBrowserContext(); const p2 = await ctx.newPage();
  await p2.setRequestInterception(true);
  p2.on('request', r => /\/model\/mmini\/onnx\//.test(r.url()) ? r.abort() : r.continue());
  await p2.goto(url, { waitUntil: 'domcontentloaded' });
  try { await p2.waitForFunction(() => document.querySelector('#status-text').textContent.includes('Could not'), { timeout: 90000 }); } catch {}
  const txt = await p2.$eval('#status-text', e => e.textContent);
  ok(txt.includes('Could not'), `load failure status: "${txt}"`);
  ok(await p2.$eval('#go', e => e.disabled), '#go stays disabled');
} finally { await browser.close(); }
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
