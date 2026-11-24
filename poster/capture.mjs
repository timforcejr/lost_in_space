// Take the poster's screenshots from the running app and dump the example pairs' numbers to poster/fig/hero.json
//       and poster/fig/order_pair.json.
// Run:  node poster/capture.mjs http://127.0.0.1:8137/
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const url = process.argv[2] || 'http://127.0.0.1:8137/';
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fig');
fs.mkdirSync(out, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1400,1200'] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1200, deviceScaleFactor: 2 });
await page.goto(url, { waitUntil: 'domcontentloaded' });
const done = a => page.waitForFunction(a => window.__mm && window.__mm.state.result && window.__mm.state.result.a === a && window.__mm.orderDone, { timeout: 300000 }, a);
const submit = async (a, b) => { await page.evaluate((a, b) => { document.querySelector('#in-a').value = a; document.querySelector('#in-b').value = b; document.querySelector('#form').requestSubmit(); }, a, b); await done(a); };
const HERO = ['The cat sleeps on the sofa.', 'El gato duerme en el sofá.'], ORDER = ['The dog bit the man.', 'The man bit the dog.'];
await done(HERO[0]);                                                                               // default pair: EN / ES

// numbers behind the example pairs (redrawn natively on the poster)
const dump = () => page.evaluate(() => {
  const R = window.__mm.state.result, r = a => a.map(v => +v.toFixed(4));
  return { a: R.a, b: R.b, sim: R.an.sim, total: R.an.wm.total, ms: R.ms,
    A: R.gA.map(g => ({ text: g.text, punct: g.punct })), B: R.gB.map(g => ({ text: g.text, punct: g.punct })),
    scores: Object.fromEntries(Object.entries(R.an.scores).map(([k, s]) => [k, { A: r(s.A), B: r(s.B) }])),
    W: R.an.wm.W.map(r), layerSim: r(R.an.layerSim), del: R.del,
    grad: R.grad && { ms: R.grad.ms, sim: R.grad.A.sim, igSum: R.grad.A.igSum, igTarget: R.grad.A.igTarget, steps: R.grad.steps },
    order: R.order && { scrambled: R.order.variants.map(v => v.text), cosShuffle: r(R.order.cosShuffle), simOtherShuffle: R.order.simOtherShuffle, probeNatural: r(R.order.probeNatural), probeShuffle: r(R.order.probeShuffle) } };
});
await submit(...ORDER);
fs.writeFileSync(`${out}/order_pair.json`, JSON.stringify(await dump(), null, 1));
await submit(...HERO);
const hero = await dump();
fs.writeFileSync(`${out}/hero.json`, JSON.stringify(hero, null, 1));

// one wide viewer, UI text enlarged so it survives being printed
await page.addStyleTag({ content: `
  .stage { grid-template-columns: 1fr !important; } .why { display: none; }
  #viewer { height: 780px; min-height: 0; }
  .overlay { zoom: 1.45; } .honesty, .layer-note { display: none; }
  .label { font-size: 20px !important; } .label.sent { font-size: 22px !important; }` });
const v = await page.$('#viewer');
await page.click('#btn-words');                                   // word points off: a clean picture of the two sentences and what lights up
await page.click('#seg-head button[data-v="tsne"]');
await sleep(3500); await v.screenshot({ path: `${out}/viewer_pair_tsne.png` });

// four maps of the same cloud, coloured by topic, no chrome
await page.addStyleTag({ content: `#viewer { width: 880px; height: 880px; } .overlay, .labels { display: none !important; }` });
const click = sel => page.evaluate(sel => document.querySelector(sel).click(), sel);   // overlays are hidden, so click through the DOM
await click('#seg-color button[data-v="topic"]');
await click('#btn-reset');
for (const h of ['pca', 'fc', 'ae', 'tsne']) {
  await click(`#seg-head button[data-v="${h}"]`);
  await sleep(2600); await v.screenshot({ path: `${out}/map_${h}.png` });
}
await browser.close();
console.log('wrote', fs.readdirSync(out).join(', '));
