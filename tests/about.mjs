// About.html loads without errors, has its sections, fills the reference list and the per-language sentence, follows the light/dark switch,
// has a working "Back" link; index.html's footer has the three links, each resolves to an existing section, and the old footer details and
// the "Using this responsibly" card are gone from the main page. Needs no model, so it is quick.
// Run: node tests/about.mjs http://127.0.0.1:8000/       (exit 0 = pass)
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';
const base = (process.argv[2] || 'http://127.0.0.1:8000/').replace(/index\.html$/, '');
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox'] });
let fails = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
const errors = [], mlm = [];
const open = async u => { const p = await browser.newPage(); await p.setViewport({ width: 1200, height: 900 });
  p.on('pageerror', e => errors.push(u + ' ' + e.message)); p.on('console', m => { if (m.type() === 'error') errors.push(u + ' ' + m.text()); });
  p.on('response', r => { if (r.status() >= 400) errors.push(`${u} http ${r.status()} ${r.url()}`); });
  p.on('request', r => { if (/\/(model|data)\/mlm\//.test(r.url())) mlm.push(r.url()); });
  await p.goto(base + u, { waitUntil: 'networkidle0' }); return p; };

const a = await open('about.html');
const info = await a.evaluate(() => ({ ids: ['cards', 'responsible', 'limits', 'related', 'references', 'credits'].map(i => [i, !!document.getElementById(i) && document.getElementById(i).querySelector('h2')?.textContent.trim()]),
  refs: document.querySelectorAll('#refs li').length, refsLoading: document.querySelector('#refs li')?.textContent.startsWith('Loading'),
  fair: document.querySelector('#fair-note')?.textContent || '', back: document.querySelector('#btn-back')?.getAttribute('href'), theme: !!document.querySelector('.theme-switch'),
  grad: /grad-?cam/i.test(document.querySelector('.about-main').innerText), abs: [...document.querySelectorAll('a[href],link[href],script[src]')].map(e => e.getAttribute('href') || e.getAttribute('src')).filter(h => /^\//.test(h)),
  credits: document.getElementById('credits').innerText, cites: document.querySelectorAll('.cite').length, lis: document.querySelectorAll('#limits li').length }));
ok(info.ids.every(([, t]) => t), 'about.html has the six sections: ' + info.ids.map(x => x[1]).join(' | '));
ok(info.refs === 72 && !info.refsLoading, `reference list filled with all 72 entries (${info.refs}; number 12 is retired)`);
ok(/almost equally well/.test(info.fair) && /least faithful for \w+ \(occlusion gap \d\.\d\d, against \d\.\d\d to \d\.\d\d for the others\)/.test(info.fair), 'per-language sentence filled with the data-driven occlusion range');
ok(info.back === 'index.html', 'Back link is relative (index.html)');
ok(info.theme, 'theme switch present');
ok(!info.grad, 'no Grad-CAM text on about.html (reference list included)');
ok(info.abs.length === 0, 'no absolute (root) paths in links or scripts');
ok(info.lis >= 13 && info.cites > 10, `limits (${info.lis} items) and citations kept (${info.cites} cites)`);
ok(/Runs in your browser; nothing leaves your device\./.test(info.credits) && /Visualization to build understanding of how an AI learns meaning\./.test(info.credits) && /Ananya Singh and Cherry Lian/.test(info.credits) && /AI use: Claude Code/.test(info.credits) && /AI_USAGE\.md/.test(info.credits) && !/see AI_USAGE\.md in the repository/.test(info.credits) && /Apache-2\.0/.test(info.credits), 'credits: poster wording, teammates, licenses, AI usage statement');
await a.evaluate(() => document.querySelector('.theme-switch button[data-theme="light"]').click());
ok(await a.evaluate(() => document.documentElement.getAttribute('data-theme') === 'light' && getComputedStyle(document.body).backgroundColor !== 'rgb(13, 13, 13)'), 'light/dark switch works on about.html');
fs.mkdirSync('tests/out_declutter', { recursive: true });
await a.screenshot({ path: 'tests/out_declutter/about.png', fullPage: true });

const i = await open('index.html');
const f = await i.evaluate(() => ({ links: [...document.querySelectorAll('footer a')].map(x => [x.textContent.trim(), x.getAttribute('href')]), details: document.querySelectorAll('footer details').length,
  fold: !!document.querySelector('.card.fold'), resp: /Using this responsibly/.test(document.querySelector('.cards').innerText), fontSize: getComputedStyle(document.querySelector('footer')).fontSize }));
const want = { 'Limits': 'about.html#limits', 'References': 'about.html#references', 'Credits and AI use': 'about.html#credits' };
ok(Object.entries(want).every(([t, h]) => f.links.some(([tt, hh]) => tt === t && hh === h)), 'index footer links: ' + JSON.stringify(f.links));
ok(f.details === 0 && !f.fold && !f.resp, 'no footer details and no "Using this responsibly" card on the main page');
for (const [, h] of Object.entries(want)) { const [u, id] = h.split('#'); const p = await open(u); ok(await p.evaluate(id => !!document.getElementById(id), id), `${h} resolves`); await p.close(); }
const click = await open('index.html'); await click.click('footer a[href="about.html#credits"]'); await click.waitForNavigation({ waitUntil: 'load' }).catch(() => {});
ok(/about\.html#credits$/.test(click.url()), 'clicking the footer link opens about.html#credits: ' + click.url());
await click.click('#btn-back'); await click.waitForNavigation({ waitUntil: 'load' }).catch(() => {});
ok(/index\.html$|\/$/.test(click.url().split('#')[0]), 'Back to the page returns to index.html: ' + click.url());
const mp = await i.evaluate(() => ({ label: document.querySelector('label[for="model-pick"]')?.textContent, sel: document.querySelector('#model-pick')?.value, desc: document.querySelector('#model-desc')?.textContent }));
ok(mp.label === 'Model:' && mp.sel === 'a' && /trained on translations/.test(mp.desc || ''), 'index header: "Model:" picker, default model selected, descriptor: ' + JSON.stringify(mp));
ok(mlm.length === 0, 'no request for model B files while the default model is loaded ' + JSON.stringify(mlm.slice(0, 3)));
ok(errors.length === 0, 'no console or network errors ' + JSON.stringify(errors.slice(0, 3)));
await browser.close(); process.exit(fails ? 1 : 0);
