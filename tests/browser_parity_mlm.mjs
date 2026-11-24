// Title: browser_parity_mlm.mjs, browser parity for the loaded model (B = ?model=b, and A as the control).
// What it does: serves web/ on 127.0.0.1, opens it in headless Chrome with ?model=b (then ?model=a), runs the 5 probe pairs of build/export_model.py
//   (the 7 probe sentences) through the page, and records for each: the browser similarity (window.__mm.state.result.an.sim), the browser integrated-gradients
//   result (token scores, sum, score change) and ids/tokens/word groups. Compares the similarities with eval/variants/mlm/export_parity.json (B) and
//   eval/export_parity.json (A) probe_sims_q8 / probe_sims_torch; tolerance 0.01 against probe_sims_q8. Writes eval/variants/mlm/browser_parity.json.
//   tests/browser_parity_mlm_ig.py then adds the Python IG comparison (same graph, same inputs) to the same file.
// Run (repo root):  node tests/browser_parity_mlm.mjs      (exit 0 = similarity parity pass for B)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { startServer } from './static_server_mlm.mjs';

const PROBE = ['The cat sleeps on the sofa.', 'El gato duerme en el sofá.', 'बिल्ली सोफे पर सो रही है।', '猫在沙发上睡觉。', 'The stock market fell sharply today.', 'Dog bites man.', 'Man bites dog.'];
const PAIRS = [['cat_en_es', 0, 1], ['cat_en_hi', 0, 2], ['cat_en_zh', 0, 3], ['cat_vs_stocks', 0, 4], ['dog_bites_man_vs_man_bites_dog', 5, 6]];
const TOL = 0.01;
const wait = ms => new Promise(r => setTimeout(r, ms));
const srv = await startServer('web');
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', protocolTimeout: 900000,
  args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
const problems = [], out = { generated: new Date().toISOString(), tolerance_vs_q8: TOL, probes: PROBE, runner: 'headless Chrome ' + (await browser.version()) + ', 1 WASM thread, web/ on 127.0.0.1', models: {} };
const refs = { b: JSON.parse(fs.readFileSync('eval/variants/mlm/export_parity.json')), a: JSON.parse(fs.readFileSync('eval/export_parity.json')) };

async function runModel(id) {
  const ctx = await browser.createBrowserContext(), page = await ctx.newPage();
  await page.setViewport({ width: 1500, height: 1000 });
  const errs = [];
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) errs.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => errs.push('pageerror ' + String(e).slice(0, 300)));
  const t0 = Date.now();
  await page.goto(srv.url + '?model=' + id, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__mm && window.__mm.state.result, { timeout: 600000 });
  const loaded = { model: await page.evaluate(() => document.documentElement.dataset.model), title: await page.evaluate(() => document.title), boot_s: (Date.now() - t0) / 1000 };
  const pairs = {};
  for (const [name, i, j] of PAIRS) {
    const a = PROBE[i], b = PROBE[j];
    await page.evaluate((a, b) => { document.querySelector('#in-a').value = a; document.querySelector('#in-b').value = b; document.querySelector('#form').requestSubmit(); }, a, b);
    await page.waitForFunction((a, b) => window.__mm && window.__mm.state.result && window.__mm.state.result.a === a && window.__mm.state.result.b === b && window.__mm.gradDone, { timeout: 600000 }, a, b);
    pairs[name] = await page.evaluate(() => {
      const R = window.__mm.state.result, g = R.grad, grp = gs => gs.map(x => ({ text: x.text, tokens: x.tokens, punct: !!x.punct }));
      const dir = d => d && { sim_graph: d.sim, sim_base: d.simBase, ig_sum: d.igSum, ig_target: d.igTarget, ig_tokens: Array.from(d.ig) };
      return { sim_browser: R.an.sim, idsA: R.A.ids, idsB: R.B.ids, tokensA: R.A.tokens, tokensB: R.B.tokens, groupsA: grp(R.gA), groupsB: grp(R.gB),
        gradError: window.__mm.state.gradError || null, igA: dir(g && g.A), igB: dir(g && g.B), ig_words_A: g ? R.an.scores.ig.A : null, ig_words_B: g ? R.an.scores.ig.B : null, grad_ms: g ? g.ms : null };
    });
    console.log(id, name, pairs[name].sim_browser.toFixed(4), pairs[name].igA ? 'IG sum ' + pairs[name].igA.ig_sum.toFixed(4) + ' vs ' + pairs[name].igA.ig_target.toFixed(4) : 'NO IG', pairs[name].gradError || '');
  }
  await ctx.close();
  return { loaded, pairs, console_problems: errs };
}

for (const id of ['b', 'a']) {
  const r = await runModel(id), ref = refs[id], rows = {};
  let maxQ = 0, maxT = 0;
  for (const [name] of PAIRS) {
    const s = r.pairs[name].sim_browser, q = ref.probe_sims_q8[name], t = ref.probe_sims_torch[name];
    rows[name] = { browser: s, python_q8: q, python_torch: t, abs_diff_q8: Math.abs(s - q), abs_diff_torch: Math.abs(s - t) };
    maxQ = Math.max(maxQ, rows[name].abs_diff_q8); maxT = Math.max(maxT, rows[name].abs_diff_torch);
  }
  const igGaps = Object.fromEntries(PAIRS.map(([name]) => { const p = r.pairs[name]; return [name, p.igA ? { A_sum: p.igA.ig_sum, A_target: p.igA.ig_target, A_gap: p.igA.ig_sum - p.igA.ig_target, B_gap: p.igB.ig_sum - p.igB.ig_target, B_sum: p.igB.ig_sum, B_target: p.igB.ig_target } : null]; }));
  out.models[id] = { ...r, similarity: { rows, max_abs_diff_vs_q8: maxQ, max_abs_diff_vs_torch: maxT, pass_vs_q8: maxQ <= TOL }, ig_completeness: igGaps };
  console.log(id, 'max |browser - python q8| =', maxQ.toFixed(5), maxQ <= TOL ? 'PASS' : 'FAIL', '| vs torch', maxT.toFixed(5));
  if (r.console_problems.length) { console.log('console problems', r.console_problems); problems.push(...r.console_problems); }
}
out.pass_b = out.models.b.similarity.pass_vs_q8;
fs.writeFileSync('eval/variants/mlm/browser_parity.json', JSON.stringify(out, null, 1));
await browser.close(); await srv.close();
console.log('wrote eval/variants/mlm/browser_parity.json');
process.exit(out.pass_b && out.models.a.similarity.pass_vs_q8 ? 0 : 1);
