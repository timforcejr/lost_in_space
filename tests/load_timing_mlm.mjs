// Title: load_timing_mlm.mjs, cold / warm load timing for model A (default) and model B (?model=b).
// What it does: serves web/ on 127.0.0.1 (GitHub-Pages-like cache headers, see static_server_mlm.mjs) and, in headless Chrome with a FRESH profile per
//   profile-run (empty HTTP cache, empty localStorage), measures for two network profiles (unthrottled; 20 Mbit/s down+up, 20 ms latency, via CDP
//   Network.emulateNetworkConditions): A cold; A warm (page.reload); switch A->B through the picker (B cold, A already cached); B warm reload; B warm by
//   navigation; switch B->A; and, in a second fresh profile, B cold loaded directly (?model=b). Per step: time until the model is loaded (the first "Thinking..." chip: the first pair starts) and to the first result
//   chip (page-side MutationObserver on #status-text), time until the whole chain is done (deletion, gradients, order probe), bytes transferred by file
//   group (CDP encodedDataLength, both at first result and when settled, cross-checked with the server's own log), renderer RSS and JS heap at settle.
//   Writes eval/variants/mlm/load_timing.json.
// Run (repo root):  node tests/load_timing_mlm.mjs [profileName ...]      profileNames: unthrottled, mbit20 (default: both)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { startServer } from './static_server_mlm.mjs';

const wait = ms => new Promise(r => setTimeout(r, ms));
const PROFILES = { unthrottled: null, mbit20: { offline: false, latency: 20, downloadThroughput: 20e6 / 8, uploadThroughput: 20e6 / 8 } };
const want = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(PROFILES);
const srv = await startServer('web');
const group = u => {
  const p = new URL(u).pathname;
  if (/\/model\/mlm\//.test(p)) return /onnx_data$/.test(p) ? 'mlm_weights_data' : /\.onnx$/.test(p) ? 'mlm_weights_graph' : 'mlm_config';
  if (/\/model\/mmini\/onnx\//.test(p)) return /onnx_data$/.test(p) ? 'mmini_weights_data' : 'mmini_weights_graph';
  if (/\/model\/mmini\//.test(p)) return 'tokenizer_and_config(shared)';
  if (/\/model\/mgrad_mlm\//.test(p)) return 'mgrad_mlm';
  if (/\/model\/mgrad\//.test(p)) return 'mgrad';
  if (/\/data\/mlm\//.test(p)) return 'data_mlm';
  if (/\/data\//.test(p)) return 'data_default';
  if (/\/vendor\//.test(p)) return 'vendor(js+wasm)';
  return 'app(html/js/css)';
};
const WEIGHT_GROUPS = ['mmini_weights_data', 'mmini_weights_graph', 'mlm_weights_data', 'mlm_weights_graph'];

async function rssOf(browser) {
  const cdp = await browser.target().createCDPSession();
  try {
    const { processInfo } = await cdp.send('SystemInfo.getProcessInfo');
    let kb = 0; const pids = [];
    for (const p of processInfo.filter(p => p.type === 'renderer')) { try { const m = /VmRSS:\s+(\d+) kB/.exec(fs.readFileSync(`/proc/${p.id}/status`, 'utf8')); if (m) { kb += +m[1]; pids.push(p.id); } } catch {} }
    return { renderer_rss_mb: Math.round(kb / 1024), renderer_pids: pids };
  } catch (e) { return { renderer_rss_mb: null, err: String(e) }; } finally { await cdp.detach().catch(() => {}); }
}

async function session(profileName, steps) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lis-timing-'));      // a fresh user-data dir = an empty HTTP cache and empty localStorage
  const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new', userDataDir: dir, protocolTimeout: 1800000,
    args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=1500,1100', '--enable-precise-memory-info', '--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
  const page = await browser.newPage(); await page.setViewport({ width: 1500, height: 1000 });
  const cdp = await page.createCDPSession(); await cdp.send('Network.enable');
  if (PROFILES[profileName]) await cdp.send('Network.emulateNetworkConditions', PROFILES[profileName]);
  const problems = [];
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) problems.push(m.text().slice(0, 200)); });
  page.on('pageerror', e => problems.push('pageerror ' + String(e).slice(0, 200)));
  await page.evaluateOnNewDocument(() => {
    window.__t = { log: [], origin: performance.timeOrigin };
    const rec = () => { const e = document.querySelector('#status-text'); if (!e) return; const t = e.textContent, L = window.__t.log; if (!(/^(Ready|Thinking|Could not|The )/.test(t) || !L.some(x => /^Loading/.test(x[1])))) return; if (!L.length || L[L.length - 1][1] !== t) L.push([performance.timeOrigin + performance.now(), t]); };
    new MutationObserver(rec).observe(document, { subtree: true, childList: true, characterData: true });
  });
  let reqs = new Map();
  cdp.on('Network.requestWillBeSent', e => { if (!reqs.has(e.requestId)) reqs.set(e.requestId, { url: e.request.url, t0: e.timestamp, wall0: e.wallTime * 1000 }); });
  cdp.on('Network.responseReceived', e => { const r = reqs.get(e.requestId); if (r) { r.status = e.response.status; r.cache = !!(e.response.fromDiskCache || e.response.fromPrefetchCache || e.response.fromServiceWorker); } });
  cdp.on('Network.loadingFinished', e => { const r = reqs.get(e.requestId); if (r) { r.bytes = e.encodedDataLength; r.end = r.wall0 + (e.timestamp - r.t0) * 1000; } });
  cdp.on('Network.loadingFailed', e => { const r = reqs.get(e.requestId); if (r) { r.failed = e.errorText; r.end = r.wall0 + (e.timestamp - r.t0) * 1000; } });

  const bytesBy = (until = Infinity) => { const g = {}; let total = 0, n = 0; for (const r of reqs.values()) { if (r.end == null || r.end > until || r.url.startsWith('data:')) continue; const k = group(r.url); (g[k] ??= { mb: 0, requests: 0, from_cache: 0 }); g[k].mb += (r.bytes || 0) / 1e6; g[k].requests++; if (r.cache || r.bytes === 0) g[k].from_cache++; total += r.bytes || 0; n++; } for (const v of Object.values(g)) v.mb = +v.mb.toFixed(3); return { total_mb: +(total / 1e6).toFixed(3), requests: n, groups: g }; };
  const settled = async (prevOrigin, timeout = 1500000) => { const t0 = Date.now(); for (;;) { try { if (await page.evaluate(o => window.__t.origin !== o && !!(window.__mm && window.__mm.state.result && window.__mm.orderDone), prevOrigin)) return true; } catch {} if (Date.now() - t0 > timeout) return false; await wait(250); } };
  const results = [];

  async function step(name, action, expectModel) {
    reqs = new Map(); const hit0 = srv.hits.length, tStart = Date.now(); const prevOrigin = await page.evaluate(() => window.__t ? window.__t.origin : 0).catch(() => 0);
    const t0 = await action();                                               // returns epoch ms of "t=0" (navigation start / the moment the picker changed)
    const ok = await settled(prevOrigin), tSettled = Date.now();
    const info = await page.evaluate(() => ({ log: window.__t.log, origin: window.__t.origin, model: document.documentElement.dataset.model, title: document.title,
      chip: document.querySelector('#status-text').textContent, heap_mb: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null, heap_total_mb: performance.memory ? Math.round(performance.memory.totalJSHeapSize / 1048576) : null,
      grad: !!(window.__mm.state.result && window.__mm.state.result.grad), gradError: window.__mm.state.gradError || null }));
    await wait(500);
    const loaded = info.log.find(x => /^Thinking/.test(x[1])), first = info.log.find(x => /^Ready/.test(x[1]) && /s in your browser/.test(x[1]) && !/runs in/.test(x[1]));
    const mem = await rssOf(browser);
    const sv = srv.hits.slice(hit0).filter(h => h[3] >= tStart);
    const wgt = g => srv.hits.slice(hit0).filter(h => g.test(h[1])).map(h => ({ status: h[0], mb: +(h[2] / 1e6).toFixed(2) }));
    const row = { step: name, profile: profileName, expected_model: expectModel, model_loaded: info.model, title: info.title, settled: ok, chain_done_s: +((tSettled - t0) / 1000).toFixed(1),
      model_ready_first_pair_starts_s: loaded ? +((loaded[0] - t0) / 1000).toFixed(2) : null, first_result_chip_s: first ? +((first[0] - t0) / 1000).toFixed(2) : null,
      bytes_until_first_result: first ? bytesBy(first[0]) : null, bytes_until_settled: bytesBy(), ...mem, js_heap_used_mb: info.heap_mb, js_heap_total_mb: info.heap_total_mb, ig_ran: info.grad, grad_error: info.gradError,
      server_bytes_mb: +(sv.reduce((s, h) => s + h[2], 0) / 1e6).toFixed(1), server_onnx_data: wgt(/onnx_data$/).filter(x => x.mb > 0),
      weight_groups_with_bytes: Object.fromEntries(WEIGHT_GROUPS.map(g => [g, bytesBy().groups[g] ? bytesBy().groups[g].mb : 0])) };
    results.push(row);
    console.log(`${profileName} ${name}: model=${row.model_loaded} model ready ${row.model_ready_first_pair_starts_s}s first result ${row.first_result_chip_s}s chain ${row.chain_done_s}s bytes first ${row.bytes_until_first_result && row.bytes_until_first_result.total_mb} MB settled ${row.bytes_until_settled.total_mb} MB rss ${row.renderer_rss_mb} MB`);
    return row;
  }
  const nav = (u, opts) => async () => { const t = Date.now(); await page.goto(u, { waitUntil: 'domcontentloaded', ...opts }); return await page.evaluate(() => window.__t.origin); };
  const reload = async () => { await page.reload({ waitUntil: 'domcontentloaded' }); return await page.evaluate(() => window.__t.origin); };
  const pick = id => async () => { const t = Date.now(); await page.evaluate(id => { const s = document.querySelector('#model-pick'); s.value = id; s.dispatchEvent(new Event('change', { bubbles: true })); }, id); return t; };
  await steps({ step, nav, reload, pick, srvUrl: srv.url });
  await browser.close(); fs.rmSync(dir, { recursive: true, force: true });
  return { results, problems };
}

const all = { generated: new Date().toISOString(), chrome: null, note: 'times in seconds from navigation start (cold/warm/reload) or from the moment the picker changed (switch); first_result_chip_s = first "Ready ... s in your browser" chip (model loaded and first pair computed); chain_done_s = until the deletion test, gradients and word-order test are done', profiles: {}, problems: [] };
for (const prof of want) {
  const run = {};
  const a = await session(prof, async ({ step, nav, reload, pick, srvUrl }) => {
    await step('A_cold', nav(srvUrl), 'a');
    await step('A_warm_reload', reload, 'a');
    await step('switch_A_to_B (B cold, A cached)', pick('b'), 'b');
    await step('B_warm_reload', reload, 'b');
    await step('B_warm_navigate', nav(srvUrl), 'b');
    await step('switch_B_to_A (A warm)', pick('a'), 'a');
  });
  const b = await session(prof, async ({ step, nav, srvUrl }) => { await step('B_cold_direct (fresh profile, ?model=b)', nav(srvUrl + '?model=b'), 'b'); });
  run.steps = [...a.results, ...b.results]; all.problems.push(...a.problems, ...b.problems);
  all.profiles[prof] = run;
  fs.writeFileSync('eval/variants/mlm/load_timing.json', JSON.stringify(all, null, 1));
}
await srv.close();
console.log('wrote eval/variants/mlm/load_timing.json');
