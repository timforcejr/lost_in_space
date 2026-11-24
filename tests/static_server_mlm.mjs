// Title: static_server_mlm.mjs, a small static file server for web/ on 127.0.0.1 used by browser_parity_mlm.mjs and load_timing_mlm.mjs.
// What it does: serves web/ on an ephemeral port bound to 127.0.0.1 only, with GitHub-Pages-like headers (ETag, Last-Modified, Cache-Control: max-age=600, 304 on a matching ETag,
// gzip for text types, sent chunked without Content-Length as GitHub Pages does), so warm-load behavior resembles the deployed site. Logs nothing. Export: startServer(webDir) -> {url, close(), hits}.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.onnx_data': 'application/octet-stream', '.bin': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.ico': 'image/x-icon', '.txt': 'text/plain; charset=utf-8', '.md': 'text/plain; charset=utf-8' };
const GZ = new Set(['.html', '.js', '.mjs', '.css', '.json', '.svg', '.txt']);

export function startServer(webDir, { cacheControl = 'public, max-age=600' } = {}) {
  const root = path.resolve(webDir), hits = [];
  const server = http.createServer((req, res) => {
    try {
      const u = new URL(req.url, 'http://x');
      let p = path.normalize(path.join(root, decodeURIComponent(u.pathname)));
      if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
      if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
      if (!fs.existsSync(p)) { hits.push([404, u.pathname, 0, Date.now()]); res.writeHead(404).end('not found'); return; }
      const st = fs.statSync(p), ext = path.extname(p), etag = `"${st.size.toString(16)}-${Math.floor(st.mtimeMs).toString(16)}"`;
      const h = { 'Content-Type': TYPES[ext] || 'application/octet-stream', ETag: etag, 'Last-Modified': st.mtime.toUTCString(), 'Cache-Control': cacheControl, 'Accept-Ranges': 'none' };
      if (req.headers['if-none-match'] === etag) { hits.push([304, u.pathname, 0, Date.now()]); res.writeHead(304, h).end(); return; }
      if (GZ.has(ext) && /gzip/.test(req.headers['accept-encoding'] || '')) {
        const body = zlib.gzipSync(fs.readFileSync(p)); hits.push([200, u.pathname, req.method === 'HEAD' ? 0 : body.length, Date.now(), req.method]); res.writeHead(200, { ...h, 'Content-Encoding': 'gzip', Vary: 'Accept-Encoding' }); res.end(body); return;
      }
      hits.push([200, u.pathname, req.method === 'HEAD' ? 0 : st.size, Date.now(), req.method]);
      res.writeHead(200, { ...h, 'Content-Length': st.size });
      fs.createReadStream(p).pipe(res);
    } catch (e) { res.writeHead(500).end(String(e)); }
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ url: `http://127.0.0.1:${server.address().port}/`, hits, close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }) })));
}
