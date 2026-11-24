// Title: popout.js, pop the graphic out into its own window and dock it back.
// What it does: "Pop out" opens "?view=graphic" in a new window: the whole 3-D map block (graphic, legend, every control) fills
// that window (and resizes with it). This window becomes "?view=main": it keeps the sentence boxes and all the explanation tools; the map block is removed from
// its layout, and its (hidden) controls follow the pop-out's through the BroadcastChannel below. Camera rotation and zoom stay local.
// Docking back: the pop-out posts "dock" on pagehide / beforeunload / unload and a "beat" every second; this window polls the
// window handle's .closed flag every 500 ms (the handle is lost if this window is reloaded: then the "dock" message, or no beat
// for ~3 s, docks it). A reload of the pop-out also posts "dock", so a "dock" waits 2 s for a new beat before it acts. The button
// reads "Dock" while popped out; in the pop-out it is a "Dock" chip that closes the window. All of it is static (GitHub Pages).
// Older names stay: "?view=map" (boxes + map) and "?view=explain" (explanations only) still work; no parameter is the normal page.
// Every open window of the page stays in sync through a BroadcastChannel (same origin, no server): the sentence pair and the controls.
// main.js is not touched. The pair is caught by wrapping history.replaceState (main.js writes #a=..&b=.. after every
// run, whatever started it); controls are replayed with .click() or a dispatched input event on the same elements.
// Not synced, by design: camera rotation and zoom, hover tips, help pop-ups, text typed before "Map them".
(() => {
  const root = document.documentElement;
  const $ = s => document.querySelector(s);
  const VIEWS = ['map', 'explain', 'graphic', 'main'];   // graphic = the pop-out, main = the window it came from
  const origReplace = history.replaceState.bind(history);

  // ------------------------------------------------------------------ views
  let view = new URLSearchParams(location.search).get('view');
  if (!VIEWS.includes(view)) view = null;
  const urlFor = v => {
    const q = new URLSearchParams(location.search);
    if (v) q.set('view', v); else q.delete('view');
    const s = q.toString();
    return location.pathname + (s ? '?' + s : '') + location.hash;
  };
  function setView(v) {
    view = v;
    VIEWS.forEach(x => root.classList.toggle('pv-' + x, x === v));
    origReplace(history.state, '', urlFor(v));
    const note = $('#popout-note'); if (note) note.hidden = v !== 'main';
    const btn = $('#btn-popout');
    if (btn) {
      btn.textContent = v ? 'Dock' : 'Pop out';
      btn.title = v === 'main' ? 'Bring the map back into this window and close the map window'
        : v ? 'Show the whole page in this window again'
        : 'Open the map in its own window (for a second screen); this window keeps the sentences, settings and explanations';
    }
  }
  function message(html, keep) {
    let m = $('#popout-msg');
    if (!m) { m = document.createElement('p'); m.id = 'popout-msg'; m.className = 'popout-msg'; m.setAttribute('role', 'status'); ($('#btn-popout') || document.body).after(m); }
    clearTimeout(message.t);
    m.replaceChildren(); if (!html) { m.hidden = true; return; }
    m.append(...html); m.hidden = false;
    if (!keep) message.t = setTimeout(() => { m.hidden = true; m.replaceChildren(); }, 6000);
  }
  const text = t => [document.createTextNode(t)];

  // ------------------------------------------------------------------ main window: pop out, watch the pop-out, dock back
  let popWin = null, lastBeat = 0, grace = 3000, dockTimer = 0, poll = 0;
  function watchPopout() {
    clearInterval(poll);
    poll = setInterval(() => {
      if (view !== 'main') { clearInterval(poll); return; }
      if (popWin) { if (popWin.closed) dockBack('window closed'); }                    // the handle is authoritative while we have it
      else if (Date.now() - lastBeat > grace) dockBack('no heartbeat');                // handle lost (reload) or a link-opened tab
    }, 500);
  }
  function enterMain(win, startGrace) {
    popWin = win || null; lastBeat = Date.now(); grace = startGrace; clearTimeout(dockTimer);
    setView('main'); watchPopout();
  }
  function dockBack(why, quiet) {
    clearInterval(poll); clearTimeout(dockTimer); popWin = null;
    if (view !== 'main') return;
    setView(null);
    window.dispatchEvent(new Event('resize'));                                           // the map re-measures its box
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));              // again once the panel has its full height
    if (!quiet) message(text('The map is back in this window.'));
    dockBack.why = why;
  }
  function dockClick() {                                                                 // the "Dock" button in the main window
    try { if (popWin && !popWin.closed) popWin.close(); } catch { /* ignore */ }
    post({ t: 'closepop' });                                                             // also reaches a pop-out whose handle we lost
    dockBack('button');
  }
  function popOut() {
    const mq = new URLSearchParams(location.search).get('model');
    const target = '?view=graphic' + (mq ? '&model=' + encodeURIComponent(mq) : '') + location.hash;
    const w = window.open(target, 'lis-graphic', 'popup,width=1100,height=800');
    if (w) { message(null); enterMain(w, 10000); return; }
    // pop-up blocked: a link the reader clicks is a user gesture, so it opens; this window then keeps the controls
    const a = document.createElement('a'); a.href = target; a.target = '_blank'; a.textContent = new URL(target, location.href).href;
    a.addEventListener('click', () => { message(null); enterMain(null, 10000); });
    message([document.createTextNode('The browser blocked the pop-up. Open this address in a second window: '), a], true);
  }

  // ------------------------------------------------------------------ pop-out window: heartbeat, dock message, Dock button
  function startGraphic() {
    const beat = () => post({ t: 'beat' });
    beat(); setInterval(beat, 1000);
    const bye = () => post({ t: 'dock' });
    for (const ev of ['pagehide', 'beforeunload', 'unload']) window.addEventListener(ev, bye);
    const tr = $('#viewer .overlay.tr');
    if (tr) {
      const b = document.createElement('button'); b.id = 'btn-dock'; b.type = 'button'; b.className = 'chip-btn'; b.textContent = 'Dock';
      b.title = 'Close this window and bring the map back into the main window';
      b.addEventListener('click', () => {
        bye(); window.close();
        setTimeout(() => { if (!window.closed) message(text('This browser did not let the page close its own window. Close this window yourself; the map returns to the main window.'), true); }, 400);
      });
      tr.append(b);
    }
  }

  // ------------------------------------------------------------------ controls (container id -> read / write)
  const seg = attr => ({
    read: el => el.querySelector('button.on')?.dataset[attr] ?? null,
    write: (el, v) => { const b = el.querySelector(`button[data-${attr}="${CSS.escape(v)}"]`); if (b && !b.classList.contains('on')) b.click(); },
  });
  const range = { ev: 'input', read: el => el.value, write: (el, v) => { if (el.value !== v) { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); } } };
  const CONTROLS = {                                      // replay order matters: color before the topic legend
    'seg-color': seg('v'),
    'legend3d': {                                         // topic isolate (color = topic): index of the only lit button, or -1
      read: el => { const bs = [...el.querySelectorAll('button')], on = bs.filter(b => !b.classList.contains('off')); return bs.length && on.length === 1 && bs.length > 1 ? String(bs.indexOf(on[0])) : '-1'; },
      write: (el, v) => {
        const cur = CONTROLS.legend3d.read(el); if (cur === v) return;
        const bs = el.querySelectorAll('button');
        if (v === '-1') bs[+cur]?.click(); else bs[+v]?.click();
      },
    },
    'seg-head': seg('v'),
    'lang-toggles': {                                     // "1101": which languages are shown; buttons are rebuilt on every change
      read: el => [...el.querySelectorAll('button')].map(b => (b.classList.contains('off') ? '0' : '1')).join(''),
      write: (el, v) => {
        for (const want of ['1', '0']) {                  // switch on first: main.js always keeps one language on
          for (let k = 0; k < v.length; k++) {
            const bs = $('#lang-toggles').querySelectorAll('button');
            if (v[k] === want && bs[k] && (bs[k].classList.contains('off') ? '0' : '1') !== want) bs[k].click();
          }
        }
      },
    },
    'btn-words': { read: el => (el.classList.contains('on') ? '1' : '0'), write: (el, v) => { if (CONTROLS['btn-words'].read(el) !== v) el.click(); } },
    'layer': range,
    'tabs': seg('m'),
    'seg-matrix': seg('v'),
    'seg-bench': seg('v'),
  };
  const ACTIONS = ['btn-play', 'btn-scramble'];           // replayed as a click; not part of a snapshot
  const readAll = () => Object.fromEntries(Object.entries(CONTROLS).map(([id, c]) => [id, $('#' + id) ? c.read($('#' + id)) : null]).filter(([, v]) => v != null));

  // ------------------------------------------------------------------ sync state
  const me = Math.random().toString(36).slice(2) + Date.now().toString(36);
  const chan = 'BroadcastChannel' in window ? new BroadcastChannel('lost-in-space') : null;
  const post = msg => { try { chan?.postMessage({ ...msg, from: me }); } catch { /* window closing */ } };
  const keyOf = (a, b) => JSON.stringify([a, b]);
  let ver = 0, verId = '';                                // Lamport clock for pairs: (ver, id) orders concurrent changes
  const newer = (v, id) => v > ver || (v === ver && id > verId);
  let booted = false, fresh = true, curKey = null, pending = null;
  const remoteKeys = new Set();                           // pairs asked for by other windows that this window has not yet finished
  let wanted = {}, applying = false;
  const ready = () => !!(window.__mm && window.__mm.state && window.__mm.state.result);
  const curModel = () => root.dataset.model || null;      // set by js/models.js when main.js starts
  function followModel(id, once) {                        // another window has model `id`: load it here too
    if (!id || !curModel() || id === curModel() || !window.lisModel) return false;
    if (once) {
      const q = new URLSearchParams(location.search);
      if (q.get('fallback') === id) return false;   // this window just fell back from that model: no loop
      if (q.has('model')) return false;             // opened with an explicit ?model=: keep it, do not adopt another tab's model
      try { const t = +sessionStorage.getItem('lis.modelAdopt') || 0; if (Date.now() - t < 15000) return false; sessionStorage.setItem('lis.modelAdopt', String(Date.now())); } catch { /* no session storage: switch anyway */ }
    }
    window.lisModel.switchTo(id);
    return true;
  }

  function flush() {                                      // apply remote control values once main.js is running and the pair is current
    if (!ready() || pending) return;
    const w = wanted; wanted = {};
    applying = true;
    try { for (const id of Object.keys(CONTROLS)) if (id in w) { const el = $('#' + id); if (el) CONTROLS[id].write(el, w[id]); } }
    finally { applying = false; }
  }
  function want(a, b) {                                   // show this pair (asked for by another window)
    const k = keyOf(a, b);
    if (k === curKey && !pending) return;
    pending = k; remoteKeys.add(k);
    $('#in-a').value = a; $('#in-b').value = b;
    if (ready()) resubmit();
    // not ready: main.js's first run reads the boxes (it keeps anything already in them)
  }
  function resubmit() {
    const st = window.__mm.state;
    if (st.busy && st.again) return;                      // main.js will already run once more with the boxes
    if (pending) { const [a, b] = JSON.parse(pending); $('#in-a').value = a; $('#in-b').value = b; }
    $('#form').requestSubmit();
  }

  // main.js calls history.replaceState('#a=..&b=..') after every successful run
  history.replaceState = function (s, t, url) {
    const r = origReplace(s, t, url);
    try { onUrl(url); } catch (e) { /* never break main.js */ }
    return r;
  };
  function onUrl(url) {
    if (url == null) return;
    const h = new URLSearchParams(new URL(String(url), location.href).hash.slice(1));
    const a = h.get('a'), b = h.get('b'); if (a == null || b == null) return;
    // main.js writes only the hash: keep ?view=.. in the address so a reload stays in the same mode
    if (view && !new URL(String(url), location.href).searchParams.has('view')) origReplace(history.state, '', urlFor(view));
    const k = keyOf(a, b), first = !booted; booted = true;
    if (k !== curKey) {
      curKey = k;
      if (pending && k === pending) { pending = null; remoteKeys.clear(); }
      else if (remoteKeys.has(k)) { /* an older remote pair finishing: stay quiet */ }
      else if (!first) {                                  // a run this window's reader started: tell the others
        pending = null; remoteKeys.clear(); fresh = false;
        ver += 1; verId = me;
        post({ t: 'pair', a, b, ver, vid: verId });
      }
    }
    setTimeout(() => { if (pending && curKey !== pending && ready()) resubmit(); flush(); }, 0);
  }

  function onMessage(m) {
    if (!m || m.from === me) return;
    if (m.t === 'beat') { if (view === 'main') { lastBeat = Date.now(); grace = 3000; clearTimeout(dockTimer); } return; }
    if (m.t === 'dock') { if (view === 'main') { clearTimeout(dockTimer); dockTimer = setTimeout(() => dockBack('dock message'), 2000); } return; }   // wait: a reload posts it too
    if (m.t === 'closepop') { if (view === 'graphic') window.close(); return; }
    if (m.t === 'model') { followModel(m.id, false); return; }
    if (m.t === 'hello' && view === 'graphic') post({ t: 'beat' });
    if (m.t === 'hello') {
      if (!ready()) return;                               // nothing worth adopting yet
      const R = window.__mm.state.result;
      post({ t: 'snapshot', a: R.a, b: R.b, ver, vid: verId, ctl: { ...readAll(), ...wanted }, model: curModel() });
    } else if (m.t === 'snapshot') {
      if (fresh && m.model && followModel(m.model, true)) return;   // a new window adopts the open windows' model first
      if (!fresh && !newer(m.ver, m.vid)) return;
      fresh = false;
      if (newer(m.ver, m.vid)) { ver = m.ver; verId = m.vid; }
      Object.assign(wanted, m.ctl || {});
      want(m.a, m.b);
      flush();
    } else if (m.t === 'pair') {
      if (!newer(m.ver, m.vid)) return;
      ver = m.ver; verId = m.vid; fresh = false;
      want(m.a, m.b);
    } else if (m.t === 'ctl') {
      if (ACTIONS.includes(m.id)) { if (ready()) { applying = true; try { $('#' + m.id)?.click(); } finally { applying = false; } } return; }
      if (!(m.id in CONTROLS)) return;
      wanted[m.id] = m.v;
      flush();
    }
  }

  function watch() {
    for (const [id, c] of Object.entries(CONTROLS)) {
      const el = $('#' + id); if (!el) continue;
      el.addEventListener(c.ev || 'click', () => {
        if (applying) return;                             // a replayed change: do not echo it back
        setTimeout(() => {                                // after main.js has updated the control
          if (!ready()) return;
          const v = c.read(el);
          if (v != null) { fresh = false; post({ t: 'ctl', id, v }); }
        }, 0);
      });
    }
    for (const id of ACTIONS) $('#' + id)?.addEventListener('click', () => { if (!applying && ready()) post({ t: 'ctl', id, v: 'click' }); });
  }

  function start() {
    setView(view);
    $('#btn-popout')?.addEventListener('click', () => (view === 'main' ? dockClick() : view ? (message(null), setView(null)) : popOut()));
    if (view === 'main') { lastBeat = Date.now(); grace = 3000; watchPopout(); }     // this window was reloaded while popped out
    if (view === 'graphic') startGraphic();
    if (!chan) return;
    chan.onmessage = ev => { try { onMessage(ev.data); } catch (e) { /* ignore a malformed message */ } };
    watch();
    window.addEventListener('lis-model-pick', ev => post({ t: 'model', id: ev.detail && ev.detail.id }));
    post({ t: 'hello', model: curModel() });
  }
  if (view) VIEWS.forEach(x => root.classList.toggle('pv-' + x, x === view));   // as early as possible: less flash
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
