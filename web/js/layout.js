// Title: layout.js, arrange the page's units in two columns, drag them, hide them.
// What it does: the main page has seven units (.blk in #blocks): the 3D map, the Explanation and five evidence cards. They sit in two columns (.col).
// Every unit shows only its title header and one small "−" ("+" to restore) in the header's top-right corner, which folds the unit to its header bar. The header IS the
// drag handle (cursor: grab, subtle hover highlight; no grip icon, no arrow buttons): drag it into either column, to any position, also below the explanation; or focus it
// (tab) and press the arrow keys: up / down inside a column, left / right to the other column. On a card the header is the card's own title (h2) and the "−" sits in the card's
// top-right corner; on the map and the explanation the header is a row above. Only the header drags (links inside it do not), so the 3-D camera keeps its mouse.
// Single-column widths (1100 px and below) have no drag: there, and only there, two small up / down buttons appear next to the "−" (touch has no arrow keys).
// The DEFAULT arrangement (nothing saved) is balanced by measured heights (balance()): map left, explanation right, the five cards split so both columns end about level, re-checked
// as the cards fill in and when the window resizes (with hysteresis, so it settles). The moment the visitor moves a unit, or has a saved layout, the columns are theirs and are never rebalanced.
// Saved per device: the column of each unit, its order and the set of folded units, as one JSON value in localStorage under "lis-layout-v2-d" (desktop / tablet)
// or "lis-layout-v2-p" (phone width, so a phone never overwrites the desktop arrangement), inside try/catch (private windows may refuse). It is not a cookie
// on purpose: it never leaves the browser, so nothing is sent to a server and no consent banner is needed; the old "lis-layout-v1" is dropped. "Reset layout" in the
// page header (shown only when something differs from the default) restores the default columns and expands everything.
// Narrow screens (1100 px and below): one column, drag is off, up / down move a unit along the flattened order (column 1 and 2 interleaved, so the saved arrangement
// still applies). While the map is popped out (html.pv-main) the map unit is gone and the remaining units are re-flowed over both (equal) columns by
// measured height (a guess until they have a box; balanced again 1.5 s and 5 s later) so both stay filled; the saved arrangement is untouched and the move controls are off until the map is docked. Drag uses the browser's drag and drop on the header
// only (the units never start a drag, so the 3-D camera keeps its mouse). The other views (?view=map / graphic, map-only) hide the headers and ignore folding.
(() => {
  const box = document.querySelector('#blocks'); if (!box) return;
  const root = document.documentElement;
  const IDS = ['map', 'why', 'near', 'trustx', 'trustm', 'evid', 'order'];
  const UNITS = ['why', 'near', 'trustx', 'trustm', 'evid', 'order'];   // the six units in the two columns; the map is the fixed full-width row above them
  const DEFAULT = [['near', 'order'], ['why', 'trustx', 'trustm', 'evid']];   // cards under the map and under the explanation; heights balance (about 2500 : 2700 px at 1400 px wide)
  const WEIGHT = {   /* fallback only: */ map: 6.3, why: 6.7, near: 10.2, trustx: 7.9, trustm: 6.3, evid: 6.3, order: 8.7 };   // measured heights / 100 at 1400 px wide, for the re-flow while the map is out
  const KEY = 'lis-layout-v2-' + (matchMedia('(max-width: 640px)').matches ? 'p' : 'd');
  const col = c => box.querySelector(`.col[data-col="${c}"]`);
  const blk = id => box.querySelector(`.blk[data-blk="${id}"]`);
  if (!col(0) || !col(1) || IDS.some(id => !blk(id))) return;
  const title = id => blk(id).dataset.title;
  const isCard = id => blk(id).classList.contains('cards');
  const narrow = () => matchMedia('(max-width: 1100px)').matches;
  const popped = () => root.classList.contains('pv-main') || root.classList.contains('pv-explain');
  const clone = c => c.map(a => a.slice());
  let cols = clone(DEFAULT), folded = new Set(), dragging = null, lastSig = '', auto = true, nBal = 0;   // auto: no saved arrangement yet, the default columns are balanced by measured heights

  const isDefault = () => auto && !folded.size;
  function load() {
    try { localStorage.removeItem('lis-layout-v1'); } catch { /* no storage */ }
    try {
      const v = JSON.parse(localStorage.getItem(KEY));
      if (!v || v.v !== 2 || !Array.isArray(v.cols) || v.cols.length !== 2) return;
      const seen = new Set();
      const c = v.cols.map(a => (Array.isArray(a) ? a.filter(x => UNITS.includes(x) && !seen.has(x) && seen.add(x)) : []));
      UNITS.forEach(id => { if (!seen.has(id)) c[id === 'why' ? 1 : 0].push(id); });
      cols = c; auto = false; folded = new Set((Array.isArray(v.folded) ? v.folded : []).filter(x => IDS.includes(x)));
    } catch { /* no storage, or a value from another version: use the default */ }
  }
  function save() {
    try { if (isDefault()) localStorage.removeItem(KEY); else localStorage.setItem(KEY, JSON.stringify({ v: 2, cols, folded: [...folded] })); } catch { /* no storage */ }
  }
  const flat = cs => { const o = []; for (let i = 0; i < Math.max(cs[0].length, cs[1].length); i++) for (const c of [0, 1]) if (cs[c][i]) o.push(cs[c][i]); return o; };
  const where = (id, cs = cols) => { const c = cs[0].includes(id) ? 0 : 1; return { c, i: cs[c].indexOf(id) }; };
  function reflow() {                                                        // map out: the other units over two columns, the split that balances the two heights best (reading order kept inside each column)
    const ids = flat(cols), hs = ids.map(id => (folded.has(id) ? 0.6 : (blk(id).offsetHeight / 100 || WEIGHT[id]))), total = hs.reduce((x, y) => x + y, 0);
    let best = 0, gap = Infinity;
    for (let m = 0; m < 1 << ids.length; m++) {                              // 2^6 subsets: bit set = right column
      const r = ids.reduce((x, _, k) => x + (m >> k & 1 ? hs[k] : 0), 0), d = Math.abs(total - 2 * r);
      if (d < gap - 1e-9 || (Math.abs(d - gap) < 1e-9 && !(m & 1))) { gap = d; best = m; }   // ties: the first unit stays on the left
    }
    const out = [[], []]; ids.forEach((id, k) => out[best >> k & 1].push(id));
    return out;
  }
  const RANK = ['why', 'near', 'order', 'trustx', 'trustm', 'evid'];
  function balance() {                                                       // default arrangement only: the six units under the map row are split so both columns end level
    if (!auto || popped() || narrow() || dragging) return;
    const hm = {}, keep = document.activeElement, sx = window.scrollX, sy = window.scrollY;   // measuring moves every card, which shortens the document for a moment and the browser clamps scrollY: restore it below. Measured height of each unit in each column's width: moved there for one synchronous measurement, no paint in between
    for (const id of RANK) { hm[id] = [0, 1].map(c => { col(c).appendChild(blk(id)); return (folded.has(id) ? 60 : blk(id).offsetHeight) + 14; }); }
    const size = cs => [cs[0].reduce((x, id) => x + hm[id][0], 0), cs[1].reduce((x, id) => x + hm[id][1], 0)];
    const of = m => [RANK.filter((_, k) => !(m >> k & 1)), RANK.filter((_, k) => m >> k & 1)];
    const gapOf = cs => { const z = size(cs); return Math.abs(z[0] - z[1]); };
    const cur = gapOf(cols);
    let best = null, gap = Infinity;
    for (let m = 1; m < 64; m++) { const cs = of(m), d = gapOf(cs); if (d < gap - 1e-9) { gap = d; best = cs; } }
    if (gap < cur - 60) { cols = best; nBal++; }
    place(true); refresh(); if (keep && keep !== document.body) keep.focus?.({ preventScroll: true });
    if (Math.abs(window.scrollY - sy) > 1) window.scrollTo(sx, sy);
  }
  let balT = 0; const balSoon = () => { clearTimeout(balT); balT = setTimeout(balance, 350); };
  const shown = () => (popped() ? reflow() : cols);

  const live = document.createElement('p'); live.className = 'sr-only'; live.setAttribute('role', 'status'); live.style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap';
  document.body.append(live);
  const resetBtn = document.querySelector('#btn-reset-layout');
  const sizeSoon = () => { window.dispatchEvent(new Event('resize')); requestAnimationFrame(() => window.dispatchEvent(new Event('resize'))); };

  function place(force) {                                                    // DOM follows the model: move only units that are out of place
    const cs = shown();
    cs.forEach((ids, c) => ids.forEach((id, k) => { const el = blk(id), host = col(c); if (host.children[k] !== el) host.insertBefore(el, host.children[k] || null); }));
    blk('map').style.setProperty('--o', -1); flat(cs).forEach((id, k) => blk(id).style.setProperty('--o', k));        // the order used when the page is one column
    const sig = JSON.stringify(cs) + [...folded] + popped();
    if (force || sig !== lastSig) { lastSig = sig; sizeSoon(); }
  }
  const handle = id => blk(id).querySelector('.blk-handle');
  function refresh() {
    if (resetBtn) resetBtn.hidden = isDefault();
    const off = popped(), one = narrow(), fl = flat(cols);
    for (const id of IDS) {
      const b = blk(id), h = b.querySelector('.blk-head'), f = folded.has(id);
      b.classList.toggle('collapsed', f);
      const m = h.querySelector('.blk-min'), t = title(id);
      m.textContent = f ? '+' : '−'; m.setAttribute('aria-expanded', String(!f));
      m.setAttribute('aria-label', `${f ? 'Expand' : 'Collapse'} the ${t} block`); m.title = m.getAttribute('aria-label');
      if (id === 'map') { h.querySelector('.blk-up').disabled = h.querySelector('.blk-down').disabled = true; const g = handle(id); g.draggable = false; g.setAttribute('aria-disabled', 'true'); g.title = 'The map is the full-width row at the top'; g.style.cursor = 'default'; continue; }
      const { c, i } = where(id), k = fl.indexOf(id);
      const first = one ? k === 0 : i === 0, last = one ? k === fl.length - 1 : i === cols[c].length - 1;
      h.querySelector('.blk-up').disabled = off || first; h.querySelector('.blk-down').disabled = off || last;
      const g = handle(id); g.setAttribute('aria-disabled', String(off));
      g.title = off ? `The map is in another window: dock it to move the ${t} block` : (one ? '' : `Drag to move the ${t} block`);
    }
  }
  function commit(msg, focusEl, id) {
    place(); save(); refresh(); if (msg) live.textContent = msg;
    if (focusEl) { const el = focusEl.disabled ? handle(id) : focusEl; el.focus(); }   // the moved unit keeps the keyboard focus
  }
  const spot = id => { const { c, i } = where(id); return narrow() ? `position ${flat(cols).indexOf(id) + 2} of ${IDS.length}` : `column ${c + 1}, position ${i + 1} of ${cols[c].length}`; };
  function move(id, dir, focusEl) {                                          // dir: up, down, left, right
    if (popped() || id === 'map') return;
    auto = false;
    const { c, i } = where(id);
    if (narrow()) {                                                          // one column: swap with the neighbor in the flattened order
      if (dir === 'left' || dir === 'right') return;
      const fl = flat(cols), k = fl.indexOf(id), o = fl[k + (dir === 'up' ? -1 : 1)]; if (!o) return;
      const a = where(id), b = where(o); cols[a.c][a.i] = o; cols[b.c][b.i] = id;
    } else if (dir === 'up' || dir === 'down') {
      const j = i + (dir === 'up' ? -1 : 1); if (j < 0 || j >= cols[c].length) return;
      [cols[c][i], cols[c][j]] = [cols[c][j], cols[c][i]];
    } else {
      const d = dir === 'left' ? 0 : 1; if (d === c) return;
      cols[c].splice(i, 1); cols[d].splice(Math.min(i, cols[d].length), 0, id);
    }
    commit(`${title(id)} moved to ${spot(id)}.`, focusEl, id);
  }
  function dropAt(id, c, beforeId) {
    auto = false;
    cols = cols.map(a => a.filter(x => x !== id));
    const j = beforeId ? cols[c].indexOf(beforeId) : cols[c].length; cols[c].splice(j < 0 ? cols[c].length : j, 0, id);
    commit(`${title(id)} moved to ${spot(id)}.`);
  }
  function toggle(id) {
    folded.has(id) ? folded.delete(id) : folded.add(id);
    place(true); save(); refresh(); balSoon(); live.textContent = `${title(id)} ${folded.has(id) ? 'collapsed' : 'expanded'}.`;
  }

  const clear = () => box.querySelectorAll('.drop-t, .drop-b, .drop-into').forEach(x => x.classList.remove('drop-t', 'drop-b', 'drop-into'));
  function targetOf(e) {                                                     // the column under the pointer (by x) and the unit to insert before (by y)
    const c = e.clientX < col(1).getBoundingClientRect().left - 7 ? 0 : 1;
    const els = [...col(c).querySelectorAll(':scope > .blk')].filter(b => b.dataset.blk !== dragging);
    for (const b of els) { const r = b.getBoundingClientRect(); if (e.clientY < r.top + r.height / 2) return { c, before: b.dataset.blk, el: b, cls: 'drop-t' }; }
    return { c, before: null, el: els[els.length - 1] || col(c), cls: els.length ? 'drop-b' : 'drop-into' };
  }

  for (const id of IDS) {                                                    // headers
    const b = blk(id), t = title(id), card = isCard(id), h = document.createElement('div');
    h.className = 'blk-head in-card';
    const mk = (cls, txt, label) => { const x = document.createElement('button'); x.type = 'button'; x.className = 'blk-move ' + cls; x.textContent = txt; x.setAttribute('aria-label', label); x.title = label; return x; };
    const up = mk('blk-up blk-1col', '▲', `Move the ${t} block up`), down = mk('blk-down blk-1col', '▼', `Move the ${t} block down`);   // shown at single-column widths only
    const min = mk('blk-min', '−', `Collapse the ${t} block`); min.setAttribute('aria-expanded', 'true'); min.setAttribute('aria-controls', 'blk-' + id); b.id = 'blk-' + id;
    let grip;                                                                // the drag handle: the unit's title heading (h2) inside its panel, the same for every unit; a card already has one, the map and the explanation get a unit title
    if (card) grip = b.querySelector('.card > h2');
    else { grip = document.createElement('h2'); grip.className = 'blk-utitle'; grip.textContent = t; }
    grip.classList.add('blk-handle'); grip.draggable = true;
    const focusEl = grip; focusEl.tabIndex = 0;
    if (!card) focusEl.setAttribute('role', 'button');
    focusEl.setAttribute('aria-label', `Move ${t}: use arrow keys`); focusEl.setAttribute('aria-keyshortcuts', 'ArrowUp ArrowDown ArrowLeft ArrowRight');
    for (const [btn, d] of [[up, 'up'], [down, 'down']]) btn.addEventListener('click', () => move(id, d, btn));
    min.addEventListener('click', () => toggle(id));
    focusEl.addEventListener('keydown', e => {
      const d = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' }[e.key]; if (!d || e.target !== focusEl) return;
      e.preventDefault(); move(id, d, focusEl);
    });
    grip.addEventListener('dragstart', e => {
      if (narrow() || popped() || id === 'map' || e.target !== grip) { e.preventDefault(); return; }   // links inside a header do not start a drag
      dragging = id; b.classList.add('dragging');
      try { e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', id); e.dataTransfer.setDragImage(b, 12, 12); } catch { /* synthetic event */ }
    });
    grip.addEventListener('dragend', () => { dragging = null; b.classList.remove('dragging'); clear(); });
    h.append(up, down, min);
    const panel = card ? b.querySelector('.card') : (id === 'map' ? b.querySelector('.viewer') : b.querySelector('.why'));   // every unit: [corner buttons, title] first inside its panel
    if (!card) panel.prepend(grip);
    panel.prepend(h);
  }
  box.addEventListener('dragover', e => {
    if (!dragging) return;
    e.preventDefault(); try { e.dataTransfer.dropEffect = 'move'; } catch { /* synthetic */ }
    const t = targetOf(e); clear(); t.el.classList.add(t.cls);
  });
  box.addEventListener('dragleave', e => { if (!box.contains(e.relatedTarget)) clear(); });
  box.addEventListener('drop', e => {
    if (!dragging) return;
    e.preventDefault();
    const t = targetOf(e), id = dragging; clear();
    dragging = null; box.querySelectorAll('.dragging').forEach(x => x.classList.remove('dragging'));
    dropAt(id, t.c, t.before);
  });
  resetBtn?.addEventListener('click', () => { cols = clone(DEFAULT); folded = new Set(); auto = true; nBal = 0; balSoon(); commit('Layout reset.'); });

  load(); place(true); refresh(); balSoon();
  if (window.ResizeObserver) { const ro = new ResizeObserver(balSoon); IDS.forEach(id => ro.observe(blk(id))); }   // cards fill in after the first paint and when the window changes size
  const later = () => { if (popped()) for (const ms of [1500, 5000]) setTimeout(() => { if (popped()) place(); }, ms); };   // content arrives after the first paint: balance again
  later();
  new MutationObserver(() => { place(); refresh(); later(); }).observe(root, { attributes: true, attributeFilter: ['class'] });   // the map unit leaves / returns on pop out / dock
  matchMedia('(max-width: 1100px)').addEventListener?.('change', refresh);
})();

// chart text at one size in either column: every svg.chart scales with its column (width 100% of a fixed viewBox); --k = that scale, css divides the text size by it
(() => {
  if (!window.ResizeObserver) return;
  const ro = new ResizeObserver(es => es.forEach(e => { const s = e.target, vb = s.viewBox && s.viewBox.baseVal, w = s.getBoundingClientRect().width; if (vb && vb.width && w) s.style.setProperty('--k', (w / vb.width).toFixed(4)); }));
  const seen = new WeakSet(), scan = n => (n.querySelectorAll ? n.querySelectorAll('svg.chart') : []).forEach(s => { if (!seen.has(s)) { seen.add(s); ro.observe(s); } });
  scan(document); new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => n.nodeType === 1 && (n.matches?.('svg.chart') ? (seen.has(n) || (seen.add(n), ro.observe(n))) : scan(n))))).observe(document.body, { childList: true, subtree: true });
})();
