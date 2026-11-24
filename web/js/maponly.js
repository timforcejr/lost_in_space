// Title: maponly.js, "map only" view of the 3-D map.
// What it does: "?maponly=1" (add "&legend=0" to hide the legend as well) or the "Map only" button in the viewer puts class
// "map-only" on <html>; css/maponly.css then hides everything but the map and its legend, so a screenshot or crop shows just the map.
// A bar at the top right (invisible until hovered or focused) has "Legend" and "Show controls"; Escape also leaves the view. Works in
// the two-screen windows too (?view=map&maponly=1); ?view=explain has no map, so the parameter is ignored there. Sentence pair, color
// and map head come from the page's own state, exactly as in the normal page. main.js is not touched.
(() => {
  const root = document.documentElement;
  const q = new URLSearchParams(location.search);
  const flag = k => q.has(k) && !['0', 'false', 'off'].includes((q.get(k) || '').toLowerCase());
  let on = flag('maponly') && q.get('view') !== 'explain', legend = !(q.has('legend') && !flag('legend'));

  const setUrl = () => {
    const p = new URLSearchParams(location.search);
    if (on) p.set('maponly', '1'); else p.delete('maponly');
    if (on && !legend) p.set('legend', '0'); else p.delete('legend');
    const s = p.toString();
    try { history.replaceState(history.state, '', location.pathname + (s ? '?' + s : '') + location.hash); } catch { /* sandboxed frame */ }
  };
  function apply(url) {
    root.classList.toggle('map-only', on);
    root.classList.toggle('mo-nolegend', !legend);
    const lg = document.querySelector('#btn-mo-legend'); if (lg) lg.setAttribute('aria-pressed', String(legend));
    if (url) setUrl();
    window.dispatchEvent(new Event('resize'));        // the map re-measures its box (it also watches the box itself)
  }
  if (on) root.classList.add('map-only');             // as early as possible
  if (!legend) root.classList.add('mo-nolegend');

  function start() {
    const viewer = document.querySelector('#viewer'); if (!viewer) return;
    const mk = (id, text, title, parent, cls = 'chip-btn') => { const b = document.createElement('button'); b.id = id; b.type = 'button'; b.className = cls; b.textContent = text; b.title = title; parent.append(b); return b; };
    const tr = viewer.querySelector('.overlay.tr');
    if (tr) mk('btn-maponly', 'Map only', 'Hide the controls and show only the map (Escape brings them back)', tr).addEventListener('click', () => { on = true; apply(true); });
    const bar = document.createElement('div'); bar.className = 'mo-bar'; viewer.append(bar);
    const lg = mk('btn-mo-legend', 'Legend', 'Show or hide the legend', bar); lg.setAttribute('aria-pressed', String(legend));
    lg.addEventListener('click', () => { legend = !legend; apply(true); });
    mk('btn-mo-exit', 'Show controls', 'Back to the full page', bar).addEventListener('click', () => { on = false; apply(true); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && on) { on = false; apply(true); } });
    apply(false);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
