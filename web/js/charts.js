// Small dependency-free SVG charts (line, horizontal bar) with hover tooltips and a legend.
//       Colors come from CSS custom properties; text always uses ink tokens, never a series color.
// Verification: visual check of tests/smoke.mjs screenshots

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent?.append(e); return e; };
const html = (tag, cls, text, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; parent?.append(e); return e; };

let tip;
function tooltip() { if (!tip) { tip = html('div', 'chart-tip'); document.body.append(tip); } return tip; }
export function showTip(ev, build) { const t = tooltip(); t.replaceChildren(); build(t); t.style.display = 'block';
  const r = t.getBoundingClientRect(); t.style.left = Math.max(8, Math.min(innerWidth - r.width - 8, ev.clientX + 14)) + 'px'; t.style.top = Math.max(8, ev.clientY - r.height - 12) + 'px'; }
export function hideTip() { if (tip) tip.style.display = 'none'; }
export function tipRow(t, color, label, value) { const row = html('div', 'tip-row', null, t); if (color) { const s = html('span', 'swatch', null, row); s.style.background = color; } html('span', 'tip-label', label, row); html('span', 'tip-value', value, row); }

const markerPath = (g, x, y, r) => ({ circle: null, square: `M${x - r},${y - r}h${2 * r}v${2 * r}h${-2 * r}z`, diamond: `M${x},${y - r * 1.3}L${x + r * 1.3},${y}L${x},${y + r * 1.3}L${x - r * 1.3},${y}z`, triangle: `M${x},${y - r * 1.25}L${x + r * 1.2},${y + r}L${x - r * 1.2},${y + r}z` })[g];
function drawMarker(parent, g, x, y, r, color, cls) {
  const d = markerPath(g, x, y, r);
  return d ? svg('path', { d, fill: color, class: cls || 'mk' }, parent) : svg('circle', { cx: x, cy: y, r, fill: color, class: cls || 'mk' }, parent);
}
export function legend(parent, items) {
  const l = html('div', 'legend', null, parent);
  for (const it of items) {
    const i = html('span', 'legend-item', null, l);
    if (it.glyph || it.dash) {   // line sample with the same dash and marker as the chart line
      const k = svg('svg', { width: 38, height: 12, viewBox: '0 0 38 12', 'aria-hidden': 'true', class: 'legend-line' }, i);
      svg('line', { x1: 1, x2: 37, y1: 6, y2: 6, stroke: it.color, 'stroke-width': Math.min(it.width || 2, 3), 'stroke-linecap': 'round', ...(it.dash ? { 'stroke-dasharray': '5 5' } : {}) }, k);
      if (it.glyph && it.glyph !== 'none') drawMarker(k, it.glyph, 19, 6, 3.4, it.color, 'mk');
    } else { const s = html('span', 'swatch', null, i); s.style.background = it.color; }
    html('span', null, it.name, i);
  }
  return l;
}

/**
 * series: [{name, color, values:number[], dash?:bool, emph?:bool}]; x: labels (same length as values)
 * opts: {yDomain:[lo,hi], yTicks:number[], xTitle, yTitle, fmt, height, endLabels:bool, xEvery}
 */
export function lineChart(el, series, x, opts = {}) {
  el.replaceChildren();
  if (series.length > 1) legend(el, series);
  const W = 520, Hh = opts.height || 220, m = { l: 44, r: opts.endLabels ? 92 : 14, t: 10, b: 36 };
  const [lo, hi] = opts.yDomain, fmt = opts.fmt || (v => v.toFixed(2));
  const root = svg('svg', { viewBox: `0 0 ${W} ${Hh}`, class: 'chart', role: 'img', 'aria-label': opts.aria || 'line chart' }, el);
  const X = i => m.l + (x.length > 1 ? i / (x.length - 1) : 0.5) * (W - m.l - m.r), Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (Hh - m.t - m.b);
  for (const t of opts.yTicks) { svg('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: t === 0 ? 'axis' : 'grid' }, root); svg('text', { x: m.l - 8, y: Y(t) + 4, class: 'tick', 'text-anchor': 'end' }, root).textContent = fmt(t); }
  const every = opts.xEvery || Math.ceil(x.length / 8);
  x.forEach((lab, i) => { if (i % every === 0 || i === x.length - 1) svg('text', { x: X(i), y: Hh - m.b + 16, class: 'tick', 'text-anchor': 'middle' }, root).textContent = lab; });
  if (opts.xTitle) svg('text', { x: (m.l + W - m.r) / 2, y: Hh - 4, class: 'axis-title', 'text-anchor': 'middle' }, root).textContent = opts.xTitle;
  const ends = [];
  for (const s of series) {
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    svg('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || (s.emph ? 3 : 2), 'stroke-linejoin': 'round', 'stroke-linecap': 'round', ...(s.dash ? { 'stroke-dasharray': '5 5' } : {}) }, root);
    if (s.glyph && s.glyph !== 'none') s.values.forEach((v, i) => { if (i < s.values.length - 1) drawMarker(root, s.glyph, X(i), Y(v), 3.6, s.color, 'mk'); });
    const last = s.values.length - 1;
    svg('circle', { cx: X(last), cy: Y(s.values[last]), r: 4, fill: s.color, class: 'end-dot' }, root);
    ends.push({ s, y: Y(s.values[last]) });
  }
  if (opts.endLabels) { // direct labels, nudged apart so they never collide
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
    for (const e of ends) svg('text', { x: W - m.r + 9, y: e.y + 4, class: 'direct' }, root).textContent = e.s.short || e.s.name;
  }
  const cross = svg('line', { y1: m.t, y2: Hh - m.b, class: 'crosshair', visibility: 'hidden' }, root);
  const dots = series.map(s => svg('circle', { r: 4.5, fill: s.color, class: 'hover-dot', visibility: 'hidden' }, root));
  const hit = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh - m.t - m.b, fill: 'transparent' }, root);
  hit.addEventListener('pointermove', ev => {
    const r = root.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(x.length - 1, Math.round((px - m.l) / (W - m.l - m.r) * (x.length - 1))));
    cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('visibility', 'visible');
    series.forEach((s, k) => { dots[k].setAttribute('cx', X(i)); dots[k].setAttribute('cy', Y(s.values[i])); dots[k].setAttribute('visibility', 'visible'); });
    showTip(ev, t => { html('div', 'tip-title', (opts.xTip || (v => v))(x[i]), t); [...series].sort((a, b) => b.values[i] - a.values[i]).forEach(s => tipRow(t, s.color, s.name, fmt(s.values[i]))); });
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); hideTip(); });
}

/** rows: [{name, value, lo?, hi?, note?, muted?}] — one color, value at the bar tip, optional CI whisker. */
export function barChart(el, rows, opts = {}) {
  el.replaceChildren();
  const W = 520, rowH = 30, m = { l: 150, r: 58, t: 6, b: opts.xTitle ? 38 : 22 }, Hh = m.t + m.b + rows.length * rowH;
  const max = opts.max ?? Math.max(...rows.map(r => r.hi ?? r.value)) * 1.05, fmt = opts.fmt || (v => v.toFixed(2));
  const root = svg('svg', { viewBox: `0 0 ${W} ${Hh}`, class: 'chart', role: 'img', 'aria-label': opts.aria || 'bar chart' }, el);
  const X = v => m.l + Math.max(0, v) / max * (W - m.l - m.r);
  for (const t of opts.ticks || []) { svg('line', { x1: X(t), x2: X(t), y1: m.t, y2: Hh - m.b, class: t === 0 ? 'axis' : 'grid' }, root); svg('text', { x: X(t), y: Hh - m.b + 15, class: 'tick', 'text-anchor': 'middle' }, root).textContent = fmt(t); }
  if (opts.xTitle) svg('text', { x: (m.l + W - m.r) / 2, y: Hh - 1, class: 'axis-title', 'text-anchor': 'middle' }, root).textContent = opts.xTitle;
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + 6, h = 18, w = Math.max(1, X(r.value) - m.l);
    svg('text', { x: m.l - 10, y: y + 13, class: 'row-label', 'text-anchor': 'end' }, root).textContent = r.name;
    const rad = Math.min(4, w);
    svg('path', { d: `M${m.l},${y}h${w - rad}a${rad},${rad} 0 0 1 ${rad},${rad}v${h - 2 * rad}a${rad},${rad} 0 0 1 ${-rad},${rad}h${-(w - rad)}z`, fill: r.muted ? 'var(--muted-mark)' : (opts.color || 'var(--series-1)') }, root);
    if (r.lo != null) { svg('line', { x1: X(r.lo), x2: X(r.hi), y1: y + h / 2, y2: y + h / 2, class: 'whisker' }, root); }
    svg('text', { x: X(r.hi ?? r.value) + 7, y: y + 13, class: 'value' }, root).textContent = fmt(r.value);
    const hit = svg('rect', { x: 0, y: y - 6, width: W, height: rowH, fill: 'transparent' }, root);
    hit.addEventListener('pointermove', ev => showTip(ev, t => { html('div', 'tip-title', r.name, t); tipRow(t, null, opts.valueName || 'value', fmt(r.value)); if (r.lo != null) tipRow(t, null, '95% CI', `${fmt(r.lo)} – ${fmt(r.hi)}`); if (r.note) html('div', 'tip-note', r.note, t); }));
    hit.addEventListener('pointerleave', hideTip);
  });
}
