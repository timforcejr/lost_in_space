// Title: about.js, fills in the two data-driven parts of about.html.
// What it does: (1) the reference list from data/references.json, (2) the per-language sentence in "Using this responsibly" from
// data/eval.json and data/cloud.json. Both pieces of code moved here unchanged from js/main.js when the footer text moved to about.html.
// The page reads fine without them (static text stays); only these two spots stay on "Loading…" or generic wording if a file is missing.
const $ = s => document.querySelector(s);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const get = file => fetch('./data/' + file).then(r => (r.ok ? r.json() : null)).catch(() => null);

get('references.json').then(d => {
  if (!d) { $('#refs').replaceChildren(el('li', null, 'The reference list could not be loaded. Reload the page to try again (it is also kept as REFERENCES.md in the repository).')); return; }
  $('#refs').replaceChildren(...d.map(r => { const li = el('li', null, r.text + ' '); li.value = r.n; const a = el('a', null, r.url); a.href = r.url; a.rel = 'noopener'; a.target = '_blank'; li.append(a); return li; }));
});

Promise.all([get('eval.json'), get('cloud.json')]).then(([evalData, cloud]) => {
  if (evalData && cloud) {                                   // per-language numbers for the responsible-use section
    const r = cloud.metrics.full_384d.retrieval_p1_en_to, L = cloud.langs, by = evalData.by_explained_language;
    const gap = k => L.map(l => `${l.name} ${by[l.code][k].gap.toFixed(2)}`).join(', ');
    const four = ['occlusion', 'integrated_gradients', 'attention_rollout', 'pair_decomposition'];
    const lowest = four.map(k => L.reduce((p, l) => (by[l.code][k].gap < by[p.code][k].gap ? l : p), L[0]).name);
    const worst = L.map(l => l.name).sort((x, y) => lowest.filter(n => n === y).length - lowest.filter(n => n === x).length)[0];
    const oc = L.map(l => by[l.code].occlusion.gap), others = L.filter(l => l.name !== worst).map(l => by[l.code].occlusion.gap);
    document.querySelector('#fair-note').textContent = `The model is not equally good in every language, and neither are its explanations. Finding a sentence’s translation works almost equally well in all four languages, but the explanations are least faithful for ${worst} (occlusion gap ${by[L.find(l => l.name === worst).code].occlusion.gap.toFixed(2)}, against ${Math.min(...others).toFixed(2)} to ${Math.max(...others).toFixed(2)} for the others). Every benchmark pair has English on one side.`;
  }
});
