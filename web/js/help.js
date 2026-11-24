// Keep the page clean. Every <p class="note"> is hidden and its text moves behind a small "?" button on the heading
// above it (hover, keyboard focus, or click to pin). Any element with a title or data-tip also gets the same styled tip.
// main.js is not touched: it keeps writing into the (hidden) note elements, and this file reads them when the tip opens.
// Without JavaScript nothing is hidden, so the notes stay visible.
(() => {
  const root = document.documentElement;
  const pop = document.createElement('div');
  pop.id = 'help-pop'; pop.setAttribute('role', 'tooltip'); pop.hidden = true;
  document.body.append(pop);

  let owner = null, pinned = false, closeTimer = 0;

  // the heading a note belongs to: the nearest h2/h3 above it (inside .matrix-head the h2 is nested)
  const anchorFor = note => {
    for (let el = note.previousElementSibling; el; el = el.previousElementSibling) {
      if (el.matches('h2, h3')) return el;
      if (el.matches('.matrix-head')) return el.querySelector('h2');
    }
    return null;
  };

  const place = btn => {
    const r = btn.getBoundingClientRect(), vw = innerWidth, vh = innerHeight;
    pop.style.left = '0px'; pop.style.top = '0px';
    const w = pop.offsetWidth, h = pop.offsetHeight;
    const left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), vw - w - 8);
    const below = r.bottom + 8 + h <= vh - 8;
    pop.style.left = `${left}px`;
    pop.style.top = `${below ? r.bottom + 8 : Math.max(8, r.top - 8 - h)}px`;
  };

  const open = (el, texts, pin) => {
    clearTimeout(closeTimer);
    texts = texts.map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;
    pop.replaceChildren(...texts.map(t => Object.assign(document.createElement('p'), { textContent: t })));
    pop.hidden = false; owner = el; pinned = !!pin;
    el.setAttribute('aria-describedby', 'help-pop');
    place(el);
  };
  const close = () => {
    if (owner) owner.removeAttribute('aria-describedby');
    pop.hidden = true; owner = null; pinned = false;
    if (btnOpen) { btnOpen.setAttribute('aria-expanded', 'false'); btnOpen = null; }
  };
  let btnOpen = null;
  const closeSoon = () => { clearTimeout(closeTimer); closeTimer = setTimeout(() => { if (!pinned) close(); }, 120); };

  // ---- "?" buttons for notes (also for notes added or replaced later: one document-level observer)
  const groups = new Map();                 // heading -> {b, notes:Set}
  const headingOf = n => anchorFor(n) || (n.closest('.card, section, article') || document).querySelector('h2, h3');
  const wire = n => {
    if (n.classList.contains('help-wired')) return;
    const h = headingOf(n); if (!h) return;
    let g = groups.get(h);
    if (!g || !h.contains(g.b)) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'help-btn'; b.textContent = '?';
      b.setAttribute('aria-label', `About: ${h.textContent.replace(/\s+/g, ' ').trim()}`);
      b.setAttribute('aria-expanded', 'false');
      g = { b, notes: new Set() }; groups.set(h, g);
      const texts = () => [...g.notes].filter(x => x.isConnected).map(x => x.textContent);
      g.sync = () => { b.hidden = !texts().some(t => t.trim()); };
      b.addEventListener('mouseenter', () => { if (!pinned) open(b, texts()); });
      b.addEventListener('mouseleave', closeSoon);
      b.addEventListener('focus', () => { if (!pinned) open(b, texts()); });
      b.addEventListener('blur', closeSoon);
      b.addEventListener('click', e => {
        e.preventDefault();
        if (pinned && owner === b) { close(); return; }
        open(b, texts(), true); btnOpen = b; b.setAttribute('aria-expanded', 'true');
      });
      h.append(' ', b);
    }
    g.notes.add(n); n.classList.add('help-wired');
    new MutationObserver(g.sync).observe(n, { childList: true, characterData: true, subtree: true });
    g.sync();
  };
  const wireAll = node => {
    if (node.nodeType !== 1) return;
    if (node.matches('.note')) wire(node);
    node.querySelectorAll('.note').forEach(wire);
  };
  wireAll(document.body);
  new MutationObserver(ms => {
    ms.forEach(m => m.addedNodes.forEach(wireAll));
    groups.forEach(g => g.sync());          // a removed (re-created) note changes the live text
  }).observe(document.body, { childList: true, subtree: true });
  root.classList.add('help-on');   // hides only .note.help-wired; any note not wired stays visible

  // ---- plain tips: [data-tip], and title attributes (moved to data-tip so the browser's slow native tip does not double up)
  const tipTarget = e => e.target.closest && e.target.closest('[data-tip], [title]');
  document.addEventListener('mouseover', e => {
    const t = tipTarget(e); if (!t || t.classList.contains('help-btn')) return;
    if (t.hasAttribute('title') && !t.dataset.tip) { t.dataset.tip = t.getAttribute('title'); t.removeAttribute('title'); }
    if (!pinned) open(t, [t.dataset.tip || '']);
  });
  document.addEventListener('mouseout', e => { const t = tipTarget(e); if (t && !t.classList.contains('help-btn')) closeSoon(); });
  document.addEventListener('focusin', e => { const t = tipTarget(e); if (t && t.dataset.tip && !pinned) open(t, [t.dataset.tip]); });
  document.addEventListener('focusout', e => { const t = tipTarget(e); if (t && t.dataset.tip) closeSoon(); });

  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  document.addEventListener('click', e => { if (pinned && !e.target.closest('.help-btn')) close(); });
  addEventListener('scroll', () => { if (owner && !pop.hidden) place(owner); }, { passive: true });
  addEventListener('resize', () => { if (owner && !pop.hidden) place(owner); });
})();
