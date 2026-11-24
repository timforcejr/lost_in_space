// Title: theme.js, light / dark / system color theme.
// What it does: sets <html data-theme="light|dark"> from the saved choice before the first paint (loaded as a plain script in
// <head>); with no saved choice the attribute is absent and css/style.css follows prefers-color-scheme. After the page is parsed
// it adds a System / Light / Dark switch to the header (buttons with aria-pressed; Tab, Enter and Space work). The choice is kept
// in localStorage (every access is in try/catch: the page works if storage throws) and sent to other open windows of the page
// (BroadcastChannel and the storage event), so a pop-out window follows. It fires a "themechange" event on window; main.js
// listens and redraws the views whose colors it reads from CSS (no new analysis run).
(() => {
  const KEY = 'lis.theme', root = document.documentElement;
  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
  const read = () => { try { const v = localStorage.getItem(KEY); return v === 'light' || v === 'dark' ? v : 'system'; } catch { return 'system'; } };
  const write = v => { try { if (v === 'system') localStorage.removeItem(KEY); else localStorage.setItem(KEY, v); } catch { /* storage unavailable: choice lasts for this page view only */ } };
  const effective = () => (choice === 'system' ? (mq && mq.matches ? 'light' : 'dark') : choice);
  let choice = read();
  const chan = 'BroadcastChannel' in window ? new BroadcastChannel('lost-in-space-theme') : null;

  function apply(v, announce) {
    choice = v;
    if (v === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', v);
    document.querySelectorAll('.theme-switch button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.theme === v)));
    if (announce) window.dispatchEvent(new CustomEvent('themechange', { detail: { choice: v, theme: effective() } }));
  }

  function choose(v) { write(v); apply(v, true); try { chan && chan.postMessage({ theme: v }); } catch { /* window closing */ } }

  apply(choice, false);                           // before first paint (this script is in <head>)
  if (mq) (mq.addEventListener ? mq.addEventListener('change', () => { if (choice === 'system') { window.dispatchEvent(new CustomEvent('themechange', { detail: { choice, theme: effective() } })); } }) : null);
  if (chan) chan.onmessage = ev => { const v = ev.data && ev.data.theme; if (v === 'light' || v === 'dark' || v === 'system') apply(v, true); };
  window.addEventListener('storage', ev => { if (ev.key !== KEY && ev.key !== null) return; try { localStorage.getItem(KEY); } catch { return; } apply(read(), true); });   // unreadable storage: keep the current choice

  function build() {
    const host = document.querySelector('.top');
    if (!host || document.querySelector('.theme-switch')) return;
    const g = document.createElement('div'); g.className = 'theme-switch'; g.setAttribute('role', 'group'); g.setAttribute('aria-label', 'Color theme');
    for (const [v, label] of [['system', 'System'], ['light', 'Light'], ['dark', 'Dark']]) {
      const b = document.createElement('button'); b.type = 'button'; b.dataset.theme = v; b.textContent = label;
      b.setAttribute('aria-pressed', String(v === choice)); b.setAttribute('aria-label', v === 'system' ? 'Color theme: follow the system' : `Color theme: ${label}`);
      b.addEventListener('click', () => choose(v)); g.append(b);
    }
    const anchor = document.querySelector('#btn-popout'); if (anchor) anchor.after(g); else host.append(g);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
