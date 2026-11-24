/*
Build the 24 x 36 in portrait "Lost in Space" poster v3 (explain-first, one chart) as an editable .pptx, then render
poster/LostInSpace_poster_v3.pdf (the print PDF) with LibreOffice. Every number is read from eval/*.json or poster/fig/*.json and
logged with its source key to poster/out/v3_numbers.json; nothing is hand-typed. Palette: the app's dark theme with a header band.
Verification: python3 -I tests/poster_check.py poster/LostInSpace_poster_v3.pptx poster/LostInSpace_poster_v3.pdf
Run (from the repository root): node poster/build_poster_v3.js
     (needs poster/node_modules: cd poster && npm ci; and LibreOffice `soffice` on PATH for the PDF;
      set POSTER_NO_PDF=1 to skip the PDF)
*/
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const pptxgen = require('pptxgenjs');

const ROOT = path.join(__dirname, '..');
const J = p => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));
const faith = J('eval/faithfulness.json'), e5 = J('eval/faithfulness_e5.json');
const order = J('eval/order.json'), place = J('eval/placement.json'), tsne = J('eval/tsne.json'), fid = J('eval/map_fidelity.json');
const pair = J('poster/fig/order_pair.json');
const sanity = J('eval/sanity.json');   // the dog / man pair, captured from the running page
const dim = Number((Object.keys(fid.spaces).find(k => /^full_\d+d$/.test(k)) || '').replace(/\D/g, ''));   // embedding size, from the key name full_384d
const FIG = f => path.join(__dirname, 'fig', f);
const OUT_PPTX = path.join(__dirname, 'LostInSpace_poster_v3.pptx');
const OUT_PRINT = path.join(__dirname, 'LostInSpace_poster_v3.pdf');
const OUT_NUMS = path.join(__dirname, 'out', 'v3_numbers.json');
const LIVE_URL = 'https://timforcejr.github.io/lost_in_space/';
const SHORT_URL = 'timforcejr.github.io/lost_in_space';

// ---- every printed number goes through N(): it is formatted here and logged with its source key
const NUMS = [];
const N = (value, text, source) => { if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('bad number from ' + source);
  NUMS.push({ text, value, source }); return text; };
const f2 = v => v.toFixed(2).replace(/^-/, '−');

// ---- palette (the app's dark theme) and type
const K = { bg: 'FFFFFF', band: '001A57', card: 'FFFFFF', card2: 'DFEEEB', line: 'DDD8CC', text: '152C4A', text2: '33465F', muted: '56657B',
  navy: '012169', blue: '1F7A76', teal: '1F7A76', a: 'D4563D', b: '1F7A76', base: '7B8798', green: '1F7A76', white: 'FFFFFF',
  tealPale: 'DFEEEB', coralPale: 'FAE6DF', mint: '9FE3D6', brick: 'B0391F' };   // brick: print-safe dark coral (2026-11-18)   // Project 1 poster palette
const FONT = 'Arial';
const PW = 24, PH = 36, M = 0.8, W = PW - 2 * M;

const pres = new pptxgen();
pres.defineLayout({ name: 'POSTER_24x36', width: PW, height: PH });
pres.layout = 'POSTER_24x36';
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = 'Lost in Space poster v3';
pres.subject = 'Duke Explainable AI, Project 2';
pres.author = 'Tim Force';
const s = pres.addSlide();
s.background = { color: K.bg };

const text = (t, o) => s.addText(t, { fontFace: FONT, color: K.text, margin: 0, valign: 'top', isTextBox: true, ...o });
const rect = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.16, fill: { color: K.card }, line: { color: K.line, width: 1.5 }, ...o });
const HEAD = 0.55;
// layout knobs (2026-11-17: room for the Why cards). Floors per Tim's brief: QR square >= 3.8 in, call >= 44 pt, URL >= 36 pt,
// sub-line >= 22 pt, How-it-works boxes >= 1.75 in, section gaps tightened by at most 0.1 in.
// 2026-11-18 (literature section + QR band at the bottom): GAP 0.2 -> 0.15, HEAD 0.6 -> 0.55, Why cards 3.5 -> 2.75 in (shorter texts).
// 2026-11-19d (layout v5): the QR moves into the header's right block, so its square and call/URL/sub sizes come down to fit the header.
const QR_S = 3.0, QR_CALL = 30, QR_URL = 28, QR_SUB = 20, STEP_H = 1.9, WHY_H = 2.75, GAP = 0.15;
// 2026-11-19p: the Why row shrinks 3.25 -> WHY_H2; the d'Ascoli diagram takes DIAG_GROW back (0.45 -> 0.5 in boxes, 0.20 -> 0.25 in gaps); the rest
// (ROW_SLACK) is split between the Box 1/2 and Box 2/3 gaps, so Box 3 and everything below it stay where they were.
const WHY_H2 = 2.2, DIAG_GROW = 0.55, ROW_SLACK = (3.25 - WHY_H2) - DIAG_GROW;
const TK_HEAD = 36, TK_BODY = 26;   // Key takeaways sizes in the bottom band (2026-11-19d)
const heading = (t, y, name) => text(t, { x: M, y, w: W, h: 0.55, fontSize: 44, bold: true, color: K.navy, align: 'center', valign: 'middle', objectName: name });

// ================================================================= header band (2026-11-19d layout v5, Tim: "Expand the header to absorb the title in the
// center, keep Duke on the left and add my name and MSECE below it, and then put the QR and text on the top right"; "just Lost in Space in the
// center"). Three columns on the navy band: left Duke ECE logo + byline; centre title + subtitle; right call text + QR tile, URL under both.
// Text on the band is named 'Hdr ...' (tests/poster_check.py checks its contrast against the band).
const HP = 0.3, LOGO_W = 5.6, QR_TXT_W = 4.2, QR_GAP = 0.25, URL_H = 0.55;
const qrTileX = PW - M - (QR_S + 0.1), qrY = HP + 0.05, rightX = qrTileX - QR_GAP - QR_TXT_W;
const HDR = HP + QR_S + 0.1 + 0.08 + URL_H + 0.22;
s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: PW, h: HDR, fill: { color: K.band }, line: { color: K.band, width: 0 }, objectName: 'Header band' });
{
  // left: logo, byline directly below; the pair is centred vertically in the band
  const logoH = LOGO_W * 339 / 2560, byH = 1.1, lh = logoH + 0.3 + byH, ly = (HDR - lh) / 2;
  s.addImage({ path: FIG('ece_logo_white.png'), x: M - 0.05, y: ly, w: LOGO_W, h: logoH,
    altText: 'Duke Pierre R. Lamond Department of Electrical & Computer Engineering logo', objectName: 'Duke ECE logo' });
  text([{ text: 'Tim Force, MSECE', options: { breakLine: true } }, { text: 'Machine Learning Track' }],
    { x: M, y: ly + logoH + 0.3, w: LOGO_W, h: byH, fontSize: 30, bold: true, color: K.white, align: 'left', valign: 'top', objectName: 'Hdr byline' });
  // centre: 'Lost in Space' (largest text on the sheet) and Tim's tagline on two lines, between the left and right blocks
  const cx0 = M + LOGO_W + 0.3, cw0 = rightX - 0.25 - cx0, th = 1.3 + 0.12 + 1.0, ty = (HDR - th) / 2;
  text('Lost in Space', { x: cx0, y: ty, w: cw0, h: 1.3, fontSize: 88, bold: true, color: K.a, align: 'center', valign: 'middle', objectName: 'Hdr title' });
  text([{ text: 'A visualization of what a model has learned', options: { breakLine: true } }, { text: 'to encode based on the “feeling” of a word.' }],
    { x: cx0, y: ty + 1.42, w: cw0, h: 1.0, fontSize: 30, color: K.white, align: 'center', valign: 'top', objectName: 'Hdr subline' });
  // right: QR tile on the margin, the call and Tim's three scan lines to its left, the URL under both
  s.addShape(pres.ShapeType.rect, { x: qrTileX, y: qrY - 0.05, w: QR_S + 0.1, h: QR_S + 0.1, fill: { color: K.white }, line: { color: K.white, width: 0 }, objectName: 'QR tile' });
  s.addImage({ path: FIG('qr_live.png'), x: qrTileX + 0.05, y: qrY, w: QR_S, h: QR_S, altText: 'QR code: ' + LIVE_URL, objectName: 'QR live' });
  const callH = 1.05, subH = 1.45, tb = callH + 0.1 + subH, t0 = qrY + (QR_S - tb) / 2;
  text([{ text: 'Type two sentences,', options: { breakLine: true } }, { text: 'see why they match' }],
    { x: rightX, y: t0, w: QR_TXT_W, h: callH, fontSize: QR_CALL, bold: true, color: K.mint, valign: 'middle', objectName: 'Hdr QR call' });
  // Tim's wording, typed into the pptx 2026-11-18 (three lines; the third wraps in this column)
  text([{ text: 'Scan the QR code to try it.', options: { breakLine: true } }, { text: 'The model runs on your device.', options: { breakLine: true } },   // 2026-11-19n review nit
    { text: 'Nothing to install; nothing you type is sent anywhere.' }],
  { x: rightX, y: t0 + callH + 0.1, w: QR_TXT_W, h: subH, fontSize: QR_SUB, color: K.white, valign: 'top', objectName: 'Hdr QR sub' });
  // no hyperlink: LibreOffice underlines links and the line hid the underscores
  text(SHORT_URL, { x: rightX, y: qrY + QR_S + 0.05 + 0.08, w: PW - M - rightX, h: URL_H, fontSize: QR_URL, bold: true, color: K.white, align: 'center', valign: 'middle', objectName: 'Hdr QR URL' });
  console.log(`header ${HDR.toFixed(2)} in; title column ${cx0.toFixed(2)}-${(cx0 + cw0).toFixed(2)} in (${cw0.toFixed(2)} in); QR ${QR_S} in (tile ${(QR_S + 0.1).toFixed(2)} in) at x ${qrTileX.toFixed(2)}`);
}

// ================================================================= 1 sentence line under the banner (Tim, 2026-11-19d: "Dog bit man can be below the banner")
// Periods inside the quotes, as typed on the page (Tim, 2026-11-18): a visitor typing these sees the same 0.98.
let y = HDR + 0.12;
text([{ text: '“The dog bit the man.”  vs  “The man bit the dog.”  scored ' },
  { text: N(pair.sim, f2(pair.sim), 'poster/fig/order_pair.json:sim'), options: { color: K.a } }, { text: ' alike' }],
  { x: M, y, w: W, h: 0.55, fontSize: 28, bold: true, color: K.navy, align: 'center', valign: 'middle', objectName: 'Headline 2' });
y += 0.55 + 0.15;

const BUL = { bullet: { indent: 24 } };   // pptxgenjs makes each bulleted run its own paragraph: bullets only on single-run points
const bl = (t, last) => ({ text: t, options: { ...BUL, breakLine: !last } });
const pctT = v => (100 * v).toFixed(1).replace(/\.0$/, '') + '%';   // 5.0 -> '5%', 6.5 -> '6.5%' (Tim's map-card wording)
const DEF = 'definition: ';   // numbers that are true by construction (chance level, the map's dimension), not measurements
const three = N(3, '3', DEF + 'the map has three dimensions (build/build_tsne.py, the page heads output 3 coordinates)');

// ================================================================= print-safe structure (2026-11-18, per Tim: "the box outlines don't show up well on my printer.
// Maybe one box for each major section"). Three big boxes, each a solid Duke-navy outline of BIG_W pt, no fill (assume any fill vanishes on
// paper): Box 1 intro, Box 2 'What each tool shows', Box 3 literature. The footer line is plain text under a thin dark rule.
// Content stays at x = M; the box outline sits BOX_IN outside it (outer edge >= 0.5 in from the trim).
const BIG_W = 4, RULE_W = 2, OUT_W = 2.25, ARROW_W = 4, BOX_IN = 0.25, BOX_GAP = 0.12;   // 2026-11-19b: BOX_GAP 0.2 -> 0.12
const bigBox = (y0, y1, name) => s.addShape(pres.ShapeType.roundRect, { x: M - BOX_IN, y: y0, w: W + 2 * BOX_IN, h: y1 - y0, rectRadius: 0.2,
  fill: { type: 'none' }, line: { color: K.band, width: BIG_W }, objectName: name });
const hRule = (x, yy, w, name, color = K.band) => s.addShape(pres.ShapeType.line, { x, y: yy, w, h: 0, line: { color, width: RULE_W }, objectName: name });
const vRule = (x, yy, h, name) => s.addShape(pres.ShapeType.line, { x, y: yy, w: 0, h, line: { color: K.band, width: RULE_W }, objectName: name });
if (M - BOX_IN - BIG_W / 144 < 0.5) throw new Error('box outline closer than 0.5 in to the trim');

// ================================================================= BOX 1: intro
y += 0.0;   // 2026-11-19b: 0.05 -> 0 (the rule gap above already clears it)
const box1Top = y;
y += 0.12;   // 2026-11-19k: 0.18 -> 0.12, the 0.06 in goes to the Builds on / New here row (diagrams stay put)
// ---- builds on / new here (boundary statement). The two columns read as the 59-word boundary statement verbatim, split at 'New here:'
// (same words in README.md, docs/TALK.md section 2 and the web Credits). Keep "Builds on" / "New here" (tests/poster_check.py).
// 2026-11-19k (Tim: "let's make New here first bullet point shorter ... that way we can add a third bullet point"): the poster's New here
// bullets now differ from the README / TALK / web wording (three short capitalized bullets; the third is the default).
const STMT_A = ['Lost in Space builds on the NLP project, a group class project with Ananya Singh and Cherry Lian that asked whether you can visualize how a model understands the feeling of text across languages.'];
const STMT_B = ['Runs in your browser; nothing leaves your device.',
  'XAI tools: attribution, maps, layer lens, faithfulness and sanity tests.',
  'Visualization to build understanding of how an AI learns meaning.'];   // 2026-11-19l (Tim)
{
  const h = 1.81, cw = (W - 0.6) / 2;   // 2026-11-19b: 1.85 -> 1.75; 2026-11-19k: 1.81
  text([{ text: 'Builds on', options: { bold: true, color: K.teal, fontSize: 28, breakLine: true } }, { text: STMT_A[0] }],
    { x: M + 0.05, y, w: cw - 0.05, h, fontSize: 24, color: K.text2, valign: 'top', paraSpaceAfter: 4, objectName: 'Builds on text' });
  vRule(M + cw + 0.3, y + 0.05, h - 0.1, 'Builds on rule');
  text([{ text: 'New here:', options: { bold: true, color: K.brick, fontSize: 28, breakLine: true } }, bl(STMT_B[0]), bl(STMT_B[1]), bl(STMT_B[2], true)],
    { x: M + cw + 0.6, y, w: cw, h, fontSize: 24, color: K.text, valign: 'top', paraSpaceAfter: 2, objectName: 'New here text' });
  y += h + 0.15;
}

// ---- three top-down block diagrams side by side (2026-11-18, per Tim): (1) d'Ascoli et al. 2025, Fig. 1B, redrawn and simplified
// (M/EEG word window -> subject layer, one per participant -> CNN -> transformer at
// sentence level -> predicted word embedding -> contrastive loss and cosine retrieval against T5 word embeddings, a fixed target; the paper
// never says 'frozen'); (2) the NLP project's design idea (dashed brick: not something the page does); (3) what the page runs (solid navy).
// Native shapes, no image of the original figure. Credit line (CC BY 4.0) printed under the diagrams.
{
  if (Object.keys(order.scramble.languages).sort().join() !== 'en,es,hi,zh') throw new Error('diagram names English, Spanish, Chinese, Hindi');
  if (Object.keys(place.heads).sort().join() !== 'ae,fc,pca') throw new Error('compression block lists PCA, linear, autoencoder');
  const names = ['English', 'Spanish', 'Chinese', 'Hindi'];
  const FS = 22, cg = 0.4, cwD = [7.2, 6.9, 7.5];
  const cxD = [M, M + cwD[0] + cg, M + cwD[0] + cwD[1] + 2 * cg];
  const headH = 0.6, H = 4.75 + DIAG_GROW;   // 2026-11-19p: + the height the Why row gave back
  const vArrow = (x, y1, y2, o = {}) => s.addShape(pres.ShapeType.line, { x, y: y1, w: 0, h: y2 - y1,
    line: { color: o.color || K.band, width: ARROW_W, dashType: o.dash || 'solid', endArrowType: 'triangle' } });
  const box = (x, yy, w, h, fill, line, runs, o = {}) => {
    rect(x, yy, w, h, { rectRadius: 0.1, fill: { color: fill }, line: { color: line, width: OUT_W, dashType: o.dash || 'solid' }, objectName: o.name });
    text(runs, { x: x + 0.08, y: yy, w: w - 0.16, h, fontSize: FS, align: 'center', valign: o.valign || 'middle', color: o.color || K.text, objectName: o.name && o.name + ' text' });
  };
  const B_ = t2 => ({ text: t2, options: { bold: true } });
  const two = (a, b2, o = {}) => [{ text: a, options: { bold: true, breakLine: true } }, { text: b2, options: o }];
  // tiers: heights; returns the top y of each tier, spread evenly over H (equal arrow gaps), so the three columns end level
  const spread = (y0, hs) => { const g = (H - hs.reduce((p, q) => p + q, 0)) / (hs.length - 1); const ys = []; let yy = y0;
    hs.forEach(h => { ys.push(yy); yy += h + g; }); if (g < 0.2) throw new Error('diagram arrows too short: ' + g.toFixed(2)); return ys; };
  const heads = [['d’Ascoli et al. 2025', K.card, K.muted, 'solid'], ['The NLP design idea', K.coralPale, K.brick, 'dash'], ['What this page runs', K.tealPale, K.band, 'solid']];
  heads.forEach(([t2, fill, line, dash], i) => box(cxD[i], y, cwD[i], headH, fill, line, [B_(t2)], { dash, name: `Diagram ${i + 1} label`, color: K.navy }));
  const top = y + headH + 0.2;

  // (1) d'Ascoli et al. 2025, top-down (neutral grey outlines). 2026-11-19f (Tim): both paths of Fig. 1B as two parallel columns: brain path
  // (left) M/EEG word window -> subject layer -> CNN -> transformer -> predicted word embedding; word path (right) vocabulary word -> T5 language
  // model (fixed) -> target word embeddings. 2026-11-19j (Tim: "They should merge somewhere to show that they are common"): both arrows
  // converge into one 'Same embedding space' box (Y-hat and Y both live in R^d, d fixed by T5), then one arrow to the contrastive loss
  // (cosine similarity in that space). Same height budget: boxes 0.5 -> 0.45 in, the loss on one line.
  {
    const x = cxD[0], lW = 4.15, rg = 0.3, rX = x + lW + rg, rW = cwD[0] - lW - rg;
    const hs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.5], ys = spread(top, hs);   // 2026-11-19p: 0.45 -> 0.5 (arrow gaps 0.20 -> 0.25)
    const labels = [[{ text: 'M/EEG word window' }], [{ text: 'Subject layer per person' }], [{ text: 'CNN' }], [{ text: 'Transformer (sentence level)' }],
      [{ text: 'Predicted word embedding' }]];
    labels.forEach((runs, i) => {
      box(x, ys[i], lW, hs[i], K.card, K.muted, runs, { name: `dAscoli brain ${i + 1}` });
      if (i > 0) vArrow(x + lW / 2, ys[i - 1] + hs[i - 1], ys[i], { color: K.muted });
    });
    // word path: first box level with the first brain box, last box bottom-aligned with 'Predicted word embedding', the middle one centred
    const rh = [0.5, 0.8, 0.8], rEnd = ys[4] + hs[4], rg2 = (rEnd - ys[0] - rh[0] - rh[1] - rh[2]) / 2;
    const rys = [ys[0], ys[0] + rh[0] + rg2, rEnd - rh[2]];
    const wl = [[{ text: 'Vocabulary word' }], [{ text: 'T5 language', options: { breakLine: true } }, { text: 'model (fixed)' }],
      [{ text: 'Target word', options: { breakLine: true } }, { text: 'embeddings' }]];
    wl.forEach((runs, i) => {
      box(rX, rys[i], rW, rh[i], K.card, K.muted, runs, { name: `dAscoli word ${i + 1}` });
      if (i > 0) vArrow(rX + rW / 2, rys[i - 1] + rh[i - 1], rys[i], { color: K.muted });
    });
    // the two paths converge into one shared space, then one arrow to the loss
    box(x, ys[5], cwD[0], hs[5], K.card, K.muted, two('Same embedding space', 'brain output trained to land on T5 targets'), { name: 'dAscoli shared space' });
    vArrow(x + lW / 2, rEnd, ys[5], { color: K.muted });
    vArrow(rX + rW / 2, rEnd, ys[5], { color: K.muted });
    box(x, ys[6], cwD[0], hs[6], K.card, K.muted, [{ text: 'Contrastive loss, ', options: { bold: true } }, { text: 'cosine retrieval' }], { name: 'dAscoli loss' });
    vArrow(x + cwD[0] / 2, ys[5] + hs[5], ys[6], { color: K.muted });
  }
  // (2) the NLP design idea (per-language encoders into one common space), dashed brick throughout
  {
    const x = cxD[1], w = cwD[1], lg = 0.15, lw = (w - 3 * lg) / 4;
    const hs = [0.5, 0.8, 0.5, 0.5, 0.85], ys = spread(top, hs);
    const D = { color: K.brick, dash: 'dash' };
    names.forEach((n, i) => {
      const lx = x + i * (lw + lg);
      box(lx, ys[0], lw, hs[0], K.card, K.brick, [B_(n)], { dash: 'dash', name: `Lane A ${n}` });
      vArrow(lx + lw / 2, ys[0] + hs[0], ys[1], D);
      box(lx, ys[1], lw, hs[1], K.card, K.brick, [{ text: 'Own', options: { breakLine: true } }, { text: 'encoder' }], { dash: 'dash', name: `Own encoder ${n}` });
      vArrow(lx + lw / 2, ys[1] + hs[1], ys[2], D);
    });
    box(x, ys[2], w, hs[2], K.card, K.brick, [B_('Compress to a map')], { dash: 'dash', name: 'Compression A' });
    vArrow(x + w / 2, ys[2] + hs[2], ys[3], D);
    box(x, ys[3], w, hs[3], K.card, K.brick, [B_('One common embedding space')], { dash: 'dash', name: 'Common space' });
    vArrow(x + w / 2, ys[3] + hs[3], ys[4], D);
    box(x, ys[4], w, hs[4], K.coralPale, K.brick, two('Evaluate', 'translations found, language hidden'), { dash: 'dash', name: 'Evaluate' });
  }
  // (3) what the page runs: one shared tokenizer, one frozen multilingual MiniLM, one compression block, one 3-D map (solid navy)
  {
    const x = cxD[2], w = cwD[2], lg = 0.15, lw = (w - 3 * lg) / 4;
    const hs = [0.5, 0.5, 1.36, 0.78, 0.78], ys = spread(top, hs);
    names.forEach((n, i) => {
      const lx = x + i * (lw + lg);
      box(lx, ys[0], lw, hs[0], K.tealPale, K.band, [B_(n)], { name: `Lane B ${n}` });
      vArrow(lx + lw / 2, ys[0] + hs[0], ys[1]);
    });
    box(x, ys[1], w, hs[1], K.band, K.band, [B_('One shared tokenizer')], { color: K.white, name: 'Tokenizer' });
    vArrow(x + w / 2, ys[1] + hs[1], ys[2]);
    box(x, ys[2], w, hs[2], K.band, K.band, [], { name: 'Frozen MiniLM' });
    text([{ text: 'One frozen multilingual MiniLM', options: { bold: true, breakLine: true } },
      { text: `${N(order.layers - 1, String(order.layers - 1), 'eval/order.json:layers (13 hidden states = embedding output + 12 layers)')} layers · ${N(dim, dim + '-D', 'eval/map_fidelity.json:spaces.full_384d (key name)')} · int8` }],
      { x: x + 0.1, y: ys[2] + 0.04, w: w - 0.2, h: 0.76, fontSize: FS, color: K.white, align: 'center', valign: 'top', objectName: 'Frozen MiniLM text' });
    box(x + 0.25, ys[2] + hs[2] - 0.48, w - 0.5, 0.42, K.band, K.mint,   // 2026-11-19n: 0.04 in more clear of the MiniLM text
      [{ text: 'Option: per-language centering' }], { color: K.mint, dash: 'dash', name: 'Centering' });
    vArrow(x + w / 2, ys[2] + hs[2], ys[3]);
    box(x, ys[3], w, hs[3], K.card, K.band, two('Compress to a map', 'PCA · linear · autoencoder · t-SNE'), { name: 'Compression B' });
    vArrow(x + w / 2, ys[3] + hs[3], ys[4]);
    box(x, ys[4], w, hs[4], K.coralPale, K.band, two('One three-dimensional map', `Checks: trust@${N(place.k, String(place.k), 'eval/placement.json:k')}, language probe`), { name: 'Map and checks' });
  }
  y = top + H + 0.1;
  // Credit for the left diagram (CC BY 4.0 requires credit, licence link and an indication of changes). Bibliographic identifiers are
  // checked against REFERENCES.md entry 68; figure number and licence from the researcher memo (dascoli_architecture_2026-11-18.md).
  const refs = fs.readFileSync(path.join(ROOT, 'REFERENCES.md'), 'utf8');
  const REF = 'REFERENCES.md 68 (d’Ascoli et al. 2025): ';
  ['16, article 10521', '10.1038/s41467-025-65499-0'].forEach(t2 => { if (!refs.includes(t2)) throw new Error('credit line: REFERENCES.md lacks ' + t2); });
  const cite = (v, t2, what) => N(v, t2, REF + what);
  text([{ text: 'Left diagram adapted from d’Ascoli et al., Nat. Commun. ' + cite(16, '16', 'volume') + ', ' + cite(10521, '10521', 'article number') +
      ' (2025), Fig. ' + cite(1, '1', 'figure number, per the architecture memo') + 'B, doi:' + cite(10.1038, '10.1038', 'DOI prefix') + '/s41467-' +
      cite(25, '025', 'DOI') + '-' + cite(65499, '65499', 'DOI') + '-' + cite(0, '0', 'DOI') + '; CC BY ' + cite(4.0, '4.0', 'licence version, per the architecture memo') +
      ' (creativecommons.org/licenses/by/' + cite(4.0, '4.0', 'licence URL') + '). Redrawn and simplified.' }],
    { x: M, y, w: W, h: 0.3, fontSize: 16, color: K.muted, valign: 'middle', objectName: 'References credit dAscoli' });
  y += 0.3 + 0.1;
}

// ---- why / what you can see / what comes next (Tim's final wording, 2026-11-18 line-by-line review). Borderless row inside Box 1 under a
// thin navy rule, columns split by thin navy rules. No 'teaching aid' and no native-speaker line (Tim).
// 2026-11-19m (Tim: "grad cam really looks like it might be more of a distraction than it is worth. I think we should remove it everywhere."):
// the poster shows four methods; Grad-CAM stays in eval/*.json (not edited) but is not printed.
const four = ['occlusion', 'integrated_gradients', 'attention_rollout', 'pair_decomposition'];
if (!four.every(key => faith.all.pair_decomposition.gap <= faith.all[key].gap)) throw new Error('pair split ranks last of the four');
{
  hRule(M, y, W, 'Why rule');
  y += 0.12;
  const why = [
    // 2026-11-19p (Tim: "Can we condense and make those three in the middle more symmetrical?"): heading + four one-line bullets each,
    // no intro line, equal column widths (bullets measured with Arimo at 22 pt: all <= 6.6 in of 6.8 in).
    ['Why I built it', null, ['A word2vec lecture started this rabbit hole.', 'How do models learn across languages?',
      'What does it say about how we communicate?', 'Is it learning differences? Can we see them?']],
    ['What you can see', null, [`A ${three}D map of how the model sees meaning.`,
      'Close points: meanings the model treats alike.', 'Gaps between translations: meaning may differ.', 'Highlighted words: which ones drove the score.']],
    ['What comes next', null, ['Migrate to the NLP model design.', 'Verify with known bad translations.', 'Verify findings with native speakers.', 'Add additional languages.']],
  ];
  const gap = 0.35, ws3 = [0, 0, 0].map(() => (W - 2 * gap) / 3), chh = WHY_H2;   // 2026-11-19p: equal columns, chh 3.25 -> WHY_H2
  why.forEach(([head, lead, lines], i) => {
    const cw = ws3[i], x = M + ws3.slice(0, i).reduce((p, q) => p + q + gap, 0);
    if (i > 0) vRule(x - gap / 2, y + 0.05, chh - 0.1, `Why rule ${i}`);
    const runs = [{ text: head, options: { bold: true, color: K.teal, fontSize: 30, breakLine: true } }];
    if (lead) runs.push({ text: lead, options: { color: K.text, breakLine: true } });
    text([...runs, ...lines.map((t, j) => bl(t, j === lines.length - 1))],
      { x: x + 0.05, y, w: cw - 0.1, h: chh, fontSize: 22, color: K.text2, valign: 'top', paraSpaceAfter: 3, objectName: `Why ${i + 1} text` });
  });
  y += chh + 0.1;
}
bigBox(box1Top, y, 'Box 1 intro');
y += BOX_GAP + ROW_SLACK / 2;   // 2026-11-19p: half the leftover height

// ================================================================= BOX 2: what each tool shows (one takeaway per XAI tool on the site)
// 2026-11-18 layout (Tim): 'Which words made it say so?' full height on the right; the score, map and layer-lens sections stacked on the
// left; the divider sits left of centre so the chart column is wider.
const box2Top = y;
y += 0.1;
heading('What each tool shows', y, 'Heading found');
y += HEAD + 0.05;
{
  const gap = 0.45, lw = 10.0, rx = M + lw + gap, rw = W - lw - gap;
  const vg = 0.35, hA = 2.2, hC = 2.2, hD = 2.3, hB = hA + hC + hD + 2 * vg;   // 2026-11-19b: vertical card gap vg 0.45 -> 0.35
  const card = (n, title, x, yy, w, h) => {
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.05, y: yy + 0.14, w: 0.4, h: 0.4, fill: { color: K.teal }, line: { color: K.teal, width: 0 } });
    text(title, { x: x + 0.65, y: yy, w: w - 0.7, h: 0.68, fontSize: 30, bold: true, color: K.navy, valign: 'middle', objectName: `Card ${n} title` });
    return { x: x + 0.05, y: yy + 0.72, w: w - 0.1, h: h - 0.75 };
  };
  const claim = (t, last) => ({ text: t, options: { bold: true, color: K.teal, breakLine: !last } });
  const body = (runs, b, name, o = {}) => text(runs, { x: b.x, y: b.y, w: b.w, h: b.h, fontSize: 22, color: K.text2, paraSpaceAfter: 2, objectName: name, ...o });
  vRule(M + lw + gap / 2, y + 0.05, hB - 0.1, 'Tools rule column');
  hRule(M, y + hA + vg / 2, lw, 'Tools rule A-C');
  hRule(M, y + hA + vg + hC + vg / 2, lw, 'Tools rule C-D');

  // (A) the similarity score. Sources: eval/order.json scramble (RESULTS.md §5, layer 12 median 0.910) and paws.cosine_auc.all[12] (AUC 0.570,
  //     printed as 'ranked right 57% of the time': AUC = chance a random paraphrase outscores a random re-ordered look-alike).
  let b = card('A', 'The similarity score', M, y, lw, hA);
  const L = order.layers - 1, med = order.scramble.cos_band_p10_p50_p90.shuffle[1][L], auc = order.paws.cosine_auc.all[L];
  if (Object.keys(order.scramble.languages).length !== 4) throw new Error('poster text says four languages');
  if (!(auc > 0.5 && auc < 0.65)) throw new Error('similarity card says the score is barely better than a coin flip');
  body([claim('A high score is not the same meaning.'),
    bl(`Scramble every word: median still ${N(med, f2(med), `eval/order.json:scramble.cos_band_p10_p50_p90.shuffle[1][${L}]`)} like the original.`),
    bl(`Paraphrase vs look-alike: ranked right ${N(auc, Math.round(100 * auc) + '%', `eval/order.json:paws.cosine_auc.all[${L}]`)} of the time (coin flip: ${N(50, '50%', DEF + 'a random ranking has AUC 0.5')}).`, true)], b, 'Card A text');   // 2026-11-19n/o (Tim): bullet 2 shortened to one line

  // (C) the map. Neighbors kept of the 10 true nearest: PCA, linear (fc) and autoencoder heads: eval/placement.json heads.<m>.heldout_overlap10_head
  //     (5.0%, 6.5%, 8.0%); t-SNE: eval/tsne.json placement.heldout_overlap10_neighbour_centre (41.9%, placed at the similarity-weighted
  //     center of its nearest reference sentences: no formula; Tim's wording 'ignores distance, places sentence near neighbors').
  b = card('C', `The map from ${N(dim, dim + '-D', 'eval/map_fidelity.json:spaces.full_384d (key name)')} to ${three}-D`, M, y + hA + vg, lw, hC);
  const H = place.heads;
  const hdOf = m => N(H[m].heldout_overlap10_head, pctT(H[m].heldout_overlap10_head), `eval/placement.json:heads.${m}.heldout_overlap10_head`);
  const ts = tsne.placement.heldout_overlap10_neighbour_centre;
  if (!(H.ae.heldout_overlap10_head >= H.fc.heldout_overlap10_head && H.fc.heldout_overlap10_head >= H.pca.heldout_overlap10_head)) throw new Error('card C order PCA < linear < autoencoder');
  if (!(ts > H.ae.heldout_overlap10_head)) throw new Error('card C says t-SNE keeps more than the autoencoder');
  body([{ text: `Of a sentence’s ${N(place.k, String(place.k), 'eval/placement.json:k')} nearest neighbors, the map keeps:`, options: { bold: true, color: K.teal, breakLine: true } },
    bl(`PCA ${hdOf('pca')} · linear ${hdOf('fc')} · autoencoder ${hdOf('ae')}: one formula places any sentence.`),
    bl(`t-SNE ${N(ts, pctT(ts), 'eval/tsne.json:placement.heldout_overlap10_neighbour_centre')}: places a sentence at the center of its nearest neighbors.`, true)], b, 'Card C text');   // 2026-11-19n review fix

  // (D) layer by layer ("Play layers"). Source: eval/map_fidelity.json layer_lens[*].translation_cos / random_cos (RESULTS.md §6).
  //     Tim, 2026-11-18: "We will need to verify this statement when the new model drops."
  b = card('D', 'The layer lens: layer by layer', M, y + hA + vg + hC + vg, lw, hD);
  const LL = fid.layer_lens, l0 = LL[0], peak = LL.reduce((p, q) => (q.translation_cos > p.translation_cos ? q : p)), last = LL[LL.length - 1];
  const at = (l, f) => N(l[f], f2(l[f]), `eval/map_fidelity.json:layer_lens[${l.layer}].${f}`), lay = l => N(l.layer, String(l.layer), `eval/map_fidelity.json:layer_lens[${l.layer}].layer`);
  if (!(last.translation_cos < peak.translation_cos)) throw new Error('card D says translation cosine dips at the end');
  body([claim(`Translations line up by layer ${lay(peak)}; unrelated sentences split only at the end.`),
    bl(`Translations (cosine): ${at(l0, 'translation_cos')} → ${at(peak, 'translation_cos')} (layer ${lay(peak)}) → ${at(last, 'translation_cos')} (layer ${lay(last)}).`),
    bl(`Unrelated: ${at(peak, 'random_cos')} (layer ${lay(peak)}) → ${at(last, 'random_cos')} (layer ${lay(last)}).`, true)], b, 'Card D text');

  // (B) which words made it say so: the only chart, full height on the right. Sources: eval/faithfulness.json all.*.gap, n_explanations,
  //     mask_all; eval/faithfulness_e5.json; eval/sanity.json rank_corr.attention_rollout (Adebayo et al. 2018 sanity check).
  // 'Card B' is an invisible white backing shape: tests/poster_check.py reads the chart's background from it (bar contrast).
  rect(rx, y, rw, hB, { fill: { color: K.card }, line: { type: 'none' }, objectName: 'Card B' });
  b = card('B', 'Which words made it say so?', rx, y, rw, hB);
  const A = faith.all;
  const methods = [['occlusion', 'Occlusion', 'occlusion'], ['integrated_gradients', 'Integrated gradients', 'integrated gradients'],
    ['attention_rollout', 'Attention rollout', 'attention rollout'], ['pair_decomposition', 'Pair split', 'pair split']];   // Grad-CAM removed 2026-11-19m
  if (methods.length !== 4) throw new Error('card B lead says four ways');
  const rows = methods.map(([key, name, short]) => ({ k: key, name, short, gap: A[key].gap })).sort((p, q) => q.gap - p.gap);
  rows.push({ k: 'random_word', name: 'Random words', gap: A.random_word.gap });   // 2026-11-19n: 'Random order' -> 'Random words'
  rows.forEach(r => N(r.gap, f2(r.gap), `eval/faithfulness.json:all.${r.k}.gap`));
  if (rows.slice(0, -1).some(r => r.gap < 0) || !(A.random_word.gap < 0)) throw new Error('chart: only the random-order bar is negative (its label is drawn on the right)');
  const leadH = 0.85;
  body([{ text: 'Four ways to find the words that drove the score: ', options: { bold: true, color: K.navy } },
    { text: rows.filter(r => r.short).map(r => r.short).join(', ') + '.' }], { x: b.x, y: b.y - 0.05, w: b.w, h: leadH }, 'Card B lead', { fontSize: 24 });
  const textH = 1.2, cy = b.y + leadH, chartH = y + hB - textH - cy;
  // Plot area placed by hand so the bars start right at their tool-name labels and the negative label can sit right of the zero line.
  const vMin = -0.02, vMax = Math.ceil((rows[0].gap + 0.1) * 20) / 20, cX = b.x - 0.1, cW = b.w + 0.2, pl = { x: 3.45 / cW, y: 0.02, w: 1 - 3.45 / cW - 0.01, h: 0.96 };
  s.addChart(pres.charts.BAR, [{ name: 'Deletion-test gap', labels: rows.map(r => r.name), values: rows.map(r => +r.gap.toFixed(2)) }],
    { x: cX, y: cy, w: cW, h: chartH, barDir: 'bar', barGapWidthPct: 50,   // 2026-11-19m: 30 -> 50, five rows in the same area layout: pl,
      chartColors: rows.map(r => (r.k === 'random_word' ? K.base : K.blue)),
      catAxisOrientation: 'maxMin', valAxisHidden: true, valAxisLabelFontSize: 16, valAxisMinVal: vMin, valAxisMaxVal: vMax, catAxisLabelPos: 'low', valGridLine: { style: 'none' }, catGridLine: { style: 'none' },
      catAxisLabelColor: K.text, catAxisLabelFontSize: 22, catAxisLabelFontFace: FONT, catAxisLineShow: false,
      showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00;;0.00', dataLabelColor: K.navy, dataLabelFontSize: 24, dataLabelFontBold: true, dataLabelFontFace: FONT,
      showLegend: false, objectName: 'Faithfulness chart' });
  {   // the negative (random order) value label, drawn right of the zero line on the last row (the chart's own label is blanked for negatives)
    const rnd = rows[rows.length - 1], n = rows.length, zx = cX + cW * (pl.x + pl.w * (0 - vMin) / (vMax - vMin));
    const rowY = cy + chartH * (pl.y + pl.h * (n - 0.5) / n);
    text(N(rnd.gap, f2(rnd.gap), 'eval/faithfulness.json:all.random_word.gap'), { x: zx + 0.12, y: rowY - 0.25, w: 1.3, h: 0.5, fontSize: 24, bold: true, color: K.navy, valign: 'middle', objectName: 'Chart label random order' });
  }
  const lead = set => set.occlusion.gap - set.integrated_gradients.gap;
  if (!(lead(faith.mask_all) < lead(A))) throw new Error('mask sentence no longer true');
  const e5Top = Object.keys(e5.all).filter(key => methods.some(m => m[0] === key)).sort((p, q) => e5.all[q].gap - e5.all[p].gap)[0];
  if (e5Top !== 'integrated_gradients') throw new Error('second-model sentence no longer true: top is ' + e5Top);
  const d8 = sanity.depths.indexOf(8), dMax = Math.max(...sanity.depths), roll8 = sanity.rank_corr.attention_rollout[d8];
  if (d8 < 0 || dMax !== order.layers - 1) throw new Error('sanity depths changed');
  // 2026-11-19f wording guards (Tim's text under the chart): 'Reset most of the model's layers' = top 8 of 12 re-initialized (eval/sanity.json 'what');
  // 2026-11-19g: 'On a second model, the order changes' (faithfulness_e5.json reorders the four printed methods); 'Pair split barely beats guessing by
  // word length' (main-model guards below: pair split minus token_count > 0, diff < 0.1); 'Rollout barely changes when layers are reset (0.97)'
  // (top 8 of 12 layers re-initialized). The second-model pair-split guard is kept (true, no longer printed).
  if (!(8 > dMax / 2 && /top-k encoder layers were re-initialised/.test(sanity.what))) throw new Error("'most of the model's layers' no longer true");
  { const fiveK = four;   // 2026-11-19m: the four printed methods
    const ord = d => fiveK.slice().sort((p, q) => d[q].gap - d[p].gap).join();
    const base = ord(faith.all);
    if (ord(e5.all) === base) throw new Error('second model does not change the ranking');
    const e5len = e5.paired_differences.delete.pair_decomposition__minus__token_count;
    if (!(e5len.ci95[1] < 0)) throw new Error("pair split: 'on a second model it doesn't' beat the length-only guess no longer true"); }
  // Pair split claims (Tim's rule 2026-11-18: say only what the data support). eval/faithfulness.json paired_differences.delete:
  // pair split minus word-length ranking +0.063 (95% interval 0.049 to 0.078), minus random order +0.205; eval/faithfulness_e5.json:
  // pair split -0.009, no better than random order (-0.002).
  const pdD = faith.paired_differences.delete;
  const vsLen = pdD.pair_decomposition__minus__token_count, vsRnd = pdD.pair_decomposition__minus__random_word;
  if (!(vsLen.ci95[0] > 0 && vsRnd.ci95[0] > 0)) throw new Error('pair-split note: beats word length and random order on the main model');
  if (!(vsLen.diff < 0.1)) throw new Error('pair-split note: "modestly" needs a small margin over word length');
  if (!(e5.all.pair_decomposition.gap <= e5.all.random_word.gap)) throw new Error('pair-split note: no better than random order on the second model');
  const ty = cy + chartH;
  // 2026-11-19h (Tim: "can you string all of that into one fine print sentence?"): one grey 20 pt paragraph, about 3 lines at 11.85 in
  body([{ text: `Deletion test, ${N(faith.n_explanations, faith.n_explanations.toLocaleString('en-US'), 'eval/faithfulness.json:n_explanations')} explanations (higher = more faithful): no single heat map should be trusted alone. ` +
      'Pair split barely beats guessing by word length, the order changes on a second model, and attention rollout barely changes when ' + N(sanity.depths[d8], String(sanity.depths[d8]), `eval/sanity.json:depths[${d8}] (top-k encoder layers re-initialized)`) + ' of ' + N(order.layers - 1, String(order.layers - 1), 'eval/order.json:layers (13 hidden states = embedding output + 12 layers)') + ' layers are reset (' +
      N(roll8, f2(roll8), `eval/sanity.json:rank_corr.attention_rollout[${d8}]`) + ').', options: { fontSize: 20, color: K.muted } }],
  { x: b.x, y: ty, w: b.w, h: y + hB - ty }, 'Card B text', { fontSize: 24 });
  y += hB + 0.15;
}
bigBox(box2Top, y, 'Box 2 tools');
y += BOX_GAP + ROW_SLACK / 2;   // 2026-11-19p: the other half (Box 3, map tile and takeaways stay put)

// ================================================================= BOX 3: where it aligns with the literature
// Facts only from the verified records: d'Ascoli 2025 (REFERENCES.md 68; the T5 target is fixed, the paper never says frozen); Artetxe, Ruder &
// Yogatama 2020, Heffernan et al. 2022 (LASER3), WECHSEL (REFERENCES.md 71, 72, 70); Reimers & Gurevych 2020 (REFERENCES.md 2; the page's model
// is this recipe) and LaBSE (REFERENCES.md 8); Libovický 2020 ('only moderately language-neutral', abstract re-opened 2026-11-18, REFERENCES.md 35)
// and Thompson 2020 (REFERENCES.md 69). Column heads and cells: Tim's wording, 2026-11-18. No digits other than years.
// 2026-11-19e (layout v6, Tim: "keep where it aligns with literature narrow too and scale the image up to go to the bottom of box 2"):
// the map tile spans the right column from just below the Box 2 outline (level with the Box 3 outline top) down through the Box 3 row
// into the navy band, bottom 0.5 in above the trim, right image edge on the margin, aspect kept, not cropped. Box 3 is the narrow
// left column beside it; Key takeaways and the disclaimers sit in the band below Box 3, left of the tile.
const MAP_PNG = FIG('poster_map_graphic.png'), MAP_BUF = fs.readFileSync(MAP_PNG);
const MAP_AR = MAP_BUF.readUInt32BE(16) / MAP_BUF.readUInt32BE(20);   // PNG IHDR width / height
const mapY = y + 0.05, mapH = PH - 0.5 - OUT_W / 72 - 0.05 - mapY,   // 2026-11-19o: the tile's outline (half of OUT_W outside the tile) stays >= 0.5 in from the trim
      mapW = mapH * MAP_AR, mapX = PW - M - mapW;
const leftR = mapX - 0.05 - 0.3;   // right edge of Box 3's outline and of the band's left column (0.3 in clear of the map tile)
const box3Top = y;
const W3 = leftR - BOX_IN - M;      // Box 3 content width
y += 0.1;
text('Where it aligns with the literature', { x: M, y, w: W3, h: HEAD, fontSize: 44, bold: true, color: K.navy, align: 'center', valign: 'middle', objectName: 'Heading literature' });
y += HEAD + 0.08;   // clear the heading's descenders
{
  const cg = 0.25, cw = [3.0, 0, 0]; cw[1] = (W3 - cw[0] - 2 * cg) * 0.46; cw[2] = W3 - cw[0] - cw[1] - 2 * cg;
  const cx = [M, M + cw[0] + cg, M + cw[0] + cw[1] + 2 * cg];
  const rows = [
    ['d’Ascoli et al. 2025', 'Per-person brain layers, fixed space', 'Inspired the NLP project’s shared embedding space'],
    ['Artetxe, LASER3, WECHSEL', 'New language inputs, existing space', 'Embedding space can work across languages'],
    ['Reimers 2020 · LaBSE', 'Training on translations aligns them', 'Reimers: how the page’s model was trained'],
    ['Libovický · Thompson', 'Only partly neutral; meanings differ', 'High score is not the same meaning'],
  ];
  const hdrH = 0.45, rh = [0.82, 0.82, 0.82, 0.82];   // two lines per cell at 22 pt
  const tH = hdrH + rh.reduce((p, q) => p + q, 0);
  ['Paper', 'What it did', 'What it means for us'].forEach((h, i) => text(h, { x: cx[i], y: y + 0.02, w: cw[i], h: hdrH - 0.05,
    fontSize: 22, bold: true, color: K.teal, valign: 'middle', objectName: `Literature head ${i + 1}` }));
  let ry = y + hdrH + 0.05;
  rows.forEach((r, j) => {
    hRule(M, ry - 0.03, W3, `Literature rule ${j + 1}`, j === 0 ? K.band : K.muted);
    r.forEach((t, i) => text(t, { x: cx[i], y: ry + 0.02, w: cw[i], h: rh[j] - 0.06, fontSize: 22, bold: i === 0, color: i === 0 ? K.navy : K.text2,
      valign: 'middle', objectName: `Literature row ${j + 1}.${i + 1}` }));
    ry += rh[j];
  });
  y += tH + 0.05 + 0.15;
  console.log(`Box 3 table columns ${cw.map(v => v.toFixed(2)).join(' / ')} in`);
}
s.addShape(pres.ShapeType.roundRect, { x: M - BOX_IN, y: box3Top, w: leftR - (M - BOX_IN), h: y - box3Top, rectRadius: 0.2,
  fill: { type: 'none' }, line: { color: K.band, width: BIG_W }, objectName: 'Box 3 literature' });
const box3H = y - box3Top;
y += BOX_GAP + 0.2;   // 2026-11-19i: 0.08 -> 0.2 (breathing room freed by the caption move)

// ================================================================= bottom left (2026-11-19e, Tim: "the navy band can just be a take away box so that it doesn't
// overlap"): no full-width band. Under Box 3, left column only: a navy Key takeaways box, then the map caption (right-aligned toward the tile)
// and the two disclaimer lines (the only footer remnant) on the white page. Text inside the box is named 'Tk ...' (contrast checked).
const tkTop = y;
{
  if (mapY - 0.05 < box3Top - 1e-9 || mapY + mapH + 0.05 > PH - 0.5 + 1e-9) throw new Error('map tile outside its slot');
  // white tile with a thin navy outline so it reads as a tile on the white page
  s.addShape(pres.ShapeType.rect, { x: mapX - 0.05, y: mapY - 0.05, w: mapW + 0.1, h: mapH + 0.1, fill: { color: K.white }, line: { color: K.band, width: OUT_W }, objectName: 'Map tile' });
  // Source: experiments/poster_sentences_map/poster_map_graphic_v4_crop.png (draw4.py: same placements as v3, labels drawn last with a white halo), copied unmodified (no digits in the image).
  s.addImage({ path: MAP_PNG, x: mapX, y: mapY, w: mapW, h: mapH, objectName: 'Map graphic poster_map_graphic',
    altText: 'Map of the poster’s sentences placed on the page’s t-SNE map, seen from the top: a cloud of colored points. Two red points sit close together, labeled “The dog bit the man.” and “The man bit the dog.”; a navy point far to the left is labeled “A high score is not the same meaning.”' });
  const bx = M - BOX_IN, bw = leftR - bx;
  const dH = 0.62, dY = PH - 0.5 - 0.03 - dH, tkH = dY - 0.3 - tkTop;   // 2026-11-19o: +0.03 in so descenders clear the 0.5 in margin in the 300 dpi export   // 2026-11-19i: the caption row is gone; 0.3 in gap to the disclaimers
  if (tkH < 3.4) throw new Error(`Key takeaways cramped: ${tkH.toFixed(2)} in`);
  rect(bx, tkTop, bw, tkH, { rectRadius: 0.2, fill: { color: K.band }, line: { color: K.band, width: OUT_W }, objectName: 'Key takeaways box' });
  // Tim's four takeaways (2026-11-18). Honesty guard: the page explains the model's cross-language behavior; 'language is the subject' is the aim.
  text([{ text: 'Key takeaways', options: { bold: true, color: K.mint, fontSize: TK_HEAD, breakLine: true } },
    bl('Same XAI toolkit, new target: how a model links languages.'),
    bl('Take XAI visualization to languages: the model is the lens, language is the subject.'),
    bl('A high score is not the same meaning: test the explanation, not just the score.'),
    bl('A shared space can hide untranslatable meaning: next question.', true)],   // 2026-11-19n: no orphan 'question.'
  { x: bx + 0.25, y: tkTop + 0.25, w: bw - 0.4, h: tkH - 0.5, fontSize: TK_BODY, color: K.white, valign: 'middle', paraSpaceAfter: 8, objectName: 'Tk key takeaways' });
  // map caption (Tim, 2026-11-19i: "change the poster sentences statement to just t-SNE, top view and put it in the bottom left corner of the
  // image in that white space"): inside the map tile, bottom-left, navy 20 pt, clear of the dot cloud (checked in the render)
  // 2026-11-19n (review): the caption becomes a short legend, as on the page (colour = topic, web/js/main.js colorBy 'topic'; navy = poster
  // sentences, coral = the dog / man pair, experiments/poster_sentences_map/draw4.py). Bottom-left white space, clear of dots (checked in the render).
  const legH = 1.45;   // four short lines, clear of the 'The dog bit the man.' label on the right
  text([{ text: 't-SNE, top view', options: { bold: true, breakLine: true } },
    { text: '\u25CF ', options: { color: '0B1F5C' } }, { text: 'Poster sentences     ' }, { text: '\u25CF ', options: { color: 'D9553B' } }, { text: 'The dog / man pair', options: { breakLine: true } },
    { text: 'Small dots: reference sentences,', options: { breakLine: true } }, { text: 'colored by topic' }],
  { x: mapX + 0.3, y: mapY + mapH - 0.25 - legH, w: 6.2, h: legH, fontSize: 18, color: K.navy, align: 'left', valign: 'bottom', paraSpaceAfter: 2, objectName: 'Map caption legend' });
  // disclaimers (Tim, 2026-11-19): Limits | References, then the short AI-usage line; bottom-aligned with the map tile, 0.5 in above the trim
  const SEP = { text: '   |   ', options: { color: K.muted } };
  // 2026-11-19n (Tim): Limits names the second model; the pointers go to the page's Credits , no file names
  text([{ text: 'Limits: ', options: { bold: true, color: K.brick } }, { text: 'one main model (a second only for checks), four languages' }, SEP,
    { text: 'References: ', options: { bold: true, color: K.navy } }, { text: 'see Credits on the page', options: { breakLine: true } },
    { text: 'AI usage: ', options: { bold: true, color: K.navy } },
    { text: 'Claude helped with development, research, and presentation. See Credits on the page.' }],
  { x: M, y: dY, w: bw - BOX_IN, h: dH, fontSize: 16, color: K.text2, valign: 'bottom', objectName: 'Disclaimers' });
  console.log(`map tile ${(mapW + 0.1).toFixed(2)} x ${(mapH + 0.1).toFixed(2)} in at (${(mapX - 0.05).toFixed(2)}, ${(mapY - 0.05).toFixed(2)}) (image ${mapW.toFixed(2)} x ${mapH.toFixed(2)}, labels ${(175 / MAP_BUF.readUInt32BE(16) * mapW * 72).toFixed(1)} pt); Box 3 ${bw.toFixed(2)} x ${box3H.toFixed(2)} in; Key takeaways box ${bw.toFixed(2)} x ${tkH.toFixed(2)} in (${TK_HEAD}/${TK_BODY} pt) at y ${tkTop.toFixed(2)}`);
}

// ================================================================= write pptx, numbers log, PDF
fs.mkdirSync(path.dirname(OUT_NUMS), { recursive: true });
fs.writeFileSync(OUT_NUMS, JSON.stringify({ built: new Date().toISOString(), note: 'every number printed on poster v3, with its source key', numbers: NUMS }, null, 1));
pres.writeFile({ fileName: OUT_PPTX }).then(f => {
  console.log('wrote', f); console.log('wrote', OUT_NUMS, `(${NUMS.length} numbers)`);
  if (process.env.POSTER_NO_PDF) return;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'poster-v2-'));
  execFileSync('soffice', [`-env:UserInstallation=file://${tmp}/profile`, '--headless', '--convert-to', 'pdf', '--outdir', tmp, f], { stdio: 'ignore' });
  fs.copyFileSync(path.join(tmp, 'LostInSpace_poster_v3.pdf'), OUT_PRINT);   // the file for the print shop and the submission
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log('wrote', OUT_PRINT);
}).catch(e => { console.error(e); process.exit(1); });
