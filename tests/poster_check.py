"""Checks poster v3 (and v2) against its contract (standard library only; needs pdftotext and pdfinfo on PATH;
pdftoppm and node with tests/node_modules/jsqr + pngjs for the QR decode: cd tests && npm install --no-save jsqr pngjs).

Usage: python3 -I tests/poster_check.py poster/LostInSpace_poster_v3.pptx poster/LostInSpace_poster_v3.pdf

Asserts:
  - the PDF page is 1728 x 2592 pt (24 x 36 in);
  - (2026-11-19n: footer pointers now 'see Credits on the page'; REFERENCES.md / AI_USAGE.md are cut; 'deserves belief' cut)
  - the PDF text contains the live URL, "Builds on", "NLP project", "New here", "deserves belief", "Why I built it",
    the literature section ("Where it aligns with the literature", "Key takeaways", "Embedding space can work across
    languages"), the d'Ascoli figure credit ("CC BY 4.0") and the footer pointers ("REFERENCES.md", "AI_USAGE.md"),
    and none of the cut texts (including the dropped question headline, "(not shipped)", "teaching aid",
    "Nothing is trained" and "Future work (not built)");
  - the slide has exactly one chart graphicFrame and no table;
  - the map graphic (picture named "Map graphic poster_map_graphic", with alt text) is embedded and its caption
    ("t-SNE, top view") is printed inside the map tile's bottom-left quarter;
  - layout v5/v6: the QR picture sits inside the navy header band; the map picture and the disclaimer lines end >= 0.5 in
    above the bottom trim edge; the old footer, QR band, bottom band and title rule shapes are gone; the map tile starts
    below Box 2 and overlaps neither Box 3 nor the Key takeaways box;
  - jsQR decodes the live URL from a 100 dpi render of the PDF, cropped around the QR picture's pptx position;
  - text on navy (shapes named Hdr* on the header band, Tk* in the Key takeaways box) contrasts with that fill: >= 4.5:1,
    or >= 3:1 for large text (>= 24 pt, or >= 18.66 pt bold);
  - every text run is >= 20 pt, except in shapes named References*, AI use*, Map caption* or Disclaimers* (>= 14 pt);
    chart text sizes >= 16 pt;
  - chart bars contrast >= 3:1 with the card behind them (POSTER-7);
  - every number in the PDF text, other than years 1900-2099, is among the numbers the builder logged
    to poster/out/v3_numbers.json (each with its JSON source key, docs/literature_check.md for a cited paper's figures,
    "REFERENCES.md <n>" for a citation identifier in the figure credit, or "definition: ..." for a constant such as chance level).

poster/out/ is gitignored: if v3_numbers.json is missing, run `node build_poster_v3.js` in poster/ first.
"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}
REQUIRED = ["timforcejr.github.io/lost_in_space", "Lost in Space", "Builds on", "NLP project", "New here", "should be trusted alone",
            "Why I built it", "Where it aligns with the literature", "Key takeaways", "Embedding space can work across languages",
            "CC BY 4.0", "see Credits on the page", "t-SNE, top view", "Tim Force, MSECE", "Machine Learning Track",
            "see why they match", "Scan the QR code to try it.", "The model runs on your device.", "one main model (a second only for checks), four languages", "scored 0.98 alike",
            # 2026-11-19f (layout v7): both d'Ascoli paths; Tim's wording under the chart
            "Vocabulary word", "T5 language model (fixed)", "Target word embeddings", "Predicted word embedding",
            "Same embedding space", "brain output trained to land on T5 targets", "Contrastive loss, cosine retrieval",  # 2026-11-19j (v11)
            "Runs in your browser; nothing leaves your device.", "XAI tools: attribution, maps, layer lens, faithfulness and sanity tests.",
            "Visualization to build understanding of how an AI learns meaning.",
            "Four ways to find the words that drove the score: occlusion, integrated gradients, attention rollout, pair split.",  # 2026-11-19m (v14)  # 2026-11-19k (v12), third bullet 2026-11-19l (v13)
            # 2026-11-19h (layout v9): Tim's one fine-print paragraph under the chart
            "Deletion test, 1,224 explanations (higher = more faithful): no single heat map should be trusted alone. Pair split barely beats "
            "guessing by word length, the order changes on a second model, and attention rollout barely changes when 8 of 12 layers are reset (0.97).",
            # 2026-11-19n (layout v15): review fixes, footer to the page's Credits, card A bullet, map legend
            "Random words", "t-SNE 41.9%: places a sentence at the center of its nearest neighbors.", "High score is not the same meaning",
            "A shared space can hide untranslatable meaning: next question.", "Scramble every word: median still 0.91 like the original.",
            "Paraphrase vs look-alike: ranked right 57% of the time (coin flip: 50%).", "384-D",
            "Claude helped with development, research, and presentation. See Credits on the page.",
            # 2026-11-19p (layout v18): symmetric Why / What you can see / What comes next columns
            "A word2vec lecture started this rabbit hole.", "What does it say about how we communicate?",
            "Is it learning differences? Can we see them?", "A 3D map of how the model sees meaning.",
            "Close points: meanings the model treats alike.", "Gaps between translations: meaning may differ.",
            "Highlighted words: which ones drove the score.", "Migrate to the NLP model design.", "Add additional languages.",
            "Poster sentences", "The dog / man pair", "Small dots: reference sentences, colored by topic"]
GONE_SHAPES = ["Footer", "Footer rule", "QR band", "Title rule", "Bottom band"]   # removed in layouts v5/v6 (2026-11-19d/e)
ON_BAND = ("Hdr", "Tk")
CUT = ["Word order fades at the end", "Probe accuracy: natural or scrambled?", "Five explanations of one score",
       "Which word pairs carry the score?", "Can you trust the map?", "Trust@10", "Top-10 kept",
       "Type two sentences, watch them land", "faithfulness gap: the top method", "real backpropagation",
       "Why does a model say two sentences", "(not shipped)", "teaching aid", "Nothing is trained", "Future work (not built)",
       "frozen space", "neighbours", "centring",
       # cut 2026-11-19f (layout v7): old text under the chart and the old d'Ascoli side box
       "Occlusion’s lead shrinks if masked; a second model reorders.", "Occlusion's lead shrinks if masked",
       "Attention rollout fails the sanity check (0.97 rank agreement).", "rank agreement",
       "Pair split modestly beats word length; no better than random order on a second model.",
       "fixed target",
       # cut 2026-11-19g (layout v8): the v7 wording under the chart
       "a bigger score drop means more faithful", "length-only guess", "and the ranking changes", "still picks the same words",
       # cut 2026-11-19h (layout v9): the five v8 lines
       "Deletion test, 1,224 explanations: higher = more faithful.", "Pair split barely beats guessing by word length.",
       "No single heat map deserves belief.", "On a second model, the order changes.",
       "Rollout barely changes when layers are reset (0.97).",
       # cut 2026-11-19i (layout v10): the long map caption
       "Poster sentences placed on the page’s map (t-SNE, top view).", "placed on the page",
       # cut 2026-11-19k (layout v12): the old first New here bullet
       "the whole model runs in your browser; nothing you type leaves your device.",
       "Five explanation methods, tested against each other.", "Visuals that build understanding",   # cut 2026-11-19l (layout v13)
       "Grad-CAM", "Five ways", "0.26",
       # cut 2026-11-19n (layout v15)
       "REFERENCES.md", "AI_USAGE.md", "deserves belief", "ignores distance", "Random order", "384-d", "not same meaning",
       "Scan the QR code to demo", "for full citations", "one model, four languages", "the next question",
       "Real paraphrase or re-ordered look-alike?", "A coin flip is 50%.", "(a coin flip is 50%)",   # cut 2026-11-19o (v17)
       "An NLP lecture on word2vec", "What does that tell us about communicating", "Can we visualize the difference?",
       "how the model sees the world", "Close points are meanings", "Highlighted words show which"]   # cut 2026-11-19p (v18)   # cut 2026-11-19m (layout v14): Grad-CAM removed from the poster
SMALL_OK = ("References", "AI use", "Map caption", "Disclaimers")
LIVE_URL = "https://timforcejr.github.io/lost_in_space/"
EMU = 914400
TESTS = Path(__file__).resolve().parent
QR_JS = ("const {PNG}=require('pngjs'),jsQR=require('jsqr'),fs=require('fs');"
         "const p=PNG.sync.read(fs.readFileSync(process.argv[1]));"
         "const r=jsQR(new Uint8ClampedArray(p.data),p.width,p.height);process.stdout.write(r?r.data:'');")

failures = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        failures.append(msg)


def norm_num(t):
    return t.replace(",", "").replace("%", "").replace("−", "").replace("-d", "").replace("-D", "").lstrip("-")   # -D: 2026-11-19p (384-D)


def luminance(hexcol):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(c1, c2):
    l1, l2 = sorted((luminance(c1), luminance(c2)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def main(pptx, pdf):
    pptx, pdf = Path(pptx), Path(pdf)
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
    check(size and round(float(size.group(1))) == 1728 and round(float(size.group(2))) == 2592,
          f"page size 1728 x 2592 pt (got {size.group(0) if size else 'none'})")

    raw = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    flat = " ".join(raw.split())
    for t in REQUIRED:
        check(t in flat, f"PDF text contains {t!r}")
    for t in CUT:
        check(t not in flat, f"PDF text lacks cut text {t!r}")

    with zipfile.ZipFile(pptx) as z:
        slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        check(len(slides) == 1, f"one slide (got {len(slides)})")
        root = ET.fromstring(z.read(slides[0]))
        charts = [g for g in root.iter(f"{{{NS['p']}}}graphicFrame")
                  if g.find(".//c:chart", NS) is not None]
        tables = list(root.iter(f"{{{NS['a']}}}tbl"))
        check(len(charts) == 1, f"exactly 1 chart graphicFrame (got {len(charts)})")
        check(len(tables) == 0, f"no table (got {len(tables)})")

        pics = {}
        for pic in root.iter(f"{{{NS['p']}}}pic"):
            nv = pic.find("p:nvPicPr/p:cNvPr", NS)
            off, ext = pic.find("p:spPr/a:xfrm/a:off", NS), pic.find("p:spPr/a:xfrm/a:ext", NS)
            pics[nv.get("name", "")] = (nv.get("descr", ""), int(off.get("y")) + int(ext.get("cy")),
                                        tuple(int(v) / EMU for v in (off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy"))))
        sld_h = int(ET.fromstring(z.read("ppt/presentation.xml")).find("p:sldSz", NS).get("cy"))
        mp = next((k for k in pics if k.startswith("Map graphic")), None)
        check(mp is not None and len(pics[mp][0]) > 20 and any(n.startswith("ppt/media/") for n in z.namelist()),
              f"map graphic picture embedded with alt text (name {mp!r})")
        if mp in pics:
            gap = (sld_h - pics[mp][1]) / EMU
            check(gap >= 0.5 - 1e-6, f"{mp} ends {gap:.2f} in above the bottom trim (>= 0.5 in)")
        else:
            check(False, "map picture present")

        shapes = {}
        for sp in root.iter(f"{{{NS['p']}}}sp"):
            nm = sp.find("p:nvSpPr/p:cNvPr", NS).get("name", "")
            off, ext = sp.find("p:spPr/a:xfrm/a:off", NS), sp.find("p:spPr/a:xfrm/a:ext", NS)
            if off is not None and ext is not None:
                shapes[nm] = tuple(int(v) / EMU for v in (off.get("x"), off.get("y"), ext.get("cx"), ext.get("cy")))
        hb = shapes.get("Header band")
        qr = pics.get("QR live")
        check(hb is not None and qr is not None and qr[2][1] >= hb[1] + 0.2 and qr[2][1] + qr[2][3] <= hb[1] + hb[3] - 0.2,
              f"QR picture inside the header band (QR {qr[2] if qr else None}, band {hb})")
        dis = shapes.get("Disclaimers")
        check(dis is not None and sld_h / EMU - (dis[1] + dis[3]) >= 0.5 - 1e-6,
              f"disclaimer lines end >= 0.5 in above the bottom trim ({(sld_h / EMU - dis[1] - dis[3]) if dis else None})")
        gone = [n for n in GONE_SHAPES if n in shapes]
        check(not gone, f"old footer / QR band / bottom band / title rule shapes removed (found {gone})")
        mt, b2, b3, tk = (shapes.get(n) for n in ("Map tile", "Box 2 tools", "Box 3 literature", "Key takeaways box"))
        apart = lambda a, b: a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
        check(all((mt, b2, b3, tk)) and mt[1] >= b2[1] + b2[3] and apart(mt, b3) and apart(mt, tk) and apart(b3, tk),
              f"map tile below Box 2, clear of Box 3 and the Key takeaways box (tile {mt})")
        check(mt is not None and sld_h / EMU - (mt[1] + mt[3] + 2.25 / 144) >= 0.5 - 1e-6,
              f"map tile outline (2.25 pt) ends >= 0.5 in above the bottom trim ({(sld_h / EMU - mt[1] - mt[3] - 2.25 / 144) if mt else None})")   # 2026-11-19o
        mc = shapes.get("Map caption legend")   # 2026-11-19n: caption + legend, bottom-left inside the tile
        check(mt is not None and mc is not None and mc[0] >= mt[0] and mc[0] < mt[0] + mt[2] / 2 and mc[0] + mc[2] <= mt[0] + mt[2]
              and mc[1] + mc[3] <= mt[1] + mt[3] and mc[1] >= mt[1] + mt[3] / 2,
              f"map caption/legend inside the map tile, bottom-left (legend {mc})")

        bad, nruns, card_fill, low = [], 0, {}, []
        for sp in root.iter(f"{{{NS['p']}}}sp"):
            name = sp.find("p:nvSpPr/p:cNvPr", NS).get("name", "")
            fill = sp.find("p:spPr/a:solidFill/a:srgbClr", NS)
            if fill is not None:
                card_fill[name] = fill.get("val")
            floor = 1400 if name.startswith(SMALL_OK) else 2000
            for r in sp.iter(f"{{{NS['a']}}}r"):
                nruns += 1
                rpr = r.find("a:rPr", NS)
                sz = rpr.get("sz") if rpr is not None else None
                txt = (r.findtext("a:t", default="", namespaces=NS) or "")[:40]
                if sz is None or int(sz) < floor:
                    bad.append(f"{name}: {sz} {txt!r}")
                if name.startswith(ON_BAND) and txt.strip():
                    clr = rpr.find("a:solidFill/a:srgbClr", NS) if rpr is not None else None
                    big = sz and (int(sz) >= 2400 or (rpr.get("b") == "1" and int(sz) >= 1866))
                    band_fill = card_fill.get("Header band") if name.startswith("Hdr") else card_fill.get("Key takeaways box")
                    ratio = contrast(clr.get("val"), band_fill) if clr is not None and band_fill else 0
                    if ratio < (3 if big else 4.5):
                        low.append(f"{name}: {clr.get('val') if clr is not None else None} {ratio:.2f} {txt!r}")
        check(not bad, f"all {nruns} runs >= 20 pt (>= 14 pt in References*/AI use*/Map caption*/Disclaimers*)" + (": " + "; ".join(bad[:5]) if bad else ""))
        check("Header band" in card_fill and "Key takeaways box" in card_fill and not low,
              "text on the header band and in the Key takeaways box >= 4.5:1 (>= 3:1 large)" + (": " + "; ".join(low[:5]) if low else ""))

        chart_files = [n for n in z.namelist() if re.fullmatch(r"ppt/charts/chart\d+\.xml", n)]
        small, bars = [], []
        for n in chart_files:
            cx = z.read(n).decode("utf8")
            small += [int(v) for v in re.findall(r'\bsz="(\d+)"', cx) if int(v) < 1600]
            croot = ET.fromstring(cx)
            for ser in croot.iter(f"{{{NS['c']}}}ser"):
                for clr in ser.findall("c:spPr/a:solidFill/a:srgbClr", NS) + ser.findall("c:dPt/c:spPr/a:solidFill/a:srgbClr", NS):
                    bars.append(clr.get("val"))
        check(chart_files and not small, f"chart text sizes >= 16 pt in {len(chart_files)} chart(s)" + (f": found {small}" if small else ""))
        bg_name = next((n for n in ("Card B", "Card 2") if n in card_fill), None)  # the card that holds the chart
        bg = card_fill.get(bg_name)
        ratios = {c: round(contrast(c, bg), 2) for c in set(bars)} if bg else {}
        check(bg and ratios and min(ratios.values()) >= 3,
              f"chart bar contrast >= 3:1 against {bg_name} fill {bg}: {ratios}")

    ver = "v3" if "_v3" in pptx.name else "v2"   # numbers log written by the matching builder
    log = json.loads((pptx.parent / "out" / f"{ver}_numbers.json").read_text(encoding="utf8"))
    logged = {norm_num(e["text"]) for e in log["numbers"]}
    # Data numbers trace to eval/ or poster/fig/ JSON keys. Numbers quoted from a cited paper (Related-work strip) may instead
    # cite docs/literature_check.md, where the verified record holds them (rule: literature numbers only if recorded there).
    unsourced = [e for e in log["numbers"] if not re.match(r"(eval|poster/fig)/[\w.]+\.json:|docs/literature_check\.md |REFERENCES\.md \d+ |definition: ", e["source"])]
    check(not unsourced, f"every logged number has an eval/ or poster/fig/ JSON key, a docs/literature_check.md or REFERENCES.md source, or a stated definition ({len(log['numbers'])} logged)")
    found = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", raw)
    untraced = sorted({t for t in found if not re.fullmatch(r"(19|20)\d\d", t) and norm_num(t) not in logged})
    check(not untraced, f"every number in the PDF text traces to {ver}_numbers.json ({len(found)} found)"
          + (f": untraced {untraced}" if untraced else ""))

    # QR decode from the PDF render: 100 dpi, cropped 0.4 in around the QR picture's position in the pptx (layout v5: top right)
    tmp = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True, check=True).stdout.strip())
    try:
        qx, qy, qw, qh = qr[2] if qr else (0, 0, 24, 36)
        box = [max(0, int((qx - 0.4) * 100)), max(0, int((qy - 0.4) * 100)), int((qw + 0.8) * 100), int((qh + 0.8) * 100)]
        subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", "-x", str(box[0]), "-y", str(box[1]), "-W", str(box[2]), "-H", str(box[3]),
                        str(pdf), str(tmp / "qr")], check=True)
        dec = subprocess.run(["node", "-e", QR_JS, str(tmp / "qr.png")], capture_output=True, text=True,
                             env={"NODE_PATH": str(TESTS / "node_modules"), "PATH": __import__("os").environ.get("PATH", "")})
        got = dec.stdout.strip()
        check(got == LIVE_URL, f"jsQR decodes the QR from the PDF render: {got!r}" + (f" ({dec.stderr.strip()[:200]})" if dec.returncode else ""))
    finally:
        subprocess.run(["rm", "-rf", str(tmp)])

    print("OK" if not failures else f"{len(failures)} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
