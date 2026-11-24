"""
Purpose: dump tokens + Python word groups for every SIB-200 sentence so tests/grouping_parity.mjs can
check that web/js/tokens.js groups them identically.
Run:  .venv/bin/python tests/grouping_parity.py && node tests/grouping_parity.mjs
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from common import load_model, load_sib, word_groups

tok, _ = load_model()
rows = []
extra = ["I don't love this movie, honestly.", "L'eau est froide — très froide!", "well-known 3D-printed parts cost $4.50", "猫在沙发上睡觉。", "यह एक परीक्षण है।", "¿Dónde está la biblioteca?"]
texts = extra + [r["text"] for s in ("train", "dev", "test") for rs in load_sib(s).values() for r in rs]
for t in texts:
    toks = tok.convert_ids_to_tokens(tok(t, truncation=True, max_length=128)["input_ids"])
    rows.append({"tokens": toks, "groups": [[g["text"], g["tokens"], g["punct"]] for g in word_groups(toks)]})
out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
(out / "grouping_cases.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
print(len(rows), "cases written")
