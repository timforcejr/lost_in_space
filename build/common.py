"""
Purpose:          Shared helpers for the Lost in Space build pipeline: model
                  loading, mean pooling, word grouping, SIB-200 loading.
Verification:     Exercised by export_model.py / build_cloud.py /
                  eval_attribution.py runs logged in eval/*.json
"""
from __future__ import annotations

import csv
import io
import json
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
EVAL = ROOT / "eval"
CACHE = ROOT / ".cache"

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIRNAME = "mmini"  # web/model/mmini
LANGS = {"en": "eng_Latn", "es": "spa_Latn", "hi": "hin_Deva", "zh": "zho_Hans"}
LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese"}
SIB_URL = "https://huggingface.co/datasets/Davlan/sib200/resolve/main/data/{code}/{split}.tsv"
SEED = 20261115


def load_model(device: str = "cpu", model_id: str = MODEL_ID):
    """Return (tokenizer, encoder) with eager attention so attentions can be read."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE / "hf")
    model = AutoModel.from_pretrained(model_id, cache_dir=CACHE / "hf", attn_implementation="eager")
    return tok, model.eval().to(device)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens (special tokens included, as sentence-transformers does)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


@torch.no_grad()
def embed(tok, model, texts, batch_size: int = 64, layers: bool = False):
    """Unit-normalized sentence embeddings; with layers=True also per-layer mean pools [N, L+1, H]."""
    outs, per_layer = [], []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        res = model(**enc, output_hidden_states=layers)
        outs.append(mean_pool(res.last_hidden_state, enc["attention_mask"]).cpu())
        if layers:
            per_layer.append(torch.stack([mean_pool(h, enc["attention_mask"]) for h in res.hidden_states], 1).cpu())
    e = torch.nn.functional.normalize(torch.cat(outs), dim=-1).numpy()
    if layers:
        return e, torch.cat(per_layer).numpy()
    return e


def run_layers(model, h0: torch.Tensor):
    """Hidden states [h0, h1 .. hL] from layer-0 states (the embedding output), no padding.

    This is the differentiable path that integrated gradients (and the offline Grad-CAM benchmark rows) use, and the one the
    browser's gradient model (build/export_grad.py) reproduces.
    """
    hs, x = [h0], h0
    for layer in model.encoder.layer:
        out = layer(x)
        x = out if torch.is_tensor(out) else out[0]     # transformers 5 returns a tensor, 4.x a tuple
        hs.append(x)
    return hs


def load_sib(split: str):
    """SIB-200 rows for the four languages, aligned by index_id.

    Returns {lang: [{'id', 'topic', 'text'}...]} with identical id order in every language.
    """
    CACHE.mkdir(exist_ok=True)
    data = {}
    for lang, code in LANGS.items():
        path = CACHE / f"sib200_{code}_{split}.tsv"
        if not path.exists():
            with urllib.request.urlopen(SIB_URL.format(code=code, split=split), timeout=60) as r:
                path.write_bytes(r.read())
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE))
        data[lang] = {int(r["index_id"]): {"id": int(r["index_id"]), "topic": r["category"], "text": r["text"].strip()} for r in rows}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    return {lang: [data[lang][i] for i in ids] for lang in LANGS}


def is_wordlike(s: str) -> bool:
    return any(unicodedata.category(c)[0] in "LMN" for c in s)


SPECIAL = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}
GLUE = {"'", "\u2019", "-"}


def _is_punct(piece: str) -> bool:
    return not any(unicodedata.category(c)[0] in "LMN" for c in piece)


def _is_han(piece: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" for c in piece)


def word_groups(tokens):
    """Group SentencePiece tokens into display words.  MUST mirror web/js/tokens.js exactly.

    tokens: token strings for the whole sequence, special tokens included.
    Returns a list of {'text', 'tokens': [positions], 'punct': bool}.  A new word starts at a
    SentencePiece boundary; punctuation and Han pieces are always their own unit (Chinese has
    no spaces, so each piece the tokenizer produced is one unit).
    """
    groups, pending = [], False
    for i, t in enumerate(tokens):
        if t in SPECIAL:
            continue
        piece = t.replace("\u2581", "")
        starts = t.startswith("\u2581")
        if piece == "":
            pending = True
            continue
        punct, han = _is_punct(piece), _is_han(piece)
        prev = groups[-1] if groups else None
        new_word = starts or pending
        pending = False
        if prev is not None and not new_word and not prev["punct"] and not prev["han"]:
            if piece in GLUE:                      # don't, l'eau, well-known: keep as one word
                prev["text"] += piece
                prev["tokens"].append(i)
                prev["glue"] = True
                continue
            if not punct and not han:              # ordinary sub-word continuation
                prev["text"] += piece
                prev["tokens"].append(i)
                prev["glue"] = False
                continue
        groups.append({"text": piece, "tokens": [i], "punct": punct, "han": han, "glue": False})
    return groups


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
