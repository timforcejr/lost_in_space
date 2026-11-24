"""
Purpose:          Wrapper that runs the existing build/build_cloud.py and build/eval_placement.py recipe on model B
                  (microsoft/Multilingual-MiniLM-L12-H384) by patching common.MODEL_ID, load_model, WEB, EVAL before the
                  import. Writes B's cloud.json, cloud_emb.i8.bin, placement.json to web/data/mlm/ (A's files untouched)
                  and a per-model eval/variants/mlm/map_fidelity.json. Does not edit build_cloud.py or common.py.
Verification:     eval/variants/mlm/data_checks.json (build/check_mlm_data.py), eval/variants/mlm/data_README.md

Run (detached):
  PYTHONDONTWRITEBYTECODE=1 python3 -I build/build_mlm_data.py
"""
import sys, shutil, pathlib
sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / ".venv/lib/python3.13/site-packages", HERE, ROOT / "experiments/p2_25_alignment_vs_drift"):
    if str(p) not in sys.path:
        sys.path.append(str(p))
import common
import embed_any as ea

B_REPO = "microsoft/Multilingual-MiniLM-L12-H384"
SCR = ROOT / ".cache/variants/mlm"
common.MODEL_ID = B_REPO
common.WEB = SCR / "webroot"          # build_cloud writes WEB/data/*; copied to web/data/mlm below (webroot/js is a symlink to web/js for EXAMPLES)
common.EVAL = SCR / "evalroot"
common.load_model = lambda device="cpu", model_id=B_REPO: ea.load_any("mminilm")

import build_cloud, eval_placement      # noqa: E402  (bind the patched names)
assert build_cloud.MODEL_ID == B_REPO and build_cloud.WEB == SCR / "webroot"
build_cloud.main()
eval_placement.main()

DST = ROOT / "web/data/mlm"; DST.mkdir(parents=True, exist_ok=True)
for f in ("cloud.json", "cloud_emb.i8.bin", "placement.json"):
    shutil.copyfile(SCR / "webroot/data" / f, DST / f)
shutil.copyfile(SCR / "evalroot/map_fidelity.json", ROOT / "eval/variants/mlm/map_fidelity.json")
shutil.copyfile(SCR / "evalroot/placement.json", ROOT / "eval/variants/mlm/placement.json")
print("copied to", DST)
