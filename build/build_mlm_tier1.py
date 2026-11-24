"""
Purpose:          Wrapper that runs the unmodified build/eval_order.py (step order) or build/build_tsne.py (step tsne) on
                  model B (microsoft/Multilingual-MiniLM-L12-H384) by patching common.MODEL_ID/load_model/WEB/EVAL (scratch
                  dirs under .cache/variants/mlm/) before import, then copies the web file to web/data/mlm/. A's files untouched.
Verification:     eval/variants/mlm/data_checks.json, eval/variants/mlm/data_README.md

Run:  PYTHONDONTWRITEBYTECODE=1 python3 -I build/build_mlm_tier1.py order|tsne|grad
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

step = sys.argv[1]
B_REPO = "microsoft/Multilingual-MiniLM-L12-H384"
SCR = ROOT / ".cache/variants/mlm"
common.MODEL_ID = B_REPO
common.WEB = SCR / "webroot"
common.EVAL = SCR / "evalroot"
common.load_model = lambda device="cpu", model_id=B_REPO: ea.load_any("mminilm")
DST = ROOT / "web/data/mlm"; DST.mkdir(parents=True, exist_ok=True)
if step == "order":
    import eval_order as m
    assert m.WEB == SCR / "webroot"
    m.main()
    shutil.copyfile(SCR / "webroot/data/order.json", DST / "order.json")
    shutil.copyfile(SCR / "evalroot/order.json", ROOT / "eval/variants/mlm/order_eval.json")
elif step == "tsne":
    import build_tsne as m
    assert m.WEB == SCR / "webroot"
    m.main()
    shutil.copyfile(SCR / "webroot/data/tsne.json", DST / "tsne.json")
    shutil.copyfile(SCR / "evalroot/tsne.json", ROOT / "eval/variants/mlm/tsne_eval.json")
elif step == "grad":
    import export_grad as m
    assert m.OUT == SCR / "webroot/model/mgrad"
    m.main()
    out = ROOT / "web/model/mgrad_mlm"
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(m.OUT / "onnx/model.onnx", out / "onnx/model.onnx")
    shutil.copyfile(m.OUT / "config.json", out / "config.json")
    shutil.copyfile(SCR / "evalroot/grad_parity.json", ROOT / "eval/variants/mlm/grad_parity.json")
else:
    raise SystemExit("step: order|tsne|grad")
print("done", step)
