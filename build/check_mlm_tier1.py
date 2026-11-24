"""Checks B's order.json, tsne.json, grad export against A's formats; merges results into eval/variants/mlm/data_checks.json."""
import json, pathlib, math, subprocess
R = pathlib.Path(__file__).resolve().parents[1]
L = lambda p: json.loads((R / p).read_text(encoding="utf-8"))
def shape(x, depth=0):
    if isinstance(x, dict): return {k: shape(v, depth + 1) for k, v in x.items()} if depth < 3 else "dict"
    if isinstance(x, list): return ["list", len(x), shape(x[0], depth + 1) if x and depth < 3 else None]
    return type(x).__name__
def keys_equal(a, b, path=""):
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b): bad.append((path, list(a), list(b)))
        for k in a:
            if k in b: bad += keys_equal(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and a and b:
        bad += keys_equal(a[0], b[0], path + "[0]")
    elif type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        bad.append((path, type(a).__name__, type(b).__name__))
    return bad
def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    if isinstance(x, float): return math.isfinite(x)
    return True
out = {}
for name in ("order", "tsne"):
    a, b = L(f"web/data/{name}.json"), L(f"web/data/mlm/{name}.json")
    out[name] = {"key_order_and_types_match_A": not keys_equal(a, b), "diffs": keys_equal(a, b)[:5], "all_finite": finite(b)}
a, b = L("web/data/tsne.json"), L("web/data/mlm/tsne.json")
out["tsne"].update({"n_coords": len(b["coords"]), "n_points_x3": 3 * L("web/data/mlm/cloud.json")["n"], "coords_len_equals_A": len(a["coords"]) == len(b["coords"]),
                    "coords_dim_note": "A: %d values" % len(a["coords"])})
a, b = L("web/data/order.json"), L("web/data/mlm/order.json")
out["order"]["probes_shape_equals_A"] = shape(a["probes"]) == shape(b["probes"]) and shape(a["pair_probe"]) == shape(b["pair_probe"])
g = L("eval/variants/mlm/grad_parity.json")
out["grad"] = {"manual_vs_autograd_grad_rel": g["manual_vs_autograd_max"]["grad_rel"], "onnx_fp16_vs_autograd_grad_rel": g["onnx_fp16_vs_autograd_max"]["grad_rel"],
               "limit_onnx": 2e-2, "pass": g["onnx_fp16_vs_autograd_max"]["grad_rel"] < 2e-2 and g["manual_vs_autograd_max"]["grad_rel"] < 1e-4,
               "size_bytes": (R / "web/model/mgrad_mlm/onnx/model.onnx").stat().st_size}
# A files unchanged
before = (R / ".cache/variants/mlm/A_t1_before.txt").read_text().splitlines()
now = subprocess.run("cd web/data; (md5sum *.json *.bin; stat -c %n\\ %Y *.json *.bin)", shell=True, executable="/bin/bash", cwd=R, capture_output=True, text=True).stdout.splitlines()
out["A_files_unchanged_md5_and_mtime"] = before == now
out["A_files_checked"] = len(before) // 2
p = R / "eval/variants/mlm/data_checks.json"
d = json.loads(p.read_text()); d["tier1_P2_27d"] = out
p.write_text(json.dumps(d, indent=1))
print(json.dumps(out, indent=1))
