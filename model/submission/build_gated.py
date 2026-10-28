from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
ROOT = CHALLENGE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from utils import DataPoint, FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, validate_sequence  # noqa: E402
from eval_ensemble import GlobalWP  # noqa: E402
from build_submission import export_weights, load_any, make_net  # noqa: E402

QUANTA = np.array([0.12543, 0.24133, 0.46853, 0.65686])
LUT_COLUMNS = ["a7", "i0_p10", "i0_p14", "i1_p10", "i1_p14"]


def seq_class(y):
    u = np.abs(y.ravel())
    u = u[u > 1e-6]
    return int(np.argmin(np.abs(QUANTA - u.min()))) if len(u) else -1


def build_lut(per_class, seed=0):
    rng = np.random.default_rng(seed)
    seen = [dict() for _ in LUT_COLUMNS]
    for split in ("train", "valid"):
        pf = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
        n = pf.metadata.num_row_groups
        cls = np.zeros(n, np.int64)
        for g in range(n):
            t = pf.read_row_group(g, columns=list(TARGET_COLUMNS), use_threads=False)
            cls[g] = seq_class(np.stack([t[c].to_numpy() for c in TARGET_COLUMNS], 1))
        picks = np.concatenate([rng.permutation(np.where(cls == c)[0])[:per_class] for c in range(4)])
        for g in picks:
            t = pf.read_row_group(int(g), columns=LUT_COLUMNS, use_threads=False)
            for li, c in enumerate(LUT_COLUMNS):
                for v in np.unique(t[c].to_numpy().astype(np.float32)):
                    seen[li].setdefault(float(v), set()).add(int(cls[g]))
        print(f"LUT: {split} {len(picks)} secuencias", flush=True)
    vals, labs, off = [], [], [0]
    for li in range(len(LUT_COLUMNS)):
        keys = np.array(sorted(seen[li]), np.float32)
        lab = np.array([next(iter(seen[li][float(k)])) if len(seen[li][float(k)]) == 1 else -1 for k in keys], np.int64)
        vals.append(keys)
        labs.append(lab)
        off.append(off[-1] + len(keys))
        print(f"   {LUT_COLUMNS[li]}: {len(keys)} valores, {(lab < 0).sum()} ambiguos", flush=True)
    return {"lut_cols": np.array([FEATURE_COLUMNS.index(c) for c in LUT_COLUMNS], np.int64),
            "lut_vals": np.concatenate(vals), "lut_class": np.concatenate(labs), "lut_off": np.array(off, np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", required=True, help="JSON: {\"0\": [pt...], \"1\": [...], \"2\": [...], \"3\": [...], \"fallback\": [...]}")
    ap.add_argument("--lut-per-class", type=int, default=400)
    ap.add_argument("--decide-row", type=int, default=99)
    ap.add_argument("--verify-seqs", type=int, default=8)
    ap.add_argument("--bits", type=int, choices=[8, 16, 32], default=8)
    ap.add_argument("--out", default=str(CHALLENGE / "submissions"))
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    groups = [spec[str(c)] for c in range(4)] + [spec["fallback"]]
    paths = []
    for g in groups:
        for p in g:
            if p not in paths:
                paths.append(p)
    cw = np.zeros((5, len(paths)), np.float32)
    for r, g in enumerate(groups):
        for p in g:
            cw[r, paths.index(p)] = 1.0 / len(g)
    build_dir = Path(args.out) / args.name
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "solution_gated.py", build_dir / "solution.py")
    tmp = build_dir / "_base.npz"
    export_weights(paths, None, tmp, args.bits)
    W = dict(np.load(tmp))
    tmp.unlink()
    W["class_weights"] = cw
    W["decide_row"] = np.int64(args.decide_row)
    W.update(build_lut(args.lut_per_class))
    np.savez(build_dir / "weights.npz", **W)
    zip_path = Path(args.out) / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(build_dir / "solution.py", "solution.py")
        z.write(build_dir / "weights.npz", "weights.npz")
    (build_dir / "BUILD.json").write_text(json.dumps({"spec": spec, "paths": paths, "bits": args.bits,
                                                      "decide_row": args.decide_row}, indent=2))
    print(f"zip: {zip_path}  ({zip_path.stat().st_size / 1e6:.2f} MB)  {len(paths)} redes", flush=True)
    if args.verify_seqs:
        verify(build_dir, paths, cw, args.verify_seqs)


def verify(build_dir, paths, cw, n_seqs):
    import importlib
    import torch
    sys.path.insert(0, str(build_dir))
    sol = importlib.import_module("solution")
    model = sol.PredictionModel()
    refs = []
    for p in paths:
        sd, Hm, mean, std, clip, cell = load_any(p)
        refs.append((make_net(sd, Hm, cell), mean, std, clip))
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    n = pf.metadata.num_row_groups
    cls = np.zeros(n, np.int64)
    for g in range(n):
        t = pf.read_row_group(g, columns=list(TARGET_COLUMNS), use_threads=False)
        cls[g] = seq_class(np.stack([t[c].to_numpy() for c in TARGET_COLUMNS], 1))
    picks = np.concatenate([np.where(cls == c)[0][: max(1, n_seqs // 4)] for c in range(4)])
    gacc = GlobalWP()
    max_diff, t_pred, rows, ok = 0.0, 0.0, 0, 0
    for g in picks:
        tb = pf.read_row_group(int(g), use_threads=False)
        seq, need = validate_sequence(tb)
        X = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        P = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
        t = time.perf_counter()
        for s in range(SEQUENCE_LENGTH):
            v = model.predict(DataPoint(seq, s, bool(need[s]), X[s]))
            if need[s]:
                P[s] = v
        t_pred += time.perf_counter() - t
        rows += SEQUENCE_LENGTH
        ok += int(model.cls == cls[g])
        w = cw[cls[g]]
        with torch.no_grad():
            R = sum(wi * net["head"](net["core"](torch.from_numpy(np.clip((X - mean) / std, -clip, clip).astype(np.float32)).unsqueeze(0))[0]).squeeze(0).numpy()
                    for wi, (net, mean, std, clip) in zip(w, refs) if wi > 0)
        max_diff = max(max_diff, float(np.abs(P[need] - R[need]).max()))
        gacc.add(Y, P, need & tb["is_scored"].to_numpy().astype(bool))
    r = gacc.result()
    us = t_pred / rows * 1e6
    print(f"instrumento detectado bien: {ok}/{len(picks)}")
    print(f"paridad: max |solution - PyTorch| = {max_diff:.2e}")
    print(f"{len(picks)} seqs de valid: GLOBAL WP={r['wp']:.4f}")
    print(f"tiempo: {us:.1f} us/fila -> {us * 37_460_000 / 60e6:.1f} min proyectados (laptop)")


if __name__ == "__main__":
    main()
