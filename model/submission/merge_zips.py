"""Arma una entrega nueva juntando modelos de entregas ya hechas, sin reentrenar ni volver a exportar.

    python model/submission/merge_zips.py --out ens5_E_X_mask_LR_long_int8 \
        --part ens4_y2_Eswa_X_Zswa_int8:1,2 --part ens4_mask_LR_LA_long_int8:0,1,3

Cada --part es <carpeta en submissions/>:<índices de sus modelos>. Todas deben usar el mismo kernel
(solution.py idéntico), los mismos bits y el mismo tamaño de relleno. El promedio queda parejo.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

SUB = Path(__file__).resolve().parents[2] / "submissions"
SCALAR = {"bits", "hidden", "n_models", "model_weights"}


def merge(parts, out_dir: Path, make_zip: bool = True) -> int:
    arrays, build, bits, hidden, kernels = {}, {"models": [], "cells": []}, set(), set(), set()
    for src, idx in parts:
        z = np.load(SUB / src / "weights.npz")
        b = json.loads((SUB / src / "BUILD.json").read_text(encoding="utf-8"))
        for k in z.files:
            if k not in SCALAR:
                arrays.setdefault(k, []).append(z[k][idx])
        bits.add(int(z["bits"])); hidden.add(int(z["hidden"]))
        kernels.add((SUB / src / "solution.py").read_bytes())
        build["models"] += [b["models"][i] for i in idx]
        build["cells"] += [b["cells"][i] for i in idx]
    if len(bits) != 1 or len(hidden) != 1 or len(kernels) != 1:
        raise ValueError("las entregas no comparten kernel, bits o relleno")
    out = {k: np.concatenate(v, 0) for k, v in arrays.items()}
    n = len(out["cell_per_model"])
    out.update(n_models=np.array(n), bits=np.array(bits.pop()), hidden=np.array(hidden.pop()),
               model_weights=np.full(n, 1.0 / n, np.float32))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "weights.npz", **out)
    shutil.copy(SUB / parts[0][0] / "solution.py", out_dir / "solution.py")
    build.update(weights=None, hidden=int(out["hidden"]), n_models=n, bits=int(out["bits"]),
                 merged_from=[[s, list(i)] for s, i in parts])
    (out_dir / "BUILD.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    if make_zip:
        with zipfile.ZipFile(out_dir.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_dir / "solution.py", "solution.py")
            zf.write(out_dir / "weights.npz", "weights.npz")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True, help="carpeta:índices, p. ej. ens4_mask_LR_LA_long_int8:0,1,3")
    ap.add_argument("--out", required=True, help="nombre de la entrega nueva (carpeta y zip en submissions/)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    parts = [(p.split(":")[0], [int(i) for i in p.split(":")[1].split(",")]) for p in args.part]
    n = merge(parts, SUB / args.out, not args.no_zip)
    print(f"{args.out}: {n} modelos")


if __name__ == "__main__":
    main()
