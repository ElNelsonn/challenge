r"""Tabla de combinaciones de ensambles EN LA LAPTOP, sin GPU, usando las predicciones de valid ya guardadas
(`valid_preds_best.npz` / `valid_preds_swa.npz` que deja el entrenador con --save-valid-preds).

Reemplaza a `eval_ensemble.py` para todo lo que sea "probar combinaciones" (aquello tardaba 22 min de GPU por tabla).
La primera vez construye un cache con las filas puntuadas de valid (~3,5M filas, 1 min); después es instantáneo.

Uso (desde challenge/):
  python model/submission/combine_local.py --models E=model/official_gpu/results/sprint2/results/distill_128/valid_preds_best.npz \
      LR=.../student_lr5e-4/valid_preds_swa.npz ... --max-combo 4 [--hidden E=128,LR=128] [--top 25]

`--hidden` (opcional) da el tamaño de cada modelo para estimar los µs/fila y decir si entra en los 60 min;
por defecto asume LSTM 128x2. Formato: NOMBRE=HIDDEN o NOMBRE=HIDDENgru.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
ROOT = CHALLENGE.parent
CACHE = CHALLENGE / "model" / "submission" / "results" / "valid_scored_cache.npz"


def build_cache():
    """Filas puntuadas de valid: targets + a qué secuencia/fila pertenecen."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    ys, seqs, rows = [], [], []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["seq_ix", "t0", "t1", "is_scored", "need_prediction"], use_threads=False)
        m = t["is_scored"].to_numpy().astype(bool) & t["need_prediction"].to_numpy().astype(bool)
        idx = np.flatnonzero(m)
        ys.append(np.column_stack([t["t0"].to_numpy()[idx], t["t1"].to_numpy()[idx]]).astype(np.float32))
        seqs.append(np.full(len(idx), t["seq_ix"].to_numpy()[0], np.int64))
        rows.append(idx.astype(np.int32))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, y=np.concatenate(ys), seq=np.concatenate(seqs), row=np.concatenate(rows))
    print(f"cache: {sum(len(r) for r in rows):,} filas puntuadas -> {CACHE}")


def wp(y, p, w):
    out = []
    for j in (0, 1):
        wj, yj, pj = w[:, j], y[:, j], p[:, j]
        s = wj.sum()
        my, mp = (wj * yj).sum() / s, (wj * pj).sum() / s
        cov = (wj * (yj - my) * (pj - mp)).sum() / s
        vy, vp = (wj * (yj - my) ** 2).sum() / s, (wj * (pj - mp) ** 2).sum() / s
        out.append(cov / np.sqrt(vy * vp) if vy > 1e-16 and vp > 1e-16 else 0.0)
    return (out[0] + out[1]) / 2, out[0], out[1]


def macs(hidden, cell):
    g = 3 if cell == "gru" else 4
    return g * hidden * (112 + hidden) + g * hidden * 2 * hidden + 2 * hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="NOMBRE=ruta/valid_preds_*.npz")
    ap.add_argument("--hidden", default="", help="NOMBRE=128,OTRO=192gru (default: 128 LSTM)")
    ap.add_argument("--max-combo", type=int, default=4)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    if args.rebuild_cache or not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    Y = np.clip(z["y"], -2, 2).astype(np.float64)
    W = np.abs(Y)
    seq, row = z["seq"], z["row"]
    useq, sidx = np.unique(seq, return_inverse=True)

    spec = {}
    for s in args.hidden.split(",") if args.hidden else []:
        n, v = s.split("=")
        spec[n] = (int(v.rstrip("gru").rstrip("lstm")), "gru" if v.endswith("gru") else "lstm")

    P, cost = {}, {}
    for m in args.models:
        name, path = m.split("=", 1)
        d = np.load(path)
        pos = {int(s): i for i, s in enumerate(d["seq_ix"])}
        take = np.array([pos[int(s)] for s in useq])[sidx]
        P[name] = np.clip(d["preds"][take, row].astype(np.float64), -2, 2)
        h, cell = spec.get(name, (128, "lstm"))
        cost[name] = macs(h, cell)

    names = list(P)
    rows_out = []
    for r in range(1, min(args.max_combo, len(names)) + 1):
        for c in itertools.combinations(names, r):
            g, a, b = wp(Y, np.mean([P[k] for k in c], axis=0), W)
            us = 8 + sum(cost[k] for k in c) / 1000 * 0.058        # calibrado con S7 = 35m05s reales
            rows_out.append(("+".join(c), g, a, b, us, us * 37.46e6 / 60e6))
    rows_out.sort(key=lambda r: -r[1])
    print(f"\n{'modelos':38s} {'WP':>8s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s} {'min':>6s}  entra")
    for r in rows_out[:args.top]:
        print(f"{r[0]:38s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.1f} {r[5]:6.1f}  {'sí' if r[5] < 50 else 'NO'}")
    print("\n-- mejor que entra, por cantidad de modelos")
    for k in range(1, args.max_combo + 1):
        fit = [r for r in rows_out if r[5] < 50 and r[0].count("+") == k - 1][:2]
        for r in fit:
            print(f"{k}: {r[0]:38s} {r[1]:.4f}  {r[5]:.1f} min")


if __name__ == "__main__":
    main()
