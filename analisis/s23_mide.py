"""Sprint 23: adapt8000 dio 0.6794 solo, por debajo de nuestros cuatro (0.6823-0.6843). Pero el puntaje
individual NO decide: el 2026-10-22 medimos que la regla s > S*r falla y que lo unico que vale es promediar
y medir. La normalizacion adaptativa cambia QUE VE la red, asi que es el candidato con mas chance de
correlacionar bajo de todos los que probamos.

Mide: correlacion con el ensamble y con cada modelo, y TODAS las cuaternas de los 5 (cuatro es lo que entra
en los 60 min de inferencia). Tambien con los tres del sprint 24, que estan medidos y disponibles.
"""
import itertools
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")

VIEJOS = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
          "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
          "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
          "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}
NUEVOS = {"ad8000_swa": SC / "k23b/a8000/results/adapt8000/valid_preds_swa.npz",
          "ad8000_best": SC / "k23b/a8000/results/adapt8000/valid_preds_best.npz",
          "div_a": SC / "k24b/a/results/div_a/valid_preds_swa.npz",
          "div_c": SC / "k24b/c/results/div_c/valid_preds_swa.npz",
          "div_d": SC / "k24b/d/results/div_d/valid_preds_swa.npz"}

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Y = np.clip(z["y"], -2, 2).astype(np.float64)
W = np.abs(Y)
orden = np.argsort(seq, kind="stable")
tramos = np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)
print(f"{len(Y):,} filas puntuadas de {len(tramos):,} secuencias\n", flush=True)

P = {}
for grupo in (VIEJOS, NUEVOS):
    for k, p in grupo.items():
        if not p.exists():
            print(f"  (falta {k})")
            continue
        d = np.load(p)
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
        preds = d["preds"]
        out = np.zeros((len(Y), 2), np.float32)
        for t in tramos:
            s_ = int(seq[t[0]])
            if s_ in pos:
                out[t] = preds[pos[s_]][row[t]]
        P[k] = np.clip(out.astype(np.float64), -2, 2)
        print(f"  {k} listo", flush=True)


def wp(p):
    v = []
    for j in (0, 1):
        w, y, q = W[:, j], Y[:, j], p[:, j]
        s = w.sum()
        my, mq = (w * y).sum() / s, (w * q).sum() / s
        cov = (w * (y - my) * (q - mq)).sum() / s
        vy, vq = (w * (y - my) ** 2).sum() / s, (w * (q - mq) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vq))
    return (v[0] + v[1]) / 2


sub = slice(None, None, 5)
ENS = np.mean([P[k] for k in VIEJOS], 0)
S = wp(ENS)
print("\n" + "=" * 74)
print(f"EL ENSAMBLE ACTUAL: {S:.4f}")
print("=" * 74)

print("\npuntajes y correlacion con el ensamble:")
for k in list(VIEJOS) + [k for k in NUEVOS if k in P]:
    r = float(np.corrcoef(P[k][sub, 0], ENS[sub, 0])[0, 1])
    print(f"  {k:12s} {wp(P[k]):.4f}   r con el ensamble {r:.4f}")

ad = "ad8000_swa" if "ad8000_swa" in P else None
if ad:
    print(f"\ncorrelacion de la normalizacion adaptativa con cada modelo:")
    for k in list(VIEJOS) + ["div_a", "div_d"]:
        if k in P:
            print(f"  {ad} - {k:8s}: {np.corrcoef(P[ad][sub,0], P[k][sub,0])[0,1]:.4f}")
    print(f"  (referencia: los 4 viejos entre si 0.9860 | div_d con los LSTM 0.977)")

print("\n" + "=" * 74)
print("TODAS LAS CUATERNAS (4 = lo que entra en los 60 min de inferencia)")
print("=" * 74)
todos = [k for k in list(VIEJOS) + list(NUEVOS) if k in P and k != "ad8000_best"]
res = sorted(((wp(np.mean([P[k] for k in c], 0)), c) for c in itertools.combinations(todos, 4)),
             reverse=True)
for v, c in res[:8]:
    marca = " <- EL NUESTRO" if set(c) == set(VIEJOS) else ""
    print(f"  {v:.4f}  {'+'.join(c)}{marca}")
mejor, cual = res[0]
pos_n = [i for i, (v, c) in enumerate(res) if set(c) == set(VIEJOS)][0]
print(f"\n  el nuestro esta en el puesto {pos_n+1} de {len(res)} cuaternas")
print(f"  mejor: {mejor:.4f} ({'+'.join(cual)})")
print(f"  ganancia sobre el nuestro: {mejor-S:+.4f}")

print("\n" + "=" * 74)
print("Y SI ENTRARAN CINCO (no entra en presupuesto, solo para saber si hay algo)")
print("=" * 74)
r5 = sorted(((wp(np.mean([P[k] for k in c], 0)), c) for c in itertools.combinations(todos, 5)),
            reverse=True)[:3]
for v, c in r5:
    print(f"  {v:.4f}  {'+'.join(c)}  ({v-S:+.4f})")

lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos_, sco = np.array(sorted(lb)), np.array([lb[p] for p in sorted(lb)])
print("\n" + "=" * 74)
for nombre, val in (("ahora", S), ("mejor cuaterna", mejor)):
    x = val - 0.0323
    pu = ">420" if x <= sco[-1] else int(round(np.interp(-x, -sco, pos_)))
    print(f"  {nombre:16s} valid {val:.4f} -> LB {x:.4f} -> puesto ~{pu}")
print("=" * 74)
