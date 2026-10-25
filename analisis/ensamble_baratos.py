"""El ultimo rincon sin probar: REEMPLAZAR una de nuestras redes por un arbol o por la ESN.

Lo que ya estaba medido es AGREGARLOS como quinto modelo (no entran en presupuesto) y daba cero o negativo.
Lo que nunca probe es sacar una red y poner uno de ellos en su lugar, que es distinto porque:
  - correlacionan muchisimo mas bajo que cualquier red (arboles 0.927, ESN 0.876 contra 0.98 de las redes)
  - son mas baratos en inferencia, asi que liberan minutos

Limitacion honesta: los arboles y la ESN solo tienen predicciones de ~200 secuencias, asi que todo esto se
mide en esa interseccion y con mucho mas ruido que las mediciones de 1873 secuencias.
"""
import itertools
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")

REDES = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
         "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
         "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
         "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz",
         "mask": R / "s15_notebookfd82629297/results/student_mask/valid_preds_swa.npz",
         "ad8000": SC / "k23b/a8000/results/adapt8000/valid_preds_swa.npz"}
XGB = SC / "k18/results/xgb_v1/valid_predictions.npz"
ESN = SC / "esn_preds.npz"

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Yf = np.clip(z["y"], -2, 2).astype(np.float64)
orden = np.argsort(seq, kind="stable")
tramos = {int(seq[t[0]]): t for t in np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)}

# que secuencias tienen TODO
dx = np.load(XGB)
seqs_x = {int(k) for k in dx.files}
de = np.load(ESN)
seqs_e = set(np.unique(de["seq"]).tolist())
comun = sorted(seqs_x & seqs_e & set(tramos))
print(f"xgboost: {len(seqs_x)} secuencias | ESN: {len(seqs_e)} | en comun con valid: {len(comun)}")
filas = np.sort(np.concatenate([tramos[s] for s in comun]))
Y = Yf[filas]
W = np.abs(Y)
print(f"{len(Y):,} filas puntuadas en la interseccion\n", flush=True)

P = {}
for k, p in REDES.items():
    d = np.load(p)
    pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    preds = d["preds"]
    out = np.zeros((len(Yf), 2), np.float32)
    for s in comun:
        t = tramos[s]
        out[t] = preds[pos[s]][row[t]]
    P[k] = np.clip(out[filas].astype(np.float64), -2, 2)

out = np.zeros((len(Yf), 2), np.float32)
for s in comun:
    t = tramos[s]
    out[t] = dx[str(s)][row[t]]
P["xgb"] = np.clip(out[filas].astype(np.float64), -2, 2)

out = np.zeros((len(Yf), 2), np.float32)
pe = de["preds"].astype(np.float32)
clave = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(de["seq"], de["row"]))}
for s in comun:
    t = tramos[s]
    idx = [clave.get((s, int(r)), -1) for r in row[t]]
    ok = np.array(idx) >= 0
    tt = t[ok]
    out[tt] = pe[np.array(idx)[ok]]
P["esn"] = np.clip(out[filas].astype(np.float64), -2, 2)
print("cargados:", ", ".join(P), flush=True)


def wp(q):
    v = []
    for j in (0, 1):
        w, y, p_ = W[:, j], Y[:, j], q[:, j]
        s = w.sum()
        my, mp = (w * y).sum() / s, (w * p_).sum() / s
        cov = (w * (y - my) * (p_ - mp)).sum() / s
        vy, vp = (w * (y - my) ** 2).sum() / s, (w * (p_ - mp) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vp))
    return (v[0] + v[1]) / 2


NUESTRO = ["LR", "LA", "long", "b256"]
S = wp(np.mean([P[k] for k in NUESTRO], 0))
ENS = np.mean([P[k] for k in NUESTRO], 0)
sub = slice(None, None, 3)
print("\n" + "=" * 80)
print(f"EN ESTAS {len(comun)} SECUENCIAS nuestro ensamble da {S:.4f}")
print("=" * 80)
print(f"\n{'modelo':10s} {'solo':>8s} {'r con el ensamble':>20s}")
for k in P:
    r = float(np.corrcoef(P[k][sub, 0], ENS[sub, 0])[0, 1])
    print(f"  {k:8s} {wp(P[k]):8.4f} {r:20.4f}")

print("\n" + "=" * 80)
print("REEMPLAZAR UNA DE NUESTRAS CUATRO POR EL ARBOL O POR LA ESN")
print("=" * 80)
res = []
for fuera in NUESTRO:
    for dentro in ("xgb", "esn"):
        c = [k for k in NUESTRO if k != fuera] + [dentro]
        res.append((wp(np.mean([P[k] for k in c], 0)), f"-{fuera} +{dentro}"))
for v, etq in sorted(res, reverse=True):
    print(f"  {v:.4f}  ({v-S:+.4f})  {etq}")

print("\n" + "=" * 80)
print("Y LA MEJOR COMBINACION DE CUALQUIER TAMANO ENTRE LOS 8 (incluye las dos ramas raras)")
print("=" * 80)
todos = list(P)
mejor, cual = -9, None
for n in range(2, 7):
    for c in itertools.combinations(todos, n):
        v = wp(np.mean([P[k] for k in c], 0))
        if v > mejor:
            mejor, cual = v, c
print(f"  {mejor:.4f}  {'+'.join(cual)}   ({mejor-S:+.4f} sobre nuestro {S:.4f})")
print(f"\n  OJO: son {len(comun)} secuencias, una decima parte de valid. Una diferencia de +-0.002 aca")
print("  no se distingue del ruido, y la combinacion se eligio mirando estas mismas filas.")
