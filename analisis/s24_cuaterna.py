"""div_d (el GRU) correlaciona 0.977 con los otros dos nuevos, contra 0.985 del par LSTM. El cambio de
ARQUITECTURA bajo la correlacion; el de datos y maestros no.

Pero el presupuesto de inferencia aguanta CUATRO modelos (41 min de los 60), no cinco. Asi que la unica
pregunta que importa es: hay alguna CUATERNA de los 7 que le gane a la nuestra?

Y la regla que venimos usando todo el proyecto: un modelo nuevo aporta si  s > S*r  (s = su puntaje,
S = el del ensamble, r = su correlacion CON EL ENSAMBLE, no con los modelos sueltos).
"""
import itertools
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")

NUEVOS = {"div_a": SC / "k24b/a/results/div_a/valid_preds_swa.npz",
          "div_c": SC / "k24b/c/results/div_c/valid_preds_swa.npz",
          "div_d": SC / "k24b/d/results/div_d/valid_preds_swa.npz"}
VIEJOS = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
          "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
          "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
          "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Y = np.clip(z["y"], -2, 2).astype(np.float64)
W = np.abs(Y)
orden = np.argsort(seq, kind="stable")
tramos = np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)

P = {}
for grupo in (NUEVOS, VIEJOS):
    for k, p in grupo.items():
        d = np.load(p)
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
        preds = d["preds"]
        out = np.zeros((len(Y), 2), np.float32)
        for tr in tramos:
            s_ = int(seq[tr[0]])
            if s_ in pos:
                out[tr] = preds[pos[s_]][row[tr]]
        P[k] = np.clip(out.astype(np.float64), -2, 2)
print(f"{len(Y):,} filas puntuadas\n", flush=True)


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
lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos_, sco = np.array(sorted(lb)), np.array([lb[p] for p in sorted(lb)])


def puesto(v):
    x = v - 0.0323
    return ">420" if x <= sco[-1] else int(round(np.interp(-x, -sco, pos_)))


print("=" * 70)
print(f"LA REGLA s > S*r   (ensamble actual S = {S:.4f})")
print("=" * 70)
print(f"{'modelo':8s} {'s':>8s} {'r con ens':>11s} {'S*r':>8s} {'margen':>9s}   veredicto")
for k in NUEVOS:
    s_ = wp(P[k])
    r = float(np.corrcoef(P[k][sub, 0], ENS[sub, 0])[0, 1])
    print(f"{k:8s} {s_:8.4f} {r:11.4f} {S*r:8.4f} {s_-S*r:+9.4f}   "
          f"{'PASA' if s_ > S*r else 'no pasa'}")

print("\n" + "=" * 70)
print("LA PREGUNTA QUE DECIDE: MEJOR CUATERNA DE LOS 7 (4 = lo que entra en 60 min)")
print("=" * 70)
todos = list(NUEVOS) + list(VIEJOS)
res = sorted(((wp(np.mean([P[k] for k in c], 0)), c) for c in itertools.combinations(todos, 4)),
             reverse=True)
for v, c in res[:6]:
    nuevos_en = sum(1 for k in c if k in NUEVOS)
    print(f"  {v:.4f}  ({nuevos_en} nuevos)  {'+'.join(c)}")
print(f"  ...")
print(f"  {S:.4f}  (0 nuevos)  LR+LA+long+b256   <- EL NUESTRO")
mejor, cual = res[0]
print(f"\n  mejor cuaterna:   {mejor:.4f} -> LB {mejor-0.0323:.4f} -> puesto ~{puesto(mejor)}")
print(f"  el nuestro:       {S:.4f} -> LB {S-0.0323:.4f} -> puesto ~{puesto(S)}")
print(f"  GANANCIA REAL:   {mejor-S:+.4f}")

print("\n" + "=" * 70)
print("ULTIMO INTENTO: PESOS NO UNIFORMES EN LA MEJOR CUATERNA Y EN LOS 7")
print("=" * 70)
rng = np.random.default_rng(2026)
for nombre, ks in (("mejor cuaterna", list(cual)), ("los 7", todos)):
    base = wp(np.mean([P[k] for k in ks], 0))
    mejor_w, w_best = base, np.ones(len(ks)) / len(ks)
    M = np.stack([P[k] for k in ks])
    for _ in range(300):
        w = rng.dirichlet(np.ones(len(ks)) * 4.0)
        v = wp(np.tensordot(w, M, axes=1))
        if v > mejor_w:
            mejor_w, w_best = v, w
    print(f"  {nombre:15s} uniforme {base:.4f} -> mejor peso {mejor_w:.4f} ({mejor_w-base:+.4f})")
    print(f"                  pesos: {dict(zip(ks, np.round(w_best, 3)))}")
print("\n  (los pesos se eligen mirando el MISMO valid con el que se miden: sobreestiman)")
