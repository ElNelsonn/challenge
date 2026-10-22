"""Sprint 24 con los TRES que corrieron (div_a, div_c, div_d). div_d es el GRU: la receta mas distinta.

div_b quedo cancelado por falta de cuota. Con tres puedo medir la correlacion media dentro de la familia
nueva y compararla de frente con la de los cuatro viejos, en vez de con un solo par.
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
print(f"{len(Y):,} filas puntuadas de {len(np.unique(seq)):,} secuencias\n", flush=True)

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


def corr_media(ks):
    return float(np.mean([np.corrcoef(P[a][sub, 0], P[b][sub, 0])[0, 1]
                          for a, b in itertools.combinations(ks, 2)]))


print("=" * 64)
print("CORRELACION DENTRO DE CADA FAMILIA")
print("=" * 64)
rn, rv = corr_media(list(NUEVOS)), corr_media(list(VIEJOS))
print(f"  los 3 nuevos (div_a, div_c, div_d): {rn:.4f}")
print(f"  los 4 viejos:                      {rv:.4f}")
print(f"  bajo: {rn - rv:+.4f}")
for a, b in itertools.combinations(NUEVOS, 2):
    print(f"    {a}-{b}: {np.corrcoef(P[a][sub,0], P[b][sub,0])[0,1]:.4f}")

print("\npuntajes individuales:")
for k in list(NUEVOS) + list(VIEJOS):
    marca = " (70 % datos, 2 maestros)" if k in NUEVOS else ""
    print(f"  {k:8s} {wp(P[k]):.4f}{marca}")

print("\n" + "=" * 64)
print("LO QUE GANA PROMEDIAR, MEDIDO DE VERDAD (no proyectado)")
print("=" * 64)
e_n = wp(np.mean([P[k] for k in NUEVOS], 0))
s_n = max(wp(P[k]) for k in NUEVOS)
print(f"  promedio de los 3 nuevos: {e_n:.4f}  (el mejor solo {s_n:.4f} -> gana {e_n-s_n:+.4f})")
mejor3v = max(itertools.combinations(VIEJOS, 3), key=lambda c: wp(np.mean([P[k] for k in c], 0)))
e_v = wp(np.mean([P[k] for k in mejor3v], 0))
s_v = max(wp(P[k]) for k in mejor3v)
print(f"  promedio de 3 viejos ({'+'.join(mejor3v)}): {e_v:.4f}  (el mejor solo {s_v:.4f} -> gana {e_v-s_v:+.4f})")

print("\n" + "=" * 64)
print("MEZCLAR LAS DOS FAMILIAS: LA UNICA PREGUNTA QUE QUEDA")
print("=" * 64)
base = wp(np.mean([P[k] for k in VIEJOS], 0))
print(f"  nuestro ensamble actual (4 viejos): {base:.4f}  -> LB {base-0.0323:.4f}")
cru = float(np.mean([np.corrcoef(P[a][sub, 0], P[b][sub, 0])[0, 1] for a in NUEVOS for b in VIEJOS]))
print(f"  correlacion cruzada entre familias: {cru:.4f}")
todos = list(NUEVOS) + list(VIEJOS)
mejor, cual = -9, None
for n in range(3, 8):
    for c in itertools.combinations(todos, n):
        v = wp(np.mean([P[k] for k in c], 0))
        if v > mejor:
            mejor, cual = v, c
print(f"  mejor combinacion de las 7:         {mejor:.4f} = {'+'.join(cual)}")
print(f"  los 7 juntos:                       {wp(np.mean([P[k] for k in todos], 0)):.4f}")
print(f"\n  GANANCIA SOBRE LO QUE TENEMOS: {mejor-base:+.4f}  -> LB {mejor-0.0323:.4f}")

lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos = np.array(sorted(lb)); sco = np.array([lb[p] for p in pos])
for nombre, val in (("ahora", base), ("mejor de 7", mejor)):
    s_lb = val - 0.0323
    pu = ">420" if s_lb <= sco[-1] else int(round(np.interp(-s_lb, -sco, pos)))
    print(f"  {nombre:12s} valid {val:.4f} -> LB {s_lb:.4f} -> puesto ~{pu}")

# los 4 del ensamble tienen que entrar en el presupuesto de inferencia: 4 modelos = 41 min medidos.
print("\n  (recordar: el kernel aguanta 4 modelos LSTM/GRU 128x2 = 41 min de los 60)")
