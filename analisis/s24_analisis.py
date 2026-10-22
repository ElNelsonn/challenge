"""Sprint 24: ¿bajó la correlación? El número que decide todo.

Nuestros 4 modelos actuales están destilados de los MISMOS 4 maestros, correlacionan 0.9854 y promediarlos
vale apenas +0.0021. div_a y div_c vieron datos distintos (70 % cada uno), se destilaron de PARES distintos
de maestros y usan recetas distintas.

Si su correlación baja a ~0.95, promediar cuatro de estos vale ~+0.009 en vez de +0.0021, el ensamble se va
a ~0.690, el leaderboard a ~0.658 y eso es el puesto ~235 desde el ~410 actual.
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
          "div_c": SC / "k24b/c/results/div_c/valid_preds_swa.npz"}
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
print("\n" + "=" * 62)
print("EL NUMERO QUE DECIDE")
print("=" * 62)
r_nuevo = float(np.corrcoef(P["div_a"][sub, 0], P["div_c"][sub, 0])[0, 1])
rs_viejo = [np.corrcoef(P[a][sub, 0], P[b][sub, 0])[0, 1] for a, b in itertools.combinations(VIEJOS, 2)]
print(f"  correlacion entre div_a y div_c:        {r_nuevo:.4f}")
print(f"  correlacion entre los 4 viejos (media): {np.mean(rs_viejo):.4f}")
print(f"  bajo: {r_nuevo - np.mean(rs_viejo):+.4f}")

print("\npuntajes individuales:")
for k in ("div_a", "div_c"):
    print(f"  {k:8s} {wp(P[k]):.4f}  (70 % de los datos, 2 maestros)")
for k in VIEJOS:
    print(f"  {k:8s} {wp(P[k]):.4f}")

par_n = wp((P["div_a"] + P["div_c"]) / 2)
mejor_n = max(wp(P["div_a"]), wp(P["div_c"]))
print(f"\npromedio de los DOS nuevos:  {par_n:.4f}  (el mejor solo {mejor_n:.4f} -> gana {par_n-mejor_n:+.4f})")
mejores_pares = [(wp((P[a] + P[b]) / 2) - max(wp(P[a]), wp(P[b])), a, b)
                 for a, b in itertools.combinations(VIEJOS, 2)]
g = float(np.mean([x[0] for x in mejores_pares]))
print(f"promedio de DOS viejos:      gana {g:+.4f} en promedio sobre el mejor de los dos")

print("\n" + "=" * 62)
print("PROYECCION A 4 MODELOS")
print("=" * 62)
lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos = np.array(sorted(lb)); sco = np.array([lb[p] for p in pos])


def puesto(s):
    return ">420" if s <= sco[-1] else int(round(np.interp(-s, -sco, pos)))


s_ind = (wp(P["div_a"]) + wp(P["div_c"])) / 2
for r in (r_nuevo, np.mean(rs_viejo)):
    e4 = s_ind * np.sqrt(4 / (1 + 3 * r))
    e4_corr = e4 - 0.004          # la formula sobreestima ~0.004, medido en las dos familias
    print(f"  con correlacion {r:.4f}: ensamble de 4 = {e4_corr:.4f} -> LB {e4_corr-0.0323:.4f} "
          f"-> puesto ~{puesto(e4_corr-0.0323)}")
print(f"\n  nuestro ensamble actual: 0.6864 -> LB 0.6540 -> puesto ~410")

print("\nmezclar las dos familias (lo mejor de cada una):")
todos = list(NUEVOS) + list(VIEJOS)
mejor, cual = -9, None
for n in (3, 4, 5):
    for c in itertools.combinations(todos, n):
        v = wp(np.mean([P[k] for k in c], 0))
        if v > mejor:
            mejor, cual = v, c
base = wp(np.mean([P[k] for k in VIEJOS], 0))
print(f"  mejor combinacion: {'+'.join(cual)} = {mejor:.4f}")
print(f"  nuestro ensamble de 4 viejos:  {base:.4f}")
print(f"  diferencia: {mejor-base:+.4f} -> LB {mejor-0.0323:.4f} -> puesto ~{puesto(mejor-0.0323)}")
