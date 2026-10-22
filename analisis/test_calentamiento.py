"""¿El ensamble predice peor al principio de cada secuencia?

Cada secuencia arranca de cero: el estado de la red se reinicia y hay 99 filas de calentamiento. Si a la red
le cuesta "agarrar el ritmo", las primeras filas puntuadas de cada secuencia deberían salir peor que las del
final. Nunca lo miramos.

Si la diferencia es grande, hay arreglos baratos y concretos: pasar el calentamiento dos veces por la red,
aprender el estado inicial en vez de arrancar en cero, o directamente darle menos peso a esa zona.
"""
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Y = np.clip(z["y"], -2, 2).astype(np.float64)
W = np.abs(Y)
print(f"{len(Y):,} filas puntuadas, {len(np.unique(seq)):,} secuencias")

MOD = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
       "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
       "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
       "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}
acum = None
for k, p in MOD.items():
    d = np.load(p)
    pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    preds = d["preds"]
    out = np.zeros((len(Y), 2), np.float32)
    orden = np.argsort(seq, kind="stable")
    bordes = np.flatnonzero(np.diff(seq[orden])) + 1
    for tramo in np.split(orden, bordes):
        s_ = int(seq[tramo[0]])
        out[tramo] = preds[pos[s_]][row[tramo]]
    acum = out.astype(np.float64) if acum is None else acum + out
    print(f"  {k} listo", flush=True)
ENS = np.clip(acum / len(MOD), -2, 2)


def wp(m):
    v = []
    for j in (0, 1):
        w, y, q = W[m, j], Y[m, j], ENS[m, j]
        s = w.sum()
        my, mq = (w * y).sum() / s, (w * q).sum() / s
        cov = (w * (y - my) * (q - mq)).sum() / s
        vy, vq = (w * (y - my) ** 2).sum() / s, (w * (q - mq) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vq))
    return (v[0] + v[1]) / 2


print(f"\nensamble global: {wp(slice(None)):.4f}\n")
print("POR POSICION DENTRO DE LA SECUENCIA (la secuencia tiene 20.000 filas)")
print(f"  {'desde':>7s} {'hasta':>7s} {'filas':>9s} {'puntaje':>9s}")
cortes = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000),
          (5000, 10000), (10000, 15000), (15000, 20000)]
for a, b in cortes:
    m = (row >= a) & (row < b)
    if m.sum() < 500:
        continue
    print(f"  {a:>7,} {b:>7,} {m.sum():>9,} {wp(m):9.4f}")

print("\nEL ARRANQUE EN DETALLE (primeras 500 filas)")
print(f"  {'desde':>7s} {'hasta':>7s} {'filas':>9s} {'puntaje':>9s}")
for a, b in ((0, 50), (50, 100), (100, 150), (150, 250), (250, 500)):
    m = (row >= a) & (row < b)
    if m.sum() < 200:
        print(f"  {a:>7,} {b:>7,} {m.sum():>9,}    (pocas filas)")
        continue
    print(f"  {a:>7,} {b:>7,} {m.sum():>9,} {wp(m):9.4f}")

prim = (row < 2000)
print(f"\nprimeras 2.000 filas: {wp(prim):.4f}  ({prim.sum():,} filas, {100*prim.mean():.1f} % del total)")
print(f"resto:                {wp(~prim):.4f}")
print(f"diferencia:           {wp(prim) - wp(~prim):+.4f}")
