"""EL NUMERO MAS GRANDE DEL PROYECTO: se puede predecir la escala de cada secuencia desde las features?

En septiembre medimos que si supieramos la escala de cada secuencia, el ensamble pasaria de 0.6853 a 0.7053:
+0.047, seis veces la distancia al puesto 1. Y lo cerre como imposible, pero probe UNA sola cosa: estimar la
escala a partir de las PROPIAS PREDICCIONES. Nunca probe lo obvio: estimarla desde las FEATURES.

Es un problema chico: un numero por secuencia. Aca mido si tiene senal, en dos regimenes:
  - resumen de TODA la secuencia  -> cota superior, no es causal, solo dice si la informacion existe
  - resumen de las primeras 2000 filas -> lo que de verdad tendria un modelo causal temprano

Objetivo: log de la desviacion ponderada de y por secuencia (una por target). Si esto no se puede predecir,
la rama esta muerta de verdad y el sprint 23 es un tiro largo. Si se puede, hay +0.047 para ir a buscar.

Uso:  python -I escala_predecible.py [n_secuencias]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
            r"\wnn_connectome_starterpack\wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CORTE = 2000          # filas iniciales para el resumen "causal"

pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
total = pf.num_row_groups
rng = np.random.default_rng(2026)
grupos = sorted(rng.choice(total, size=min(N, total), replace=False).tolist())
print(f"{len(grupos)} de {total} secuencias de valid | corte causal: primeras {CORTE} filas\n", flush=True)

cols = [*FEATURE_COLUMNS, *TARGET_COLUMNS, "is_scored"]
Xf, Xc, Sy = [], [], []
t0 = time.time()
for n, g in enumerate(grupos):
    t = pf.read_row_group(g, columns=cols, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
    y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float64)
    sc = t["is_scored"].to_numpy().astype(bool)
    x[~np.isfinite(x)] = 0.0

    yc = np.clip(y[sc], -2, 2)
    w = np.abs(yc)
    sig = []
    for j in (0, 1):
        ww, yy = w[:, j], yc[:, j]
        s = ww.sum()
        if s <= 0:
            sig.append(np.nan); continue
        m = (ww * yy).sum() / s
        sig.append(np.sqrt(max((ww * (yy - m) ** 2).sum() / s, 1e-12)))
    if not np.all(np.isfinite(sig)):
        continue
    Sy.append(sig)
    # resumen: media y desvio de cada feature. Dos versiones.
    Xf.append(np.concatenate([x.mean(0), x.std(0)]))
    xa = x[:CORTE]
    Xc.append(np.concatenate([xa.mean(0), xa.std(0)]))
    if (n + 1) % 100 == 0:
        print(f"  {n+1}/{len(grupos)}  ({(time.time()-t0)/(n+1):.2f} s por secuencia)", flush=True)

Xf, Xc, Sy = np.array(Xf), np.array(Xc), np.array(Sy)
print(f"\nlistas {len(Sy)} secuencias en {time.time()-t0:.0f} s | resumen: {Xf.shape[1]} numeros\n")

print("=" * 70)
print("CUANTO VARIA LA ESCALA DEL TARGET ENTRE SECUENCIAS (lo que habria para ganar)")
print("=" * 70)
for j, nom in enumerate(TARGET_COLUMNS):
    s = Sy[:, j]
    p10, p50, p90 = np.percentile(s, [10, 50, 90])
    print(f"  {nom}: mediana {p50:.4f} | p10 {p10:.4f} | p90 {p90:.4f} | p90/p10 = {p90/p10:.2f}x")


def r2_ridge(X, y, lam=10.0, semilla=0):
    """R2 fuera de muestra, particion 70/30 POR SECUENCIA, con estandarizacion ajustada solo en train."""
    rs = np.random.default_rng(semilla)
    idx = rs.permutation(len(y))
    k = int(0.7 * len(y))
    tr, te = idx[:k], idx[k:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    A = (X[tr] - mu) / sd
    B = (X[te] - mu) / sd
    ym = y[tr].mean()
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
    pred = B @ coef + ym
    return 1.0 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum(), pred, y[te]


print("\n" + "=" * 70)
print("SE PUEDE PREDECIR?  (R2 fuera de muestra sobre log de la escala)")
print("=" * 70)
print("  R2 = 0 significa 'no mejor que decir siempre la media'. R2 = 1 seria perfecto.\n")
for nombre, X in (("toda la secuencia (cota superior, NO causal)", Xf),
                  (f"primeras {CORTE} filas (causal)", Xc)):
    print(f"  {nombre}:")
    for j, nom in enumerate(TARGET_COLUMNS):
        ly = np.log(Sy[:, j])
        r2s = [r2_ridge(X, ly, lam, 0)[0] for lam in (1.0, 10.0, 100.0, 1000.0)]
        mejor = max(r2s)
        # estabilidad: el mismo lambda con tres particiones distintas
        lam = (1.0, 10.0, 100.0, 1000.0)[int(np.argmax(r2s))]
        rep = [r2_ridge(X, ly, lam, s)[0] for s in (0, 1, 2)]
        print(f"    {nom}: R2 = {mejor:+.3f}  (lambda {lam:g} | otras particiones "
              f"{rep[1]:+.3f} {rep[2]:+.3f})")
    print()

print("=" * 70)
print("REFERENCIA: QUE SIGNIFICA CADA R2 EN PUNTOS")
print("=" * 70)
print("  El oraculo (escala exacta) vale +0.047 sobre el ensamble.")
print("  Si R2 = 0.5, se captura mas o menos la mitad de la varianza -> una parte de ese +0.047.")
print("  Para saber CUANTO exactamente hay que aplicarlo a las predicciones y medir el WP:")
print("  eso es el paso 2, y solo vale la pena si estos R2 no son ~0.")
