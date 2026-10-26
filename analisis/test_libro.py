"""Las columnas del libro estan ordenadas por profundidad, o no?

Nuestra nota de septiembre dice "Indice != nivel de profundidad". Si eso es cierto, entonces
cross_features() esta calculando basura: toma ap[:,0] - bp[:,0] como el spread, y si el indice 0 no es el
mejor nivel eso no es el spread sino la resta de dos niveles cualesquiera. Todo el sprint 9 (que dio cero)
habria medido features mal construidas, y las features REALES del libro nunca se habrian probado.

Esto se verifica en los datos y no se discute.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
            r"\wnn_connectome_starterpack\wnn_connectome_starterpack")
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402

IX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
BIDP = [IX[f"i0_p{j}"] for j in range(11)]
ASKP = [IX[f"i0_p{j}"] for j in range(11, 22)]
BIDV = [IX[f"i0_v{j}"] for j in range(11)]
ASKV = [IX[f"i0_v{j}"] for j in range(11, 22)]

pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
t = pf.read_row_group(0, columns=FEATURE_COLUMNS, use_threads=False)
x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
x[~np.isfinite(x)] = 0.0
bp, ap, bv, av = x[:, BIDP], x[:, ASKP], x[:, BIDV], x[:, ASKV]
print(f"secuencia de {len(x):,} filas\n")

print("=" * 78)
print("1. ESTAN ORDENADOS LOS PRECIOS?")
print("=" * 78)
print("  Un libro de ordenes real tiene los bids DESCENDENTES (el mejor primero) y los asks")
print("  ASCENDENTES. Si las columnas estan en orden de profundidad, esto se ve directo.\n")
bid_desc = np.mean(np.all(np.diff(bp, axis=1) <= 0, axis=1))
bid_asc = np.mean(np.all(np.diff(bp, axis=1) >= 0, axis=1))
ask_desc = np.mean(np.all(np.diff(ap, axis=1) <= 0, axis=1))
ask_asc = np.mean(np.all(np.diff(ap, axis=1) >= 0, axis=1))
print(f"  filas donde i0_p0..p10  (bids) son descendentes: {bid_desc:7.2%}")
print(f"  filas donde i0_p0..p10  (bids) son ascendentes:  {bid_asc:7.2%}")
print(f"  filas donde i0_p11..p21 (asks) son descendentes: {ask_desc:7.2%}")
print(f"  filas donde i0_p11..p21 (asks) son ascendentes:  {ask_asc:7.2%}")

print("\n" + "=" * 78)
print("2. EL 'SPREAD' QUE CALCULAMOS, ES UN SPREAD?")
print("=" * 78)
print("  En un libro real el mejor ask SIEMPRE esta por encima del mejor bid: spread > 0.\n")
spread_nuestro = ap[:, 0] - bp[:, 0]
print(f"  ap[:,0] - bp[:,0]  (lo que usa cross_features):")
print(f"      positivo en {np.mean(spread_nuestro > 0):7.2%} de las filas")
print(f"      media {spread_nuestro.mean():+.4f} | p10 {np.percentile(spread_nuestro,10):+.4f} "
      f"| p90 {np.percentile(spread_nuestro,90):+.4f}")
spread_real = ap.min(1) - bp.max(1)
print(f"\n  min(asks) - max(bids)  (el spread de verdad, ordenando):")
print(f"      positivo en {np.mean(spread_real > 0):7.2%} de las filas")
print(f"      media {spread_real.mean():+.4f} | p10 {np.percentile(spread_real,10):+.4f} "
      f"| p90 {np.percentile(spread_real,90):+.4f}")
print(f"\n  correlacion entre los dos: {np.corrcoef(spread_nuestro, spread_real)[0,1]:+.4f}")

print("\n" + "=" * 78)
print("3. DONDE ESTA EL MEJOR NIVEL, ENTONCES?")
print("=" * 78)
imax_b = np.argmax(bp, axis=1)
imin_a = np.argmin(ap, axis=1)
print("  indice que tiene el MEJOR bid (el precio mas alto), reparto sobre las filas:")
for i, c in zip(*np.unique(imax_b, return_counts=True)):
    print(f"      i0_p{i:<2d}: {c/len(x):6.2%}")
print("  indice que tiene el MEJOR ask (el precio mas bajo):")
for i, c in zip(*np.unique(imin_a, return_counts=True)):
    print(f"      i0_p{i+11:<2d}: {c/len(x):6.2%}")
print(f"\n  el mejor bid cambia de columna {np.mean(np.diff(imax_b) != 0):6.2%} de las filas")

print("\n" + "=" * 78)
print("4. LO MISMO PARA LOS VOLUMENES, Y EL DESEQUILIBRIO")
print("=" * 78)
d_nuestro = (bv[:, 0] - av[:, 0]) / (np.abs(bv[:, 0]) + np.abs(av[:, 0]) + 1.0)
# el desequilibrio del TOPE de verdad: volumen en el nivel de mejor precio
fil = np.arange(len(x))
bv_top, av_top = bv[fil, imax_b], av[fil, imin_a]
d_real = (bv_top - av_top) / (np.abs(bv_top) + np.abs(av_top) + 1.0)
print(f"  desequilibrio con indice 0 (lo que usamos): media {d_nuestro.mean():+.4f}")
print(f"  desequilibrio en el nivel de mejor precio:   media {d_real.mean():+.4f}")
print(f"  correlacion entre los dos: {np.corrcoef(d_nuestro, d_real)[0,1]:+.4f}")

print("\n" + "=" * 78)
print("VEREDICTO")
print("=" * 78)
if bid_desc > 0.9 or bid_asc > 0.9:
    print("  Las columnas SI estan ordenadas por profundidad: cross_features estaba bien.")
else:
    print("  Las columnas NO estan ordenadas por profundidad.")
    print("  => cross_features() calcula 'spread' y 'desequilibrio' sobre niveles arbitrarios.")
    print("  => el sprint 9 (que dio cero) midio features mal construidas.")
    print("  => las features REALES del libro (spread, microprecio, desequilibrio por profundidad)")
    print("     NUNCA se probaron en este proyecto.")
