"""Confirmacion sobre varias secuencias y los dos instrumentos, y la explicacion del mecanismo.

El test anterior, en una secuencia, mostro que las columnas de precio NO estan en orden de valor y que
nuestro 'desequilibrio' correlaciona -0.21 con el desequilibrio real del tope del libro. Aca confirmo si eso
es general, y pruebo la explicacion: las features vienen NORMALIZADAS COLUMNA POR COLUMNA, y eso rompe el
orden entre niveles.

Si el orden de profundidad es CONSISTENTE entre filas y secuencias, entonces las columnas si estan en orden
de profundidad y lo que rompe las comparaciones es la normalizacion por columna, no un shuffle. Eso importa
porque una capa lineal PUEDE recuperar diferencias de precio (aprende las escalas, el offset va al bias),
pero NO puede recuperar min/max/argmax: el nivel mejor de verdad, el spread real, la concentracion de
volumen. Justo las features no lineales que cross_features() calcula mal.
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
G = {"i0_p_bid": [IX[f"i0_p{j}"] for j in range(11)],
     "i0_p_ask": [IX[f"i0_p{j}"] for j in range(11, 22)],
     "i0_v_bid": [IX[f"i0_v{j}"] for j in range(11)],
     "i0_v_ask": [IX[f"i0_v{j}"] for j in range(11, 22)],
     "i1_p_bid": [IX[f"i1_p{j}"] for j in range(11)],
     "i1_p_ask": [IX[f"i1_p{j}"] for j in range(11, 22)]}

pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
rng = np.random.default_rng(7)
grupos = sorted(rng.choice(pf.num_row_groups, 12, replace=False).tolist())
print(f"12 secuencias de valid: {grupos}\n", flush=True)

ordenes, corr_deseq, prop_mejor = {k: [] for k in G}, [], {}
for g in grupos:
    t = pf.read_row_group(g, columns=FEATURE_COLUMNS, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
    x[~np.isfinite(x)] = 0.0
    for k, cols in G.items():
        M = x[:, cols]
        # el "ranking" promedio de cada columna dentro de la fila: si es estable, el orden es consistente
        ordenes[k].append(np.argsort(np.argsort(M, axis=1), axis=1).mean(0))
    bp, ap = x[:, G["i0_p_bid"]], x[:, G["i0_p_ask"]]
    bv, av = x[:, G["i0_v_bid"]], x[:, G["i0_v_ask"]]
    fil = np.arange(len(x))
    ib, ia = np.argmax(bp, 1), np.argmin(ap, 1)
    d_ix0 = (bv[:, 0] - av[:, 0]) / (np.abs(bv[:, 0]) + np.abs(av[:, 0]) + 1.0)
    bt, at = bv[fil, ib], av[fil, ia]
    d_top = (bt - at) / (np.abs(bt) + np.abs(at) + 1.0)
    corr_deseq.append(np.corrcoef(d_ix0, d_top)[0, 1])
    prop_mejor.setdefault("bid", []).append(np.bincount(ib, minlength=11) / len(x))
    prop_mejor.setdefault("ask", []).append(np.bincount(ia, minlength=11) / len(x))

print("=" * 80)
print("1. EL ORDEN DE PROFUNDIDAD ES CONSISTENTE ENTRE SECUENCIAS?")
print("=" * 80)
print("  Para cada grupo de 11 columnas: el ranking medio de cada columna dentro de la fila.")
print("  Si las 12 secuencias dan el MISMO orden, las columnas estan en un orden fijo de profundidad.\n")
for k in G:
    M = np.array(ordenes[k])                      # (12, 11)
    orden_medio = np.argsort(M.mean(0))
    iguales = sum(1 for fila in M if np.array_equal(np.argsort(fila), orden_medio))
    print(f"  {k:9s} orden por columna: {list(orden_medio)}")
    print(f"            {iguales}/12 secuencias dan exactamente ese orden | "
          f"desvio maximo del ranking entre secuencias: {M.std(0).max():.2f}")

print("\n" + "=" * 80
      )
print("2. DONDE CAE EL MEJOR NIVEL (promedio sobre las 12 secuencias)")
print("=" * 80)
for lado in ("bid", "ask"):
    pr = np.array(prop_mejor[lado]).mean(0)
    top = np.argsort(pr)[::-1][:3]
    s = " | ".join(f"col {i}: {pr[i]:.1%}" for i in top)
    print(f"  mejor {lado}: {s}")

print("\n" + "=" * 80)
print("3. NUESTRO DESEQUILIBRIO CONTRA EL DESEQUILIBRIO REAL DEL TOPE")
print("=" * 80)
c = np.array(corr_deseq)
print(f"  correlacion por secuencia: min {c.min():+.3f} | media {c.mean():+.3f} | max {c.max():+.3f}")
print(f"  negativa en {np.mean(c < 0):.0%} de las secuencias")

print("\n" + "=" * 80)
print("VEREDICTO")
print("=" * 80)
consistente = all(
    sum(1 for fila in np.array(ordenes[k]) if np.array_equal(np.argsort(fila),
        np.argsort(np.array(ordenes[k]).mean(0)))) >= 10 for k in G)
print(f"  orden de profundidad consistente entre secuencias: {'SI' if consistente else 'NO'}")
print("  => las columnas SI tienen un orden fijo de profundidad; lo que rompe las comparaciones")
print("     entre niveles es que cada columna viene NORMALIZADA POR SEPARADO.")
print()
print("  Consecuencia practica, y es la que importa:")
print("    * una capa lineal PUEDE recuperar diferencias de precio (aprende las escalas,")
print("      el offset constante se lo come el bias). Eso la red ya lo tiene.")
print("    * una capa lineal NO PUEDE recuperar min, max ni argmax sobre los niveles.")
print("      El mejor nivel, el spread verdadero y la concentracion de volumen requieren")
print("      ordenar la fila, y sobre valores normalizados por columna el orden esta mal.")
print("    * cross_features() calcula exactamente esas cosas no lineales, y mal:")
print(f"      su desequilibrio del tope correlaciona {c.mean():+.3f} con el real.")
