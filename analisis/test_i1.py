"""¿Queda señal sin usar en el instrumento i1, que es el "correlacionado"?

El corrector con árboles buscó a ciegas en las 190 columnas y no encontró casi nada. Pero una búsqueda a
ciegas con árboles sobre datos muy ruidosos puede perderse una relación que sí existe si es chica y lineal.

Hay una hipótesis concreta que nunca medimos de frente: el enunciado dice que i1 es un instrumento
CORRELACIONADO con i0. En los mercados eso suele significar que uno se mueve antes que el otro (lead-lag). Si
i1 se adelanta a i0, los cambios de i1 de hace k filas deberían predecir el target de i0 ahora.

Se mide directo sobre el ERROR del ensamble (lo que las redes todavía no explican), que es lo único que
importa: si las redes ya lo usan, el error no va a correlacionar con eso.
"""
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"
import sys
sys.path.insert(0, str(CH / "model" / "lgbm_deltas"))
import train_lgbm_deltas_v1 as L   # noqa: E402

DATOS = CH.parent / "datasets" / "valid.parquet"
LAGS = [1, 2, 5, 10, 20, 50, 100, 200]
N_SEQ = 120

grupos = L.ParquetGroups([str(DATOS)])
MOD = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
       "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
       "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
       "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}
acum, pos = None, None
for k, p in MOD.items():
    d = np.load(p)
    if pos is None:
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    pr = d["preds"].astype(np.float32)
    acum = pr if acum is None else acum + pr
ENS = acum / len(MOD)

COLS = list(L.FEATURE_COLUMNS)
i0 = [c for c in COLS if c.startswith("i0_")]
i1 = [c for c in COLS if c.startswith("i1_")]
print(f"{len(i0)} columnas de i0, {len(i1)} de i1, {len(COLS)-len(i0)-len(i1)} comunes")

rng = np.random.default_rng(3)
gs = rng.choice(len(grupos), N_SEQ, replace=False)

# acumuladores de correlación pesada entre el ERROR del ensamble y cada columna retrasada
acc = {}
n_filas = 0
for n, g in enumerate(gs, 1):
    seq_ix, need, x, y, scored = grupos.read(int(g))
    e = ENS[pos[int(seq_ix)]].astype(np.float64)
    y = np.clip(y.astype(np.float64), -2, 2)
    err = y - np.clip(e, -2, 2)
    m = need & scored
    w = np.abs(y)
    for lag in LAGS:
        d = np.full_like(x, np.nan, dtype=np.float64)
        d[lag:] = x[lag:] - x[:-lag]        # cambio de la columna en las ultimas `lag` filas
        mm = m & np.isfinite(d).all(1)
        if mm.sum() == 0:
            continue
        for j, nombre in enumerate(COLS):
            k = (nombre, lag)
            a = acc.setdefault(k, np.zeros(6))
            v, ej, wj = d[mm, j], err[mm, 0], w[mm, 0]
            a += [wj.sum(), (wj * v).sum(), (wj * ej).sum(),
                  (wj * v * v).sum(), (wj * ej * ej).sum(), (wj * v * ej).sum()]
    n_filas += int(m.sum())
    if n % 30 == 0:
        print(f"  {n}/{N_SEQ} secuencias", flush=True)

print(f"\n{n_filas:,} filas puntuadas\n")


def corr(a):
    s, sv, se, svv, see, sve = a
    mv, me = sv / s, se / s
    cov = sve / s - mv * me
    vv, ve = svv / s - mv * mv, see / s - me * me
    if vv <= 0 or ve <= 0:
        return 0.0
    return cov / np.sqrt(vv * ve)


res = [(abs(corr(a)), corr(a), k[0], k[1]) for k, a in acc.items()]
res.sort(reverse=True)
print("las 20 combinaciones (columna, retraso) que más correlacionan con el ERROR del ensamble:\n")
print(f"  {'columna':10s} {'retraso':>8s} {'corr con el error':>18s}")
for ab, c, nombre, lag in res[:20]:
    marca = "  <- i1" if nombre.startswith("i1_") else ""
    print(f"  {nombre:10s} {lag:8d} {c:18.4f}{marca}")

print("\nresumen por instrumento (la correlación más alta en valor absoluto):")
for etiqueta, cols in (("i0", i0), ("i1", i1), ("comunes (a*)", [c for c in COLS if not c.startswith(("i0_", "i1_"))])):
    sub = [r for r in res if r[2] in cols]
    if sub:
        ab, c, nombre, lag = sub[0]
        print(f"  {etiqueta:14s} {c:+.4f}   ({nombre}, retraso {lag})")

print("\nreferencia: con ~370.000 filas, el ruido puro da correlaciones de ±0.003 aprox.")
print("Si lo mas alto de i1 no pasa de 0.01, no queda nada ahi que las redes no esten usando.")
