"""PASO 2: la escala predicha desde las features, aplicada a las predicciones. Cuantos puntos son?

El paso 1 dio senal: R2 fuera de muestra 0.20-0.26 con el resumen de toda la secuencia y 0.08-0.14 con solo
las primeras 2000 filas. Pero R2 sobre el log de la escala no son puntos de WP. Aca se mide de verdad.

Honestidad de la medicion:
  - el ridge se ajusta SOLO en la mitad de las secuencias y se mide SOLO en la otra mitad;
  - el WP de referencia se calcula en esa misma mitad, asi que la comparacion es justa;
  - se reportan tambien los dos oraculos (escala exacta, y factor+corrimiento exactos) como techo.

Lo que se corrige: nuestras predicciones tienen una escala casi constante entre secuencias y los targets no.
El multiplicador ideal de la secuencia s es sigma_y(s)/sigma_q(s). sigma_q(s) lo conocemos en inferencia
(es de nuestras propias predicciones); el que hay que adivinar es sigma_y(s).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
            r"\wnn_connectome_starterpack\wnn_connectome_starterpack")
CH = ROOT / "challenge"
R = CH / "model" / "official_gpu" / "results"
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402

CORTE = 2000
MODELOS = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
           "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
           "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
           "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Y = np.clip(z["y"], -2, 2).astype(np.float64)
W = np.abs(Y)
orden = np.argsort(seq, kind="stable")
tramos = np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)
seqs = np.array([int(seq[t[0]]) for t in tramos])
print(f"{len(Y):,} filas puntuadas de {len(tramos):,} secuencias", flush=True)

Q = np.zeros((len(Y), 2))
for k, p in MODELOS.items():
    d = np.load(p)
    pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    preds = d["preds"]
    out = np.zeros((len(Y), 2), np.float32)
    for t in tramos:
        s_ = int(seq[t[0]])
        if s_ in pos:
            out[t] = preds[pos[s_]][row[t]]
    Q += np.clip(out.astype(np.float64), -2, 2)
Q /= len(MODELOS)
print("ensamble armado", flush=True)


def wp(q, filas=None):
    """WP global. `filas` restringe a un subconjunto (para medir solo en las secuencias de test)."""
    v = []
    for j in (0, 1):
        if filas is None:
            w, y, p = W[:, j], Y[:, j], q[:, j]
        else:
            w, y, p = W[filas, j], Y[filas, j], q[filas, j]
        s = w.sum()
        my, mp = (w * y).sum() / s, (w * p).sum() / s
        cov = (w * (y - my) * (p - mp)).sum() / s
        vy, vp = (w * (y - my) ** 2).sum() / s, (w * (p - mp) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vp))
    return (v[0] + v[1]) / 2


# -------------------------------------------------- sigma por secuencia, de y y de q
def sigmas(M, t, j):
    w = W[t, j]
    s = w.sum()
    if s <= 0:
        return np.nan
    m = (w * M[t, j]).sum() / s
    return np.sqrt(max((w * (M[t, j] - m) ** 2).sum() / s, 1e-12))


SY = np.array([[sigmas(Y, t, j) for j in (0, 1)] for t in tramos])
SQ = np.array([[sigmas(Q, t, j) for j in (0, 1)] for t in tramos])
print(f"escala de y:  mediana {np.nanmedian(SY):.4f} | p90/p10 "
      f"{np.nanpercentile(SY,90)/np.nanpercentile(SY,10):.2f}x")
print(f"escala de q:  mediana {np.nanmedian(SQ):.4f} | p90/p10 "
      f"{np.nanpercentile(SQ,90)/np.nanpercentile(SQ,10):.2f}x", flush=True)

# -------------------------------------------------- resumen de features por secuencia
# seq_ix esta SHUFFLEADO: no coincide con el indice del row group. Hay que mapear uno a otro.
pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
t0 = time.time()
grupo_de = {}
for g in range(pf.num_row_groups):
    v = int(pf.read_row_group(g, columns=["seq_ix"], use_threads=False)["seq_ix"][0].as_py())
    grupo_de[v] = g
print(f"mapeo seq_ix -> row group listo en {time.time()-t0:.0f} s "
      f"({len(grupo_de)} secuencias)", flush=True)
faltan = [int(s) for s in seqs if int(s) not in grupo_de]
assert not faltan, f"{len(faltan)} seq_ix de las predicciones no estan en el parquet: {faltan[:5]}"

X = np.zeros((len(tramos), 2 * len(FEATURE_COLUMNS)))
t0 = time.time()
for i, s_ in enumerate(seqs):
    t = pf.read_row_group(grupo_de[int(s_)], columns=FEATURE_COLUMNS, use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
    x[~np.isfinite(x)] = 0.0
    xa = x[:CORTE]
    X[i] = np.concatenate([xa.mean(0), xa.std(0)])
    if (i + 1) % 400 == 0:
        print(f"  features {i+1}/{len(seqs)} ({(time.time()-t0)/(i+1):.2f} s/seq)", flush=True)
print(f"features listas en {time.time()-t0:.0f} s\n", flush=True)

# -------------------------------------------------- particion por secuencia
rs = np.random.default_rng(2026)
perm = rs.permutation(len(tramos))
tr, te = perm[:len(perm) // 2], perm[len(perm) // 2:]
filas_te = np.concatenate([tramos[i] for i in te])
base = wp(Q, filas_te)
print("=" * 72)
print(f"REFERENCIA en las {len(te)} secuencias de test: WP = {base:.4f}")
print("=" * 72)


def aplicar(fac):
    """fac: (n_secuencias, 2). Escala las predicciones de cada secuencia."""
    out = Q.copy()
    for i, t in enumerate(tramos):
        out[t] *= fac[i]
    return out


# -------------------------------------------------- techos
ora = np.where(np.isfinite(SY / SQ), SY / SQ, 1.0)
print(f"\n  ORACULO escala exacta (sigma_y/sigma_q):      {wp(aplicar(ora), filas_te):.4f}  "
      f"({wp(aplicar(ora), filas_te)-base:+.4f})")

# oraculo factor + corrimiento: regresion ponderada por secuencia (usa los targets: es techo, no es usable)
fac2 = np.ones((len(tramos), 2)); off2 = np.zeros((len(tramos), 2))
for i, t in enumerate(tramos):
    for j in (0, 1):
        w, y, q = W[t, j], Y[t, j], Q[t, j]
        s = w.sum()
        if s <= 0:
            continue
        mq, my = (w * q).sum() / s, (w * y).sum() / s
        vq = (w * (q - mq) ** 2).sum() / s
        if vq > 1e-12:
            fac2[i, j] = (w * (q - mq) * (y - my)).sum() / s / vq
            off2[i, j] = my - fac2[i, j] * mq
Q2 = Q.copy()
for i, t in enumerate(tramos):
    Q2[t] = Q2[t] * fac2[i] + off2[i]
print(f"  ORACULO factor + corrimiento exactos:         {wp(Q2, filas_te):.4f}  "
      f"({wp(Q2, filas_te)-base:+.4f})   <- el +0.047 de septiembre")

# -------------------------------------------------- lo usable: ridge fuera de muestra
print("\n" + "=" * 72)
print("LO USABLE: ridge ajustado SOLO en las secuencias de train, medido en las de test")
print("=" * 72)
mejor = (base, None, None)
for lam in (10.0, 100.0, 1000.0, 10000.0):
    pred_sy = np.zeros((len(tramos), 2))
    r2 = []
    for j in (0, 1):
        ly = np.log(SY[:, j])
        ok = np.isfinite(ly)
        itr = np.array([i for i in tr if ok[i]])
        mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-9
        A = (X[itr] - mu) / sd
        ym = ly[itr].mean()
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ly[itr] - ym))
        pred_sy[:, j] = np.exp(((X - mu) / sd) @ coef + ym)
        ite = np.array([i for i in te if ok[i]])
        r2.append(1 - ((ly[ite] - np.log(pred_sy[ite, j])) ** 2).sum()
                  / ((ly[ite] - ly[ite].mean()) ** 2).sum())
    fac = np.where(np.isfinite(pred_sy / SQ), pred_sy / SQ, 1.0)
    v = wp(aplicar(fac), filas_te)
    print(f"  lambda {lam:>7g}: R2 test {r2[0]:+.3f}/{r2[1]:+.3f} -> WP {v:.4f}  ({v-base:+.4f})")
    if v > mejor[0]:
        mejor = (v, lam, fac)

print("\n" + "=" * 72)
if mejor[1] is None:
    print("NO SIRVE: ninguna version mejora el WP de referencia.")
else:
    print(f"MEJOR: lambda {mejor[1]:g} -> {mejor[0]:.4f} ({mejor[0]-base:+.4f} sobre {base:.4f})")
    print(f"  proyectado al leaderboard: {mejor[0]-0.0323:.4f}")
print("=" * 72)
