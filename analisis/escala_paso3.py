"""PASO 3: el +0.047 no es escala, es CONFIANZA. Se puede predecir en que secuencias predecimos bien?

El paso 2 encontro algo que cambia el diagnostico de septiembre:
  oraculo de escala exacta (sigma_y/sigma_q):  -0.0566   <- igualar la escala EMPEORA
  oraculo de factor + corrimiento:             +0.0470   <- este es el +0.047

El factor optimo de la secuencia s no es sigma_y/sigma_q, es  (sigma_y/sigma_q) * r_s , donde r_s es la
correlacion DENTRO de esa secuencia. O sea: el oraculo no iguala escalas, ENCOGE las predicciones de las
secuencias donde predecimos mal. Es una estimacion de confianza, no de escala. Eso explica por que en
septiembre fallo normalizar por la escala de las propias predicciones: la informacion que falta es otra.

Aca mido, en orden:
  1. de donde sale el +0.047: solo factor, solo corrimiento, o los dos
  2. cuanto de r_s (la confianza) se puede predecir desde las features, fuera de muestra
  3. cuantos puntos da eso aplicado de verdad

Las features incluyen el resumen de nuestras PROPIAS predicciones de la secuencia, que en inferencia
tambien tenemos (es causal: a la fila t conocemos las predicciones 0..t).
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
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")
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
n = len(tramos)
print(f"{len(Y):,} filas puntuadas de {n:,} secuencias", flush=True)

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


def wp(q, filas):
    v = []
    for j in (0, 1):
        w, y, p = W[filas, j], Y[filas, j], q[filas, j]
        s = w.sum()
        my, mp = (w * y).sum() / s, (w * p).sum() / s
        cov = (w * (y - my) * (p - mp)).sum() / s
        vy, vp = (w * (y - my) ** 2).sum() / s, (w * (p - mp) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vp))
    return (v[0] + v[1]) / 2


# -------------------------------------------------- factor, corrimiento y r por secuencia
FAC = np.ones((n, 2)); OFF = np.zeros((n, 2)); RS = np.zeros((n, 2)); SQ = np.ones((n, 2))
for i, t in enumerate(tramos):
    for j in (0, 1):
        w, y, q = W[t, j], Y[t, j], Q[t, j]
        s = w.sum()
        if s <= 0:
            continue
        mq, my = (w * q).sum() / s, (w * y).sum() / s
        vq = (w * (q - mq) ** 2).sum() / s
        vy = (w * (y - my) ** 2).sum() / s
        cov = (w * (q - mq) * (y - my)).sum() / s
        SQ[i, j] = np.sqrt(max(vq, 1e-12))
        if vq > 1e-12 and vy > 1e-12:
            FAC[i, j] = cov / vq
            OFF[i, j] = my - FAC[i, j] * mq
            RS[i, j] = cov / np.sqrt(vq * vy)

print(f"\nconfianza r_s por secuencia: mediana {np.median(RS):.3f} | "
      f"p10 {np.percentile(RS,10):.3f} | p90 {np.percentile(RS,90):.3f}")
print(f"factor optimo:               mediana {np.median(FAC):.3f} | "
      f"p10 {np.percentile(FAC,10):.3f} | p90 {np.percentile(FAC,90):.3f}")
print(f"corrimiento optimo:          mediana {np.median(OFF):+.4f} | "
      f"p10 {np.percentile(OFF,10):+.4f} | p90 {np.percentile(OFF,90):+.4f}", flush=True)

rs_ = np.random.default_rng(2026)
perm = rs_.permutation(n)
tr, te = perm[:n // 2], perm[n // 2:]
filas_te = np.concatenate([tramos[i] for i in te])
base = wp(Q, filas_te)

print("\n" + "=" * 74)
print(f"1. DE DONDE SALE EL +0.047   (referencia en test: {base:.4f})")
print("=" * 74)


def aplicar(fac=None, off=None):
    out = Q.copy()
    for i, t in enumerate(tramos):
        if fac is not None:
            out[t] = out[t] * fac[i]
        if off is not None:
            out[t] = out[t] + off[i]
    return out


for nombre, f_, o_ in (("solo el factor", FAC, None),
                       ("solo el corrimiento", None, OFF),
                       ("los dos", FAC, OFF)):
    v = wp(aplicar(f_, o_), filas_te)
    print(f"  {nombre:22s} {v:.4f}  ({v-base:+.4f})")
# y la pregunta clave: cuanto de eso es la confianza sola, sin la escala
solo_r = np.where(RS > 0, RS, 0.0)
v = wp(aplicar(solo_r, None), filas_te)
print(f"  {'solo la confianza r_s':22s} {v:.4f}  ({v-base:+.4f})   <- encoger por confianza, sin tocar escala")

# -------------------------------------------------- features (con cache)
cache = SC / "resumen_valid.npz"
if cache.exists():
    d = np.load(cache)
    assert np.array_equal(d["seqs"], seqs), "el cache es de otras secuencias"
    X = d["X"]
    print(f"\nresumen de features leido del cache ({X.shape[1]} numeros por secuencia)")
else:
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    grupo_de = {int(pf.read_row_group(g, columns=["seq_ix"], use_threads=False)["seq_ix"][0].as_py()): g
                for g in range(pf.num_row_groups)}
    X = np.zeros((n, 3 * len(FEATURE_COLUMNS)))
    t0 = time.time()
    for i, s_ in enumerate(seqs):
        t = pf.read_row_group(grupo_de[int(s_)], columns=FEATURE_COLUMNS, use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        x[~np.isfinite(x)] = 0.0
        xa = x[:CORTE]
        X[i] = np.concatenate([xa.mean(0), xa.std(0), np.abs(np.diff(xa, axis=0)).mean(0)])
        if (i + 1) % 500 == 0:
            print(f"  features {i+1}/{n} ({(time.time()-t0)/(i+1):.2f} s/seq)", flush=True)
    np.savez_compressed(cache, X=X, seqs=seqs)
    print(f"\nresumen de features listo en {time.time()-t0:.0f} s, cacheado")

# el resumen de nuestras propias predicciones tempranas: en inferencia tambien lo tenemos
PQ = np.zeros((n, 6))
for i, t in enumerate(tramos):
    q = Q[t][:CORTE]
    PQ[i] = [q[:, 0].std(), q[:, 1].std(), np.abs(q[:, 0]).mean(), np.abs(q[:, 1]).mean(),
             np.abs(np.diff(q[:, 0])).mean(), np.abs(np.diff(q[:, 1])).mean()]
XP = np.hstack([X, PQ])
print(f"features: {X.shape[1]} del libro + 6 de nuestras predicciones = {XP.shape[1]}")


def ridge_fuera(X_, y, lam, itr, ite):
    mu, sd = X_[itr].mean(0), X_[itr].std(0) + 1e-9
    A = (X_[itr] - mu) / sd
    ym = y[itr].mean()
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[itr] - ym))
    pred = ((X_ - mu) / sd) @ coef + ym
    r2 = 1 - ((y[ite] - pred[ite]) ** 2).sum() / ((y[ite] - y[ite].mean()) ** 2).sum()
    return pred, r2


print("\n" + "=" * 74)
print("2 y 3. SE PUEDE PREDECIR LA CONFIANZA, Y CUANTOS PUNTOS DA")
print("=" * 74)
mejor = (base, None)
for etiqueta, X_ in (("solo libro", X), ("libro + nuestras preds", XP)):
    print(f"\n  {etiqueta}:")
    for lam in (100.0, 1000.0, 10000.0):
        pr = np.zeros((n, 2)); r2 = []
        for j in (0, 1):
            p_, s_ = ridge_fuera(X_, RS[:, j], lam, tr, te)
            pr[:, j] = np.clip(p_, 0.0, 1.5)
            r2.append(s_)
        v = wp(aplicar(pr, None), filas_te)
        print(f"    lambda {lam:>7g}: R2 de r_s {r2[0]:+.3f}/{r2[1]:+.3f} -> WP {v:.4f} ({v-base:+.4f})")
        if v > mejor[0]:
            mejor = (v, f"{etiqueta}, lambda {lam:g}")

print("\n" + "=" * 74)
if mejor[1] is None:
    print(f"NO SIRVE: nada mejora la referencia de {base:.4f}.")
else:
    print(f"MEJOR: {mejor[1]} -> {mejor[0]:.4f} ({mejor[0]-base:+.4f})")
    print(f"  proyectado al leaderboard: {mejor[0]-0.0323:.4f}")
print("=" * 74)
