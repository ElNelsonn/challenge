"""PASO 4: el test directo. Predecir el factor Y el corrimiento por secuencia, fuera de muestra, y aplicarlos.

Lo que sabemos despues del paso 3:
  solo el factor         -0.0314
  solo el corrimiento    +0.0046
  los dos                +0.0470   <- hay una interaccion fuerte, el +0.047 los necesita juntos
  confianza r_s: PREDECIBLE, R2 = 0.57 fuera de muestra (la senal mas fuerte del proyecto)
  pero aplicar la confianza como multiplicador EMPEORA, incluso con el oraculo (-0.0244)

Entonces la pregunta ya no es si hay senal: es si el corrimiento se puede predecir. El corrimiento optimo es
  b_s = my_s - a_s * mq_s ,  con my_s = suma(|y|*y)/suma(|y|)  el sesgo direccional de los targets de la
secuencia. Predecir eso es predecir hacia donde se movio la secuencia en agregado. Si se puede, hay +0.047
para ir a buscar; si no, la rama queda cerrada con una explicacion precisa y no con una corazonada.

Se mide todo fuera de muestra: ridge ajustado en la mitad de las secuencias, aplicado y medido en la otra.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
            r"\wnn_connectome_starterpack\wnn_connectome_starterpack")
CH = ROOT / "challenge"
R = CH / "model" / "official_gpu" / "results"
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")
sys.path.insert(0, str(ROOT))

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
print(f"{len(Y):,} filas de {n:,} secuencias | ensamble armado", flush=True)


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


FAC = np.ones((n, 2)); OFF = np.zeros((n, 2)); MY = np.zeros((n, 2))
for i, t in enumerate(tramos):
    for j in (0, 1):
        w, y, q = W[t, j], Y[t, j], Q[t, j]
        s = w.sum()
        if s <= 0:
            continue
        mq, my = (w * q).sum() / s, (w * y).sum() / s
        vq = (w * (q - mq) ** 2).sum() / s
        MY[i, j] = my
        if vq > 1e-12:
            FAC[i, j] = (w * (q - mq) * (y - my)).sum() / s / vq
            OFF[i, j] = my - FAC[i, j] * mq

d = np.load(SC / "resumen_valid.npz")
assert np.array_equal(d["seqs"], seqs)
X = d["X"]
PQ = np.zeros((n, 6))
for i, t in enumerate(tramos):
    q = Q[t][:CORTE]
    PQ[i] = [q[:, 0].std(), q[:, 1].std(), np.abs(q[:, 0]).mean(), np.abs(q[:, 1]).mean(),
             q[:, 0].mean(), q[:, 1].mean()]
XP = np.hstack([X, PQ])

rs_ = np.random.default_rng(2026)
perm = rs_.permutation(n)
tr, te = perm[:n // 2], perm[n // 2:]
filas_te = np.concatenate([tramos[i] for i in te])
base = wp(Q, filas_te)


def ridge(X_, y, lam):
    mu, sd = X_[tr].mean(0), X_[tr].std(0) + 1e-9
    A = (X_[tr] - mu) / sd
    ym = y[tr].mean()
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
    pred = ((X_ - mu) / sd) @ coef + ym
    r2 = 1 - ((y[te] - pred[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return pred, r2


print("\n" + "=" * 76)
print(f"QUE SE PUEDE PREDECIR DE CADA PIEZA  (R2 fuera de muestra, {len(te)} secuencias de test)")
print("=" * 76)
for nombre, M in (("factor a_s", FAC), ("corrimiento b_s", OFF), ("sesgo del target my_s", MY)):
    fila = []
    for lam in (100.0, 1000.0, 10000.0):
        fila.append(max(ridge(XP, M[:, j], lam)[1] for j in (0, 1)))
    print(f"  {nombre:24s} R2 = {max(fila):+.3f}   (mejor de lambda 100/1000/10000)")

print("\n" + "=" * 76)
print(f"EL TEST DIRECTO: predecir los dos y aplicarlos  (referencia {base:.4f})")
print("=" * 76)
mejor = (base, None)
for lam in (100.0, 1000.0, 10000.0, 100000.0):
    pf_ = np.ones((n, 2)); po_ = np.zeros((n, 2)); r2f = []; r2o = []
    for j in (0, 1):
        a, s1 = ridge(XP, FAC[:, j], lam); pf_[:, j] = np.clip(a, 0.0, 2.0); r2f.append(s1)
        b, s2 = ridge(XP, OFF[:, j], lam); po_[:, j] = b; r2o.append(s2)
    q = Q.copy()
    for i, t in enumerate(tramos):
        q[t] = q[t] * pf_[i] + po_[i]
    v = wp(q, filas_te)
    print(f"  lambda {lam:>8g}: R2 factor {max(r2f):+.3f} | R2 corrim {max(r2o):+.3f} "
          f"-> WP {v:.4f}  ({v-base:+.4f})")
    if v > mejor[0]:
        mejor = (v, lam)

# el techo honesto: oraculo del factor + corrimiento PREDICHO (y al reves), para ver quien bloquea
print("\n  quien bloquea:")
for lam in (1000.0,):
    po_ = np.zeros((n, 2)); pf_ = np.ones((n, 2))
    for j in (0, 1):
        po_[:, j] = ridge(XP, OFF[:, j], lam)[0]
        pf_[:, j] = np.clip(ridge(XP, FAC[:, j], lam)[0], 0.0, 2.0)
    for nombre, f_, o_ in (("factor ORACULO + corrim predicho", FAC, po_),
                           ("factor predicho + corrim ORACULO", pf_, OFF)):
        q = Q.copy()
        for i, t in enumerate(tramos):
            q[t] = q[t] * f_[i] + o_[i]
        v = wp(q, filas_te)
        print(f"    {nombre:34s} {v:.4f}  ({v-base:+.4f})")

print("\n" + "=" * 76)
if mejor[1] is None:
    print(f"CERRADO: nada mejora la referencia de {base:.4f}. El +0.047 necesita informacion")
    print("que no esta en las features; es un oraculo, no una puerta.")
else:
    print(f"ABIERTO: lambda {mejor[1]:g} -> {mejor[0]:.4f} ({mejor[0]-base:+.4f}) "
          f"-> LB {mejor[0]-0.0323:.4f}")
print("=" * 76)
