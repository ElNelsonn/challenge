"""Echo State Network: la única familia de redes del cuadro que nos faltaba probar.

LA IDEA. Una red recurrente donde los pesos recurrentes son ALEATORIOS y NO se entrenan. El "reservorio"
mezcla el pasado de forma fija, y lo único que se entrena es una regresión lineal que lee del estado del
reservorio. Suena raro y funciona: el reservorio hace de memoria y la parte lineal elige qué mirar.

POR QUE NOS INTERESA. Nuestro problema entero es que todo lo que entrenamos termina correlacionando 0.98 con
lo que ya teníamos, y por la cuenta `s > S*r` eso significa que no aporta. Un reservorio aleatorio llega a
otra solución por construcción, así que es lo más distinto que podemos armar después de los árboles.

COMO SE MIDE. Igual que todo lo demás: en las mismas 200 secuencias de valid, reportando el puntaje, la
correlación con nuestro ensamble de 4 redes, y la ganancia honesta (peso elegido en una mitad, medido en la
otra).

Es causal por construcción (el estado solo depende del pasado) y el estado se reinicia en cada secuencia,
igual que exige el problema.
"""
import sys
import time
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
sys.path.insert(0, str(CH / "model" / "official_gpu"))
import train_official_gpu_v1 as T   # noqa: E402

R = CH / "model" / "official_gpu" / "results"
DATOS = CH.parent / "datasets"

N_RES = 400            # tamaño del reservorio
FUGA = 0.3             # cuánto se actualiza el estado en cada fila (1 = sin memoria, 0 = congelado)
RADIO = 0.95           # radio espectral: qué tan cerca del borde del caos trabaja el reservorio
ESCALA_IN = 0.5        # cuánto pesa la entrada contra el estado
RIDGE = 10.0
N_TRAIN = 600          # secuencias de train para ajustar la salida
PASO = 8               # se usa 1 de cada 8 filas para la regresión (alcanza y sobra)
LOTE = 24

rng = np.random.default_rng(1234)

# ---------------------------------------------------------------- reservorio aleatorio (NO se entrena)
W = rng.standard_normal((N_RES, N_RES)).astype(np.float32) / np.sqrt(N_RES)
vals = np.abs(np.linalg.eigvals(W.astype(np.float64)))
W *= np.float32(RADIO / vals.max())
Win = (rng.standard_normal((N_RES, 112)).astype(np.float32) * ESCALA_IN / np.sqrt(112))
bias = (rng.standard_normal(N_RES).astype(np.float32) * 0.1)
print(f"reservorio: {N_RES} unidades, radio espectral {RADIO}, fuga {FUGA}", flush=True)


def correr_reservorio(X):
    """X: (B, T, 112) ya normalizado -> estados (B, T, N_RES). Causal: el estado en t solo ve hasta t."""
    B, Tt, _ = X.shape
    h = np.zeros((B, N_RES), np.float32)
    out = np.empty((B, Tt, N_RES), np.float32)
    XW = X @ Win.T                      # (B, T, N) de una, que es lo caro y BLAS lo hace rápido
    for t in range(Tt):
        pre = XW[:, t] + h @ W.T + bias
        np.tanh(pre, out=pre)
        h = (1.0 - FUGA) * h + FUGA * pre
        out[:, t] = h
    return out


# ---------------------------------------------------------------- normalizador (el mismo que usan las redes)
import torch   # noqa: E402
st = torch.load(R / "sprint6/results/student_lr5e-4/best_model.pt", map_location="cpu",
                weights_only=False)["pipeline"]
MU, SD = st["mean"][:112].astype(np.float32), st["std"][:112].astype(np.float32)
print(f"normalizador tomado de la alumna LR ({len(MU)} features)", flush=True)

train = T.ParquetGroups([str(DATOS / "train.parquet")])
valid = T.ParquetGroups([str(DATOS / "valid.parquet")])
ids_tr = T.select_groups(len(train), N_TRAIN, 4321)
print(f"train: {len(train)} secuencias, uso {len(ids_tr)}", flush=True)

D = N_RES + 112 + 1
G = np.zeros((2, D, D))
B_ = np.zeros((2, D))
t0 = time.time()
for k in range(0, len(ids_tr), LOTE):
    lote = ids_tr[k:k + LOTE]
    datos = [train.read(i) for i in lote]
    X = np.stack([(d[2] - MU) / SD for d in datos]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    Y = np.stack([np.clip(d[3], -2, 2) for d in datos]).astype(np.float32)
    H = correr_reservorio(X)
    # se descartan las primeras 200 filas: el reservorio todavía está arrancando
    z = np.concatenate([H[:, 200::PASO], X[:, 200::PASO],
                        np.ones((len(lote), H[:, 200::PASO].shape[1], 1), np.float32)], 2)
    z = z.reshape(-1, D).astype(np.float64)
    y = Y[:, 200::PASO].reshape(-1, 2).astype(np.float64)
    for j in (0, 1):
        w = np.abs(y[:, j])
        zw = z * w[:, None]
        G[j] += z.T @ zw
        B_[j] += zw.T @ y[:, j]
    if (k // LOTE) % 5 == 0:
        print(f"  {k + len(lote)}/{len(ids_tr)} secuencias ({time.time()-t0:.0f}s)", flush=True)

coef = [np.linalg.solve(G[j] + RIDGE * np.eye(D), B_[j]) for j in (0, 1)]
print(f"salida ajustada en {time.time()-t0:.0f}s\n", flush=True)

# ---------------------------------------------------------------- evaluación en las MISMAS 200 secuencias
ids_va = T.select_groups(len(valid), 200, 2027)
z_cache = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
preds, seqs, rows, ys = [], [], [], []
t0 = time.time()
for k in range(0, len(ids_va), LOTE):
    lote = ids_va[k:k + LOTE]
    datos = [valid.read(i) for i in lote]
    X = np.nan_to_num(np.stack([(d[2] - MU) / SD for d in datos]).astype(np.float32), nan=0.0)
    H = correr_reservorio(X)
    z = np.concatenate([H, X, np.ones((len(lote), X.shape[1], 1), np.float32)], 2)
    p = np.stack([z @ c.astype(np.float32) for c in coef], -1)
    for b, d in enumerate(datos):
        m = d[1] & d[4]
        idx = np.flatnonzero(m)
        preds.append(p[b][idx]); ys.append(np.clip(d[3][idx], -2, 2))
        seqs.append(np.full(len(idx), int(d[0]))); rows.append(idx)
    if (k // LOTE) % 3 == 0:
        print(f"  valid {k + len(lote)}/{len(ids_va)} ({time.time()-t0:.0f}s)", flush=True)

P_esn = np.clip(np.concatenate(preds).astype(np.float64), -2, 2)
Y = np.concatenate(ys).astype(np.float64)
SEQ = np.concatenate(seqs); ROW = np.concatenate(rows)
Wt = np.abs(Y)
print(f"\n{len(Y):,} filas puntuadas de {len(ids_va)} secuencias\n")


def wp(p, m=None):
    m = slice(None) if m is None else m
    v = []
    for j in (0, 1):
        w, y, q = Wt[m, j], Y[m, j], p[m, j]
        s = w.sum()
        my, mq = (w * y).sum() / s, (w * q).sum() / s
        cov = (w * (y - my) * (q - mq)).sum() / s
        vy, vq = (w * (y - my) ** 2).sum() / s, (w * (q - mq) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vq))
    return (v[0] + v[1]) / 2


# el ensamble de 4 redes, en las mismas filas
MOD = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
       "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
       "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
       "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}
acum = None
useq = np.unique(SEQ)
for k, p in MOD.items():
    d = np.load(p)
    pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    pr = d["preds"]
    out = np.zeros((len(Y), 2), np.float32)
    for s_ in useq:
        m = SEQ == s_
        out[m] = pr[pos[int(s_)]][ROW[m]]
    acum = out.astype(np.float64) if acum is None else acum + out
ENS = np.clip(acum / len(MOD), -2, 2)

S = wp(ENS)
s_esn = wp(P_esn)
r = float(np.corrcoef(P_esn[:, 0], ENS[:, 0])[0, 1])
print("=" * 62)
print(f"  ensamble de 4 redes        {S:.4f}")
print(f"  Echo State Network         {s_esn:.4f}")
print(f"  correlacion ESN <-> ensamble: {r:.4f}")
print(f"  (arboles LightGBM 0.9271 | XGBoost 0.9279 | otra LSTM 0.985)")
print(f"  umbral s > S*r = {S:.4f} x {r:.4f} = {S*r:.4f}  ->  "
      f"{'LO PASA' if s_esn > S*r else 'NO lo pasa'} ({s_esn - S*r:+.4f})")
A = (SEQ % 2 == 0)
pesos = np.linspace(0, 0.8, 161)
gA = [wp((1 - a) * ENS + a * P_esn, A) for a in pesos]
mj = pesos[int(np.argmax(gA))]
bB, mB = wp(ENS, ~A), wp((1 - mj) * ENS + mj * P_esn, ~A)
print(f"\n  peso elegido en la mitad A: {mj:.2f}")
print(f"  mitad B: solo redes {bB:.4f} -> con la ESN {mB:.4f}   GANANCIA HONESTA {mB - bB:+.4f}")
print("=" * 62)
print("\ncurva completa (las 200 secuencias):")
for a in (0.0, 0.05, 0.1, 0.2, 0.3):
    print(f"  peso {a:.2f}: {wp((1 - a) * ENS + a * P_esn):.4f}")
np.savez_compressed(Path(__file__).parent / "esn_preds.npz",
                    preds=P_esn.astype(np.float16), seq=SEQ, row=ROW)
