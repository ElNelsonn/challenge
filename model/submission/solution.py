"""Connectome submission: ensamble de N redes recurrentes de 2 capas (GRU o LSTM, promedio ponderado),
todo en un kernel numba.

Archivos: solution.py + weights.npz. Cada red trae su propio normalizador (mean/std, clip), tipo de celda
(CELL: 0 = GRU, 1 = LSTM) y estado (h0, h1; c0, c1 para LSTM). Pesos apilados como globales (N, ...) y
rellenados a 4*Hmax filas de compuerta (GRU usa 3H, LSTM 4H); numba los embebe como constantes al compilar.
Determinista. Solo numpy + numba.

Kernel v3 (2026-09-19): agrega LSTM (compuertas en el orden de PyTorch i, f, g, o; c = f*c + i*g; h = o*tanh(c)).

Kernel v2 (2026-09-18), 1.6x más rápido que v1 (3x128: 54 vs 85 us/fila en Ryzen 7735U):
  * matrices traspuestas (K, 3H): out[:] += W^T[k,:] * x[k], en bloques de 4 k -> sin reducciones
    horizontales, vectoriza limpio;
  * pesos int8 con escala por fila de salida (4x menos ancho de banda; sin pérdida medible de métrica:
    0.6831 vs 0.6831 en 120 seqs de valid). El mismo kernel acepta int16 o float32 (escala 1);
  * sigmoid/tanh con la aproximación racional 13/6 de Eigen (error ~1e-7) en vez de exp/tanh de libm.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numba import njit

N_FEATURES = 112
_W = np.load(Path(__file__).resolve().parent / "weights.npz", allow_pickle=False)
N_MODELS = int(_W["n_models"])
HIDDEN = int(_W["hidden"])                                    # H máximo (los arrays están rellenados a este tamaño)
HID = np.ascontiguousarray(_W["hidden_per_model"], dtype=np.int64)  # (N,) H real de cada modelo
CELL = np.ascontiguousarray(_W["cell_per_model"], dtype=np.int64)   # (N,) 0 = GRU (3 compuertas z,r,n), 1 = LSTM (4: i,f,g,o)
MEAN = np.ascontiguousarray(_W["mean"], dtype=np.float32)      # (N, 112)
STD = np.ascontiguousarray(_W["std"], dtype=np.float32)        # (N, 112)
CLIP = np.ascontiguousarray(_W["input_clip"], dtype=np.float32)       # (N,) recorte de z por modelo (inf = sin recorte)
# matrices traspuestas (N, K, 4H) del dtype exportado (int8 / int16 / float32) + escala por fila de salida (N, 4H)
W1T, S_W1 = np.ascontiguousarray(_W["W1T"]), np.ascontiguousarray(_W["sW1"], dtype=np.float32)   # (N, 112, 4H)
R1T, S_R1 = np.ascontiguousarray(_W["R1T"]), np.ascontiguousarray(_W["sR1"], dtype=np.float32)   # (N, H, 4H)
W2T, S_W2 = np.ascontiguousarray(_W["W2T"]), np.ascontiguousarray(_W["sW2"], dtype=np.float32)   # (N, H, 4H)
R2T, S_R2 = np.ascontiguousarray(_W["R2T"]), np.ascontiguousarray(_W["sR2"], dtype=np.float32)   # (N, H, 4H)
WB1 = np.ascontiguousarray(_W["Wb1"], dtype=np.float32)        # (N, 4H)  sesgos (GRU: z, r, n; LSTM: i, f, g, o)
RB1 = np.ascontiguousarray(_W["Rb1"], dtype=np.float32)
WB2 = np.ascontiguousarray(_W["Wb2"], dtype=np.float32)
RB2 = np.ascontiguousarray(_W["Rb2"], dtype=np.float32)
WO = np.ascontiguousarray(_W["Wo"], dtype=np.float32)          # (N, H, 2)
BO = np.ascontiguousarray(_W["bo"], dtype=np.float32)          # (N, 2)
WEIGHTS = np.ascontiguousarray(_W["model_weights"], dtype=np.float32)  # (N,) suman 1

FAST = {"reassoc", "contract", "arcp", "afn", "nsz"}
TANH_CLAMP = 7.90531110763549805


@njit(fastmath=FAST)
def _matvec_t(WT, K, x, b, S, out, R):
    """out[r] = b[r] + S[r] * sum_k WT[k, r] * x[k]  (r < R). WT de forma (K_pad, R_pad), bloques de 4 k."""
    for r in range(R):
        out[r] = np.float32(0.0)
    k = 0
    while k + 4 <= K:
        x0 = x[k]
        x1 = x[k + 1]
        x2 = x[k + 2]
        x3 = x[k + 3]
        for r in range(R):
            out[r] += (np.float32(WT[k, r]) * x0 + np.float32(WT[k + 1, r]) * x1
                       + np.float32(WT[k + 2, r]) * x2 + np.float32(WT[k + 3, r]) * x3)
        k += 4
    while k < K:
        xk = x[k]
        for r in range(R):
            out[r] += np.float32(WT[k, r]) * xk
        k += 1
    for r in range(R):
        out[r] = b[r] + S[r] * out[r]


@njit(fastmath=FAST)
def _tanh(x):
    """tanh racional 13/6 (Eigen generic_fast_tanh_float): |error| < 3e-7, pura aritmética -> vectoriza."""
    if x > TANH_CLAMP:
        x = TANH_CLAMP
    elif x < -TANH_CLAMP:
        x = -TANH_CLAMP
    x2 = x * x
    p = x2 * -2.76076847742355e-16 + 2.00018790482477e-13
    p = x2 * p + -8.60467152213735e-11
    p = x2 * p + 5.12229709037114e-08
    p = x2 * p + 1.48572235717979e-05
    p = x2 * p + 6.37261928875436e-04
    p = x2 * p + 4.89352455891786e-03
    p = x * p
    q = x2 * 1.19825839466702e-06 + 1.18534705686654e-04
    q = x2 * q + 2.26843463243900e-03
    q = x2 * q + 4.89352518554385e-03
    return p / q


@njit(fastmath=FAST)
def _gates_gru(h, H, gi, gh):
    """Compuertas GRU (orden z, r, n) con sigmoid(x) = 0.5 + 0.5*tanh(x/2). Actualiza h in-place."""
    for j in range(H):
        z = 0.5 + 0.5 * _tanh(0.5 * (gi[j] + gh[j]))
        r = 0.5 + 0.5 * _tanh(0.5 * (gi[H + j] + gh[H + j]))
        n = _tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = (1.0 - z) * n + z * h[j]


@njit(fastmath=FAST)
def _gates_lstm(h, c, H, gi, gh):
    """Compuertas LSTM (orden PyTorch i, f, g, o). Actualiza h y c in-place."""
    for j in range(H):
        i = 0.5 + 0.5 * _tanh(0.5 * (gi[j] + gh[j]))
        f = 0.5 + 0.5 * _tanh(0.5 * (gi[H + j] + gh[H + j]))
        g = _tanh(gi[2 * H + j] + gh[2 * H + j])
        o = 0.5 + 0.5 * _tanh(0.5 * (gi[3 * H + j] + gh[3 * H + j]))
        cj = f * c[j] + i * g
        c[j] = cj
        h[j] = o * _tanh(cj)


@njit(fastmath=FAST)
def step_row(x, h0, h1, c0, c1, z, gi, gh, out):
    """Una fila: actualiza h0[N,H], h1[N,H] (y c0, c1 en LSTM) in-place; out = promedio ponderado de las N salidas."""
    o0 = 0.0
    o1 = 0.0
    for m in range(N_MODELS):
        c = CLIP[m]
        for j in range(N_FEATURES):
            v = (x[j] - MEAN[m, j]) / STD[m, j]
            if v > c:
                v = c
            elif v < -c:
                v = -c
            z[j] = v
        H = HID[m]
        G = 3 * H if CELL[m] == 0 else 4 * H
        _matvec_t(W1T[m], N_FEATURES, z, WB1[m], S_W1[m], gi, G)
        _matvec_t(R1T[m], H, h0[m], RB1[m], S_R1[m], gh, G)
        if CELL[m] == 0:
            _gates_gru(h0[m], H, gi, gh)
        else:
            _gates_lstm(h0[m], c0[m], H, gi, gh)
        _matvec_t(W2T[m], H, h0[m], WB2[m], S_W2[m], gi, G)
        _matvec_t(R2T[m], H, h1[m], RB2[m], S_R2[m], gh, G)
        if CELL[m] == 0:
            _gates_gru(h1[m], H, gi, gh)
        else:
            _gates_lstm(h1[m], c1[m], H, gi, gh)
        g0 = BO[m, 0]
        g1 = BO[m, 1]
        for k in range(H):
            g0 += h1[m, k] * WO[m, k, 0]
            g1 += h1[m, k] * WO[m, k, 1]
        o0 += WEIGHTS[m] * g0
        o1 += WEIGHTS[m] * g1
    out[0] = o0
    out[1] = o1


class PredictionModel:
    def __init__(self):
        self.h0 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.h1 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.c0 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.c1 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.z = np.zeros(N_FEATURES, np.float32)
        self.gi = np.zeros(4 * HIDDEN, np.float32)
        self.gh = np.zeros(4 * HIDDEN, np.float32)
        self.out = np.zeros(2, np.float32)
        self.seq_ix = None
        step_row(np.zeros(N_FEATURES, np.float32), self.h0, self.h1, self.c0, self.c1, self.z, self.gi, self.gh, self.out)  # compila ahora
        self._reset()

    def _reset(self):
        self.h0[:] = 0.0
        self.h1[:] = 0.0
        self.c0[:] = 0.0
        self.c1[:] = 0.0

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        x = np.asarray(data_point.state, dtype=np.float32)
        step_row(x, self.h0, self.h1, self.c0, self.c1, self.z, self.gi, self.gh, self.out)
        if not data_point.need_prediction:
            return None
        return self.out.copy()
