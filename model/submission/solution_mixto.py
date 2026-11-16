import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numba import njit

N_FEATURES = 112
DIR = Path(__file__).resolve().parent
_W = np.load(DIR / "weights.npz", allow_pickle=False)
N_MODELS = int(_W["n_models"])
HIDDEN = int(_W["hidden"])
HID = np.ascontiguousarray(_W["hidden_per_model"], dtype=np.int64)
CELL = np.ascontiguousarray(_W["cell_per_model"], dtype=np.int64)
MEAN = np.ascontiguousarray(_W["mean"], dtype=np.float32)
STD = np.ascontiguousarray(_W["std"], dtype=np.float32)
CLIP = np.ascontiguousarray(_W["input_clip"], dtype=np.float32)
W1T, S_W1 = np.ascontiguousarray(_W["W1T"]), np.ascontiguousarray(_W["sW1"], dtype=np.float32)
R1T, S_R1 = np.ascontiguousarray(_W["R1T"]), np.ascontiguousarray(_W["sR1"], dtype=np.float32)
W2T, S_W2 = np.ascontiguousarray(_W["W2T"]), np.ascontiguousarray(_W["sW2"], dtype=np.float32)
R2T, S_R2 = np.ascontiguousarray(_W["R2T"]), np.ascontiguousarray(_W["sR2"], dtype=np.float32)
WB1 = np.ascontiguousarray(_W["Wb1"], dtype=np.float32)
RB1 = np.ascontiguousarray(_W["Rb1"], dtype=np.float32)
WB2 = np.ascontiguousarray(_W["Wb2"], dtype=np.float32)
RB2 = np.ascontiguousarray(_W["Rb2"], dtype=np.float32)
WO = np.ascontiguousarray(_W["Wo"], dtype=np.float32)
BO = np.ascontiguousarray(_W["bo"], dtype=np.float32)
WEIGHTS = np.ascontiguousarray(_W["model_weights"], dtype=np.float32)

_RUTA_F = DIR / "fusion.npz"
_F = np.load(_RUTA_F, allow_pickle=False) if _RUTA_F.exists() else None
N_FUSION = int(_F["n_fusion"]) if _F is not None else 0
if N_FUSION:
    F_IN = int(_F["n_in"])
    F_EMB = int(_F["emb"])
    F_HID = int(_F["hidden"])
    F_MID = int(_F["mid"])
    F_ALFA = np.ascontiguousarray(_F["alfas"], dtype=np.float64)
    F_MEAN = np.ascontiguousarray(_F["mean"], dtype=np.float32)
    F_STD = np.ascontiguousarray(_F["std"], dtype=np.float32)
    FET, S_FE, FEB = np.ascontiguousarray(_F["FET"]), np.ascontiguousarray(_F["sFE"], dtype=np.float32), np.ascontiguousarray(_F["FEB"], dtype=np.float32)
    FIT, S_FI, FIB = np.ascontiguousarray(_F["FIT"]), np.ascontiguousarray(_F["sFI"], dtype=np.float32), np.ascontiguousarray(_F["FIB"], dtype=np.float32)
    FHT, S_FH, FHB = np.ascontiguousarray(_F["FHT"]), np.ascontiguousarray(_F["sFH"], dtype=np.float32), np.ascontiguousarray(_F["FHB"], dtype=np.float32)
    FMT, S_FM, FMB = np.ascontiguousarray(_F["FMT"]), np.ascontiguousarray(_F["sFM"], dtype=np.float32), np.ascontiguousarray(_F["FMB"], dtype=np.float32)
    F_O = np.ascontiguousarray(_F["FO"], dtype=np.float32)
    F_OB = np.ascontiguousarray(_F["FOB"], dtype=np.float32)
    F_W = np.ascontiguousarray(_F["weights"], dtype=np.float32)
    N_SPANS = F_ALFA.shape[0]

FAST = {"reassoc", "contract", "arcp", "afn", "nsz"}
TANH_CLAMP = 7.90531110763549805


@njit(fastmath=FAST)
def _matvec_t(WT, K, x, b, S, out, R):
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
    for j in range(H):
        z = 0.5 + 0.5 * _tanh(0.5 * (gi[j] + gh[j]))
        r = 0.5 + 0.5 * _tanh(0.5 * (gi[H + j] + gh[H + j]))
        n = _tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = (1.0 - z) * n + z * h[j]


@njit(fastmath=FAST)
def _gates_lstm(h, c, H, gi, gh):
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


if N_FUSION:
    @njit(fastmath=FAST)
    def step_fusion(x, primera, prev, emas, feats, zf, h, e, gi, gh, cat, mm, out):
        for j in range(N_FEATURES):
            xj = x[j]
            if not np.isfinite(xj):
                xj = np.float32(0.0)
            feats[j] = xj
            if primera:
                feats[N_FEATURES + j] = 0.0
            else:
                feats[N_FEATURES + j] = xj - prev[j]
            prev[j] = xj
            for s in range(N_SPANS):
                if primera:
                    emas[s, j] = xj
                else:
                    emas[s, j] = F_ALFA[s] * xj + (1.0 - F_ALFA[s]) * emas[s, j]
                feats[(2 + s) * N_FEATURES + j] = np.float32(xj - emas[s, j])
        o0 = 0.0
        o1 = 0.0
        for n in range(N_FUSION):
            for i in range(F_IN):
                zf[i] = (feats[i] - F_MEAN[n, i]) / F_STD[n, i]
            _matvec_t(FET[n], F_IN, zf, FEB[n], S_FE[n], e, F_EMB)
            for i in range(F_EMB):
                if e[i] < 0.0:
                    e[i] = 0.01 * e[i]
            _matvec_t(FIT[n], F_EMB, e, FIB[n], S_FI[n], gi, 3 * F_HID)
            _matvec_t(FHT[n], F_HID, h[n], FHB[n], S_FH[n], gh, 3 * F_HID)
            _gates_gru(h[n], F_HID, gi, gh)
            for i in range(F_HID):
                cat[i] = h[n, i]
            for i in range(F_EMB):
                cat[F_HID + i] = e[i]
            _matvec_t(FMT[n], F_HID + F_EMB, cat, FMB[n], S_FM[n], mm, F_MID)
            g0 = F_OB[n, 0]
            g1 = F_OB[n, 1]
            for i in range(F_MID):
                v = mm[i]
                if v < 0.0:
                    v = 0.01 * v
                g0 += v * F_O[n, i, 0]
                g1 += v * F_O[n, i, 1]
            o0 += F_W[n] * g0
            o1 += F_W[n] * g1
        out[0] += o0
        out[1] += o1


class PredictionModel:
    def __init__(self):
        self.h0 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.h1 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.c0 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.c1 = np.zeros((N_MODELS, HIDDEN), np.float32)
        self.z = np.zeros(N_FEATURES, np.float32)
        self.gi = np.zeros(4 * max(HIDDEN, 1), np.float32)
        self.gh = np.zeros(4 * max(HIDDEN, 1), np.float32)
        self.out = np.zeros(2, np.float32)
        if N_FUSION:
            self.prev = np.zeros(N_FEATURES, np.float32)
            self.emas = np.zeros((N_SPANS, N_FEATURES), np.float64)
            self.feats = np.zeros(F_IN, np.float32)
            self.zf = np.zeros(F_IN, np.float32)
            self.fh = np.zeros((N_FUSION, F_HID), np.float32)
            self.fe = np.zeros(F_EMB, np.float32)
            self.fgi = np.zeros(3 * F_HID, np.float32)
            self.fgh = np.zeros(3 * F_HID, np.float32)
            self.fcat = np.zeros(F_HID + F_EMB, np.float32)
            self.fmm = np.zeros(F_MID, np.float32)
        self.seq_ix = None
        self._paso(np.zeros(N_FEATURES, np.float32), True)
        self._reset()

    def _reset(self):
        self.h0[:] = 0.0
        self.h1[:] = 0.0
        self.c0[:] = 0.0
        self.c1[:] = 0.0
        self.primera = True
        if N_FUSION:
            self.fh[:] = 0.0

    def _paso(self, x, primera):
        step_row(x, self.h0, self.h1, self.c0, self.c1, self.z, self.gi, self.gh, self.out)
        if N_FUSION:
            step_fusion(x, primera, self.prev, self.emas, self.feats, self.zf, self.fh, self.fe, self.fgi,
                        self.fgh, self.fcat, self.fmm, self.out)

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        self._paso(np.asarray(data_point.state, dtype=np.float32), self.primera)
        self.primera = False
        if not data_point.need_prediction:
            return None
        return self.out.copy()
