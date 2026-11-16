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
CLASS_W = np.ascontiguousarray(_W["class_weights"], dtype=np.float32)
N_CLASSES = CLASS_W.shape[0] - 1
WARM_W = np.ascontiguousarray((CLASS_W > 0).any(0).astype(np.float32) / max(1, int((CLASS_W > 0).any(0).sum())))
LUT_COLS = np.ascontiguousarray(_W["lut_cols"], dtype=np.int64)
LUT_VALS = np.ascontiguousarray(_W["lut_vals"], dtype=np.float32)
LUT_CLASS = np.ascontiguousarray(_W["lut_class"], dtype=np.int64)
LUT_OFF = np.ascontiguousarray(_W["lut_off"], dtype=np.int64)
DECIDE = int(_W["decide_row"])
MIN_VOTES = 3

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
def step_row(x, h0, h1, c0, c1, z, gi, gh, out, wts):
    o0 = 0.0
    o1 = 0.0
    for m in range(N_MODELS):
        if wts[m] == 0.0:
            continue
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
        o0 += wts[m] * g0
        o1 += wts[m] * g1
    out[0] = o0
    out[1] = o1


@njit
def vote(x, votes):
    for li in range(LUT_COLS.shape[0]):
        v = x[LUT_COLS[li]]
        lo = LUT_OFF[li]
        hi = LUT_OFF[li + 1]
        a = lo
        b = hi
        while a < b:
            mid = (a + b) // 2
            if LUT_VALS[mid] < v:
                a = mid + 1
            else:
                b = mid
        if a < hi and LUT_VALS[a] == v:
            k = LUT_CLASS[a]
            if k >= 0:
                votes[k] += 1


def decide(votes):
    best = int(np.argmax(votes))
    if votes[best] <= 0 or (votes == votes[best]).sum() > 1:
        return N_CLASSES
    return best


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
        self.votes = np.zeros(N_CLASSES, np.int64)
        self.seq_ix = None
        x0 = np.zeros(N_FEATURES, np.float32)
        step_row(x0, self.h0, self.h1, self.c0, self.c1, self.z, self.gi, self.gh, self.out, WARM_W)
        vote(x0, self.votes)
        self._reset()

    def _reset(self):
        self.h0[:] = 0.0
        self.h1[:] = 0.0
        self.c0[:] = 0.0
        self.c1[:] = 0.0
        self.votes[:] = 0
        self.step = 0
        self.cls = -1
        self.wts = WARM_W

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        x = np.asarray(data_point.state, dtype=np.float32)
        if self.cls < 0:
            vote(x, self.votes)
            claro = self.votes.max() >= MIN_VOTES and (self.votes > 0).sum() == 1
            if claro or self.step >= DECIDE or data_point.need_prediction:
                self.cls = decide(self.votes)
                self.wts = CLASS_W[self.cls]
        step_row(x, self.h0, self.h1, self.c0, self.c1, self.z, self.gi, self.gh, self.out, self.wts)
        self.step += 1
        if not data_point.need_prediction:
            return None
        return self.out.copy()
