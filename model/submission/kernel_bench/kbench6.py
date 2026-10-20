"""Round-robin de todas las variantes, fijado a un núcleo: comparación relativa confiable aunque la máquina esté cargada."""
import sys, time, math
from pathlib import Path
import numpy as np
from numba import njit
import psutil

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
BUILD = CH / "submissions" / "ens3_A_B_Cswa"
W = np.load(BUILD / "weights.npz")
N = int(W["n_models"]); H = int(W["hidden"]); D = 112
FAST = {"reassoc", "contract", "arcp", "afn", "nsz"}
ROWS = 20000
rng = np.random.default_rng(0)
X = rng.standard_normal((ROWS, D)).astype(np.float32)
MEAN, STD, CLIP = W["mean"], W["std"], W["input_clip"]
W1, R1, WB1, RB1 = W["W1"], W["R1"], W["Wb1"], W["Rb1"]
W2, R2, WB2, RB2 = W["W2"], W["R2"], W["Wb2"], W["Rb2"]
WO, BO, WT, HID = W["Wo"], W["bo"], W["model_weights"], W["hidden_per_model"]
NM = int(sys.argv[1]) if len(sys.argv) > 1 else N
CORE = int(sys.argv[2]) if len(sys.argv) > 2 else 14
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
T = lambda A: np.ascontiguousarray(np.transpose(A, (0, 2, 1)))


def quant(Wf, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(Wf).max(axis=2, keepdims=True).clip(1e-8)
    sc = (amax / qmax).astype(np.float32)
    q = np.clip(np.rint(Wf / sc), -qmax, qmax).astype(np.int8 if bits == 8 else np.int16)
    return np.ascontiguousarray(q), np.ascontiguousarray(sc[..., 0])


# ------------------------------------------------------------------ matvecs
@njit(fastmath=FAST)
def mv_rows(Wm, K, x, b, out, R, S):          # 1 fila por vez (kernel actual). S ignorado
    for row in range(R):
        s = b[row]
        for k in range(K):
            s += Wm[row, k] * x[k]
        out[row] = s


@njit(fastmath=FAST)
def mv_rows8(Wm, K, x, b, out, R, S):         # 8 filas por bloque
    for r0 in range(0, R, 8):
        s0 = b[r0]; s1 = b[r0 + 1]; s2 = b[r0 + 2]; s3 = b[r0 + 3]
        s4 = b[r0 + 4]; s5 = b[r0 + 5]; s6 = b[r0 + 6]; s7 = b[r0 + 7]
        for k in range(K):
            xk = x[k]
            s0 += Wm[r0, k] * xk; s1 += Wm[r0 + 1, k] * xk; s2 += Wm[r0 + 2, k] * xk; s3 += Wm[r0 + 3, k] * xk
            s4 += Wm[r0 + 4, k] * xk; s5 += Wm[r0 + 5, k] * xk; s6 += Wm[r0 + 6, k] * xk; s7 += Wm[r0 + 7, k] * xk
        out[r0] = s0; out[r0 + 1] = s1; out[r0 + 2] = s2; out[r0 + 3] = s3
        out[r0 + 4] = s4; out[r0 + 5] = s5; out[r0 + 6] = s6; out[r0 + 7] = s7


@njit(fastmath=FAST)
def mv_T4(WTm, K, x, b, out, R, S):           # traspuesto (K, R), bloques de 4 k
    for r in range(R):
        out[r] = b[r]
    k = 0
    while k + 4 <= K:
        x0 = x[k]; x1 = x[k + 1]; x2 = x[k + 2]; x3 = x[k + 3]
        for r in range(R):
            out[r] += WTm[k, r] * x0 + WTm[k + 1, r] * x1 + WTm[k + 2, r] * x2 + WTm[k + 3, r] * x3
        k += 4
    while k < K:
        xk = x[k]
        for r in range(R):
            out[r] += WTm[k, r] * xk
        k += 1


@njit(fastmath=FAST)
def mv_rows_q(Wm, K, x, b, out, R, S):        # entero con escala por fila
    for row in range(R):
        s = np.float32(0.0)
        for k in range(K):
            s += np.float32(Wm[row, k]) * x[k]
        out[row] = b[row] + S[row] * s


@njit(fastmath=FAST)
def mv_T4_q(WTm, K, x, b, out, R, S):         # traspuesto entero, escala por columna de salida (= fila original)
    for r in range(R):
        out[r] = np.float32(0.0)
    k = 0
    while k + 4 <= K:
        x0 = x[k]; x1 = x[k + 1]; x2 = x[k + 2]; x3 = x[k + 3]
        for r in range(R):
            out[r] += np.float32(WTm[k, r]) * x0 + np.float32(WTm[k + 1, r]) * x1 + np.float32(WTm[k + 2, r]) * x2 + np.float32(WTm[k + 3, r]) * x3
        k += 4
    while k < K:
        xk = x[k]
        for r in range(R):
            out[r] += np.float32(WTm[k, r]) * xk
        k += 1
    for r in range(R):
        out[r] = b[r] + S[r] * out[r]


# ------------------------------------------------------------------ no-linealidades
@njit(fastmath=FAST)
def ftanh(x):
    if x > 7.90531110763549805:
        x = 7.90531110763549805
    elif x < -7.90531110763549805:
        x = -7.90531110763549805
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
def gates_libm(h, Hn, gi, gh):
    for j in range(Hn):
        z = 1.0 / (1.0 + math.exp(-(gi[j] + gh[j])))
        r = 1.0 / (1.0 + math.exp(-(gi[Hn + j] + gh[Hn + j])))
        n = math.tanh(gi[2 * Hn + j] + r * gh[2 * Hn + j])
        h[j] = (1.0 - z) * n + z * h[j]


@njit(fastmath=FAST)
def gates_fast(h, Hn, gi, gh):
    for j in range(Hn):
        z = 0.5 + 0.5 * ftanh(0.5 * (gi[j] + gh[j]))
        r = 0.5 + 0.5 * ftanh(0.5 * (gi[Hn + j] + gh[Hn + j]))
        n = ftanh(gi[2 * Hn + j] + r * gh[2 * Hn + j])
        h[j] = (1.0 - z) * n + z * h[j]


def make_step(mv, gates):
    @njit(fastmath=FAST)
    def step(x, h0, h1, z, gi, gh, out, MEAN, STD, CLIP, W1, R1, WB1, RB1, W2, R2, WB2, RB2, WO, BO, WT, HID, NM, S1w, S1r, S2w, S2r):
        o0 = 0.0
        o1 = 0.0
        for m in range(NM):
            c = CLIP[m]
            for j in range(D):
                v = (x[j] - MEAN[m, j]) / STD[m, j]
                if v > c:
                    v = c
                elif v < -c:
                    v = -c
                z[j] = v
            Hn = HID[m]
            mv(W1[m], D, z, WB1[m], gi, 3 * Hn, S1w[m])
            mv(R1[m], Hn, h0[m], RB1[m], gh, 3 * Hn, S1r[m])
            gates(h0[m], Hn, gi, gh)
            mv(W2[m], Hn, h0[m], WB2[m], gi, 3 * Hn, S2w[m])
            mv(R2[m], Hn, h1[m], RB2[m], gh, 3 * Hn, S2r[m])
            gates(h1[m], Hn, gi, gh)
            g0 = BO[m, 0]
            g1 = BO[m, 1]
            for k in range(Hn):
                g0 += h1[m, k] * WO[m, k, 0]
                g1 += h1[m, k] * WO[m, k, 1]
            o0 += WT[m] * g0
            o1 += WT[m] * g1
        out[0] = o0
        out[1] = o1
    return step


ones = np.ones((N, 3 * H), np.float32)
F32 = (W1, R1, W2, R2, ones, ones, ones, ones)
F32T = (T(W1), T(R1), T(W2), T(R2), ones, ones, ones, ones)
q16 = [quant(M, 16) for M in (W1, R1, W2, R2)]
I16 = (q16[0][0], q16[1][0], q16[2][0], q16[3][0], q16[0][1], q16[1][1], q16[2][1], q16[3][1])
I16T = (T(q16[0][0]), T(q16[1][0]), T(q16[2][0]), T(q16[3][0]), q16[0][1], q16[1][1], q16[2][1], q16[3][1])


def mk(step, wts):
    s = dict(h0=np.zeros((N, H), np.float32), h1=np.zeros((N, H), np.float32), z=np.zeros(D, np.float32),
             gi=np.zeros(3 * H, np.float32), gh=np.zeros(3 * H, np.float32), out=np.zeros(2, np.float32))
    w1, r1, w2, r2, s1w, s1r, s2w, s2r = wts

    def f(x):
        step(x, s["h0"], s["h1"], s["z"], s["gi"], s["gh"], s["out"], MEAN, STD, CLIP, w1, r1, WB1, RB1, w2, r2, WB2, RB2, WO, BO, WT, HID, NM, s1w, s1r, s2w, s2r)
    return f, s


variants = [
    ("V1  filas f32 + libm", mk(make_step(mv_rows, gates_libm), F32)),
    ("V5  8filas f32 + libm", mk(make_step(mv_rows8, gates_libm), F32)),
    ("V6  traspuesto f32 + libm", mk(make_step(mv_T4, gates_libm), F32T)),
    ("V8  filas int16 + libm", mk(make_step(mv_rows_q, gates_libm), I16)),
    ("V11 traspuesto int16 + libm", mk(make_step(mv_T4_q, gates_libm), I16T)),
    ("V10 filas f32 + ftanh", mk(make_step(mv_rows, gates_fast), F32)),
    ("V13 8filas f32 + ftanh", mk(make_step(mv_rows8, gates_fast), F32)),
    ("V14 traspuesto f32 + ftanh", mk(make_step(mv_T4, gates_fast), F32T)),
    ("V12 filas int16 + ftanh", mk(make_step(mv_rows_q, gates_fast), I16)),
    ("V15 traspuesto int16 + ftanh", mk(make_step(mv_T4_q, gates_fast), I16T)),
]
if __name__ == "__main__":
    p = psutil.Process(); p.cpu_affinity([CORE]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    print(f"modelos usados: {NM}  núcleo {CORE}  reps {REPS}", flush=True)
    for n, (f, s) in variants:
        f(X[0])
    best = {n: 1e9 for n, _ in variants}
    for rep in range(REPS):
        for n, (f, s) in variants:
            t = time.perf_counter()
            for r in range(ROWS):
                f(X[r])
            best[n] = min(best[n], (time.perf_counter() - t) / ROWS * 1e6)
        print(f"rep {rep}: " + "  ".join(f"{n.split()[0]}={best[n]:.1f}" for n, _ in variants), flush=True)
    ref = best[variants[0][0]]
    print()
    for n, _ in variants:
        print(f"{n:32s} {best[n]:6.1f} us/fila  x{ref/best[n]:.2f}   ({best[n]*37.46e6/60e6:5.1f} min laptop, {best[n]*37.46e6/60e6*1.15:5.1f} servidor)")
    # paridad de todas contra V1
    for n, (f, s) in variants:
        s["h0"][:] = 0
        s["h1"][:] = 0
    outs = {n: [] for n, _ in variants}
    for r in range(5000):
        for n, (f, s) in variants:
            f(X[r])
            outs[n].append(s["out"].copy())
    o1 = np.array(outs[variants[0][0]])
    for n, _ in variants[1:]:
        o = np.array(outs[n])
        print(f"paridad {n:32s} max|dif| {np.abs(o - o1).max():.2e}  corr {min(np.corrcoef(o[:, 0], o1[:, 0])[0, 1], np.corrcoef(o[:, 1], o1[:, 1])[0, 1]):.7f}")
