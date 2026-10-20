"""Segunda vuelta: int8 traspuesto, bloques de 8 k, y overhead de llamada según cantidad de argumentos."""
import sys, time
import numpy as np
from numba import njit
import psutil
sys.argv = ["kbench6.py", "3", "14", "5"]
from kbench6 import *

@njit(fastmath=FAST)
def mv_T8_q(WTm, K, x, b, out, R, S):
    for r in range(R):
        out[r] = np.float32(0.0)
    k = 0
    while k + 8 <= K:
        x0 = x[k]; x1 = x[k + 1]; x2 = x[k + 2]; x3 = x[k + 3]
        x4 = x[k + 4]; x5 = x[k + 5]; x6 = x[k + 6]; x7 = x[k + 7]
        for r in range(R):
            out[r] += (np.float32(WTm[k, r]) * x0 + np.float32(WTm[k + 1, r]) * x1 + np.float32(WTm[k + 2, r]) * x2 + np.float32(WTm[k + 3, r]) * x3
                       + np.float32(WTm[k + 4, r]) * x4 + np.float32(WTm[k + 5, r]) * x5 + np.float32(WTm[k + 6, r]) * x6 + np.float32(WTm[k + 7, r]) * x7)
        k += 8
    while k < K:
        xk = x[k]
        for r in range(R):
            out[r] += np.float32(WTm[k, r]) * xk
        k += 1
    for r in range(R):
        out[r] = b[r] + S[r] * out[r]

q8 = [quant(M, 8) for M in (W1, R1, W2, R2)]
I8T = (T(q8[0][0]), T(q8[1][0]), T(q8[2][0]), T(q8[3][0]), q8[0][1], q8[1][1], q8[2][1], q8[3][1])
variants = [
    ("V1  filas f32 + libm", mk(make_step(mv_rows, gates_libm), F32)),
    ("V15 T4 int16 + ftanh", mk(make_step(mv_T4_q, gates_fast), I16T)),
    ("V17 T8 int16 + ftanh", mk(make_step(mv_T8_q, gates_fast), I16T)),
    ("V16 T4 int8 + ftanh", mk(make_step(mv_T4_q, gates_fast), I8T)),
    ("V18 T8 int8 + ftanh", mk(make_step(mv_T8_q, gates_fast), I8T)),
]
p = psutil.Process(); p.cpu_affinity([CORE]); p.nice(psutil.HIGH_PRIORITY_CLASS)
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
    print(f"{n:28s} {best[n]:6.1f} us/fila  x{ref/best[n]:.2f}   ({best[n]*37.46e6/60e6:5.1f} min laptop, {best[n]*37.46e6/60e6*1.15:5.1f} servidor)")

# overhead de llamada vs cantidad de argumentos
@njit
def f2(x, ws):
    ws[0] = x[0]
@njit
def f7(x, a, b, c, d, e, g):
    g[0] = x[0]
ws = np.zeros(4096, np.float32); a = [np.zeros(8, np.float32) for _ in range(6)]
for name, fn in (("2 args", lambda x: f2(x, ws)), ("7 args", lambda x: f7(x, *a))):
    fn(X[0]); t = time.perf_counter()
    for r in range(ROWS): fn(X[r])
    print(f"overhead llamada numba {name}: {(time.perf_counter()-t)/ROWS*1e6:.2f} us")
