import numpy as np
from instrumentos_comun import *
z = np.load(SC + r"\hid_valid_LR.npz")
H, Y, P, C, S = z["h"], z["y"].astype(np.float64), z["p"].astype(np.float64), z["c"], z["s"]
useq = np.unique(S)
rng = np.random.default_rng(11)
sidx = np.searchsorted(useq, S)
fseq = rng.permutation(len(useq)) % 2
F = fseq[sidx]


def fit(mask, per_class=False):
    out = {}
    groups = range(4) if per_class else [None]
    for g in groups:
        k = mask if g is None else mask & (C == g)
        idx = np.where(k)[0]
        A = np.zeros((2, 129, 129)); b = np.zeros((2, 129))
        for s in range(0, len(idx), 400000):
            j = idx[s:s + 400000]
            X = np.hstack([H[j].astype(np.float64), np.ones((len(j), 1))])
            y = np.clip(Y[j], -2, 2)
            for t in range(2):
                w = np.abs(y[:, t])
                A[t] += (X * w[:, None]).T @ X; b[t] += X.T @ (w * y[:, t])
        out[g] = [np.linalg.solve(A[t] + 1e-3 * np.eye(129), b[t]) for t in range(2)]
    return out


def apply(coefs, mask, per_class=False):
    Q = np.zeros((mask.sum(), 2)); idx = np.where(mask)[0]
    for s in range(0, len(idx), 400000):
        j = idx[s:s + 400000]
        X = np.hstack([H[j].astype(np.float64), np.ones((len(j), 1))])
        for t in range(2):
            if per_class:
                q = np.zeros(len(j))
                for g in range(4):
                    kk = C[j] == g
                    q[kk] = X[kk] @ coefs[g][t]
            else:
                q = X @ coefs[None][t]
            Q[s:s + len(j), t] = q
    return Q


for per_class in (False, True):
    Qcv = np.zeros_like(P)
    for f in range(2):
        co = fit(F != f, per_class)
        Qcv[F == f] = apply(co, F == f, per_class)
    tag = "por instrumento" if per_class else "comun"
    print(f"cabeza {tag} ajustada en filas puntuadas (cross-fit): {score(Y, Qcv):.4f}   original {score(Y, P):.4f}")
    for c in range(4):
        k = C == c
        print(f"   inst {c}: {score(Y[k], Qcv[k]):.4f} vs {score(Y[k], P[k]):.4f}")
    for f in range(2):
        k = F == f
        print(f"   mitad {f}: {score(Y[k], Qcv[k]):.4f} vs {score(Y[k], P[k]):.4f}")
