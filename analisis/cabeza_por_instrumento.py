import sys, numpy as np
from instrumentos_comun import SC, wp
TAG = sys.argv[1]
tr = np.load(SC + rf"\hid_train_{TAG}.npz"); va = np.load(SC + rf"\hid_valid_{TAG}.npz")
Ht, yt, ct = tr["h"], tr["y"].astype(np.float64), tr["c"]
Hv, yv, cv = va["h"], va["y"].astype(np.float64), va["c"]
P0 = va["p"].astype(np.float64)
print("train rows", len(Ht), "valid rows", len(Hv), flush=True)


def gram(H, y, m):
    yc = np.clip(y[m], -2, 2); w = np.abs(yc)
    A = np.zeros((129, 129)); b = np.zeros((129, 2))
    idx = np.where(m)[0]
    for s in range(0, len(idx), 200000):
        ii = idx[s:s + 200000]
        X = np.c_[H[ii].astype(np.float64), np.ones(len(ii))]
        ww = np.abs(np.clip(y[ii], -2, 2))
        for j in range(2):
            pass
        yield_ = None
        A_j = [(X * ww[:, j:j + 1]).T @ X for j in range(2)]
        b_j = [(X * ww[:, j:j + 1]).T @ np.clip(y[ii, j], -2, 2) for j in range(2)]
        if s == 0:
            AA = A_j; BB = b_j
        else:
            AA = [AA[j] + A_j[j] for j in range(2)]; BB = [BB[j] + b_j[j] for j in range(2)]
    return AA, BB


def solve(A, b, lam=1e-4):
    A = A + lam * np.trace(A) / len(A) * np.eye(len(A))
    return np.linalg.solve(A, b)


def predict(H, beta, m):
    out = np.zeros(m.sum())
    idx = np.where(m)[0]
    for s in range(0, len(idx), 200000):
        ii = idx[s:s + 200000]
        out[s:s + len(ii)] = H[ii].astype(np.float64) @ beta[:128] + beta[128]
    return out


def sc(P, m=None):
    m = np.ones(len(yv), bool) if m is None else m
    return 0.5 * (wp(yv[m, 0], P[m, 0]) + wp(yv[m, 1], P[m, 1]))


Pg = np.zeros_like(P0); Pc = np.zeros_like(P0)
alltr = np.ones(len(Ht), bool); allva = np.ones(len(Hv), bool)
AA, BB = gram(Ht, yt, alltr)
for j in range(2):
    Pg[:, j] = predict(Hv, solve(AA[j], BB[j]), allva)
for c in range(4):
    AA, BB = gram(Ht, yt, ct == c)
    for j in range(2):
        Pc[cv == c, j] = predict(Hv, solve(AA[j], BB[j]), cv == c)
print(f"{'':12s} {'global':>8s} " + " ".join(f"{'c%d' % c:>8s}" for c in range(4)))
for name, P in [("original", P0), ("refit-all", Pg), ("refit-class", Pc)]:
    print(f"{name:12s} {sc(P):8.4f} " + " ".join(f"{sc(P, cv == c):8.4f}" for c in range(4)), flush=True)
