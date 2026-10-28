import numpy as np
from instrumentos_comun import *

seq, row, Y = load_cache()
Q = np.load(SC + r"\ens_scored.npy")
cls = np.load(SC + r"\cls_scored.npy")
Yc = np.clip(Y, -2, 2)
print("global", round(score(Y, Q), 5))
tw = np.abs(Yc).sum(0)
for c in range(4):
    m = cls == c
    w = np.abs(Yc[m])
    print(f"class {c}: rows {m.sum():8d} wshare {np.round(w.sum(0)/tw,3)} score {score(Y[m], Q[m]):.4f} t0 {wp(Y[m,0],Q[m,0]):.4f} t1 {wp(Y[m,1],Q[m,1]):.4f}",
          "wmean y", np.round((w * Yc[m]).sum(0) / w.sum(0), 3), "wmean q", np.round((w * Q[m]).sum(0) / w.sum(0), 3),
          "wsd y", np.round(np.sqrt((w * Yc[m]**2).sum(0) / w.sum(0)), 3), "wsd q", np.round(np.sqrt((w * Q[m]**2).sum(0) / w.sum(0)), 3))


def design(q, cl):
    X = np.zeros((len(q), 8))
    X[np.arange(len(q)), cl] = q
    X[np.arange(len(q)), 4 + cl] = 1.0
    return X


def fit(y, q, cl):
    w = np.abs(np.clip(y, -2, 2)); X = design(q, cl); s = np.sqrt(w)
    return np.linalg.lstsq(X * s[:, None], np.clip(y, -2, 2) * s, rcond=None)[0]


def fit_scale_only(y, q, cl):
    w = np.abs(np.clip(y, -2, 2)); X = np.zeros((len(q), 5)); X[np.arange(len(q)), cl] = q; X[:, 4] = 1; s = np.sqrt(w)
    return np.linalg.lstsq(X * s[:, None], np.clip(y, -2, 2) * s, rcond=None)[0]


useq = np.unique(seq)
rng = np.random.default_rng(0)
G = {"affine": [], "scale": []}
for rep in range(5):
    perm = rng.permutation(useq); A = set(perm[: len(perm) // 2].tolist())
    ma = np.isin(seq, list(A)); mb = ~ma
    for tr, te in [(ma, mb), (mb, ma)]:
        Qa = Q.copy(); Qs = Q.copy()
        for j in range(2):
            p = fit(Y[tr, j], Q[tr, j], cls[tr]); Qa[:, j] = design(Q[:, j], cls) @ p
            p2 = fit_scale_only(Y[tr, j], Q[tr, j], cls[tr])
            X = np.zeros((len(Q), 5)); X[np.arange(len(Q)), cls] = Q[:, j]; X[:, 4] = 1; Qs[:, j] = X @ p2
        base = score(Y[te], Q[te])
        G["affine"].append(score(Y[te], Qa[te]) - base); G["scale"].append(score(Y[te], Qs[te]) - base)
print("honest gain per-class affine: mean %+.5f sd %.5f" % (np.mean(G["affine"]), np.std(G["affine"])))
print("honest gain per-class scale : mean %+.5f sd %.5f" % (np.mean(G["scale"]), np.std(G["scale"])))
for j in range(2):
    p = fit(Y[:, j], Q[:, j], cls); print("full-fit t%d slopes" % j, np.round(p[:4], 3), "intercepts", np.round(p[4:], 3))
