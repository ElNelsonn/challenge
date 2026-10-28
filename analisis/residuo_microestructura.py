import sys, numpy as np, pyarrow.parquet as pq
from scipy.special import ndtr
from instrumentos_comun import *
CLS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 240
pf = pq.ParquetFile(ROOT + r"\datasets\valid.parquet")
cv = np.load(SC + r"\classes_valid.npz")
z = np.load(ENS["LR"]); PR = z["preds"]; pos = {int(s): i for i, s in enumerate(z["seq_ix"])}
BID = [10, 5, 2, 1, 7, 4, 6, 9, 8, 0, 3]
ASK = [14, 21, 11, 15, 17, 12, 13, 18, 20, 16, 19]
cols = ["seq_ix", "is_scored", "t0", "t1"] + [f"i0_p{j}" for j in (10, 14)] + [f"i0_v{j}" for j in BID[:3] + ASK[:3]] + \
       [f"i1_p{j}" for j in (10, 14)] + [f"i1_v{j}" for j in (10, 14)]


def since_change(x):
    ch = np.r_[True, np.diff(x) != 0]
    idx = np.where(ch, np.arange(len(x)), 0)
    last = np.maximum.accumulate(idx)
    return np.arange(len(x)) - last, last


def last_dir(x):
    d = np.sign(np.r_[0, np.diff(x)])
    idx = np.where(d != 0, np.arange(len(x)), 0)
    last = np.maximum.accumulate(idx)
    return d[last]


def ema(x, a):
    out = np.empty_like(x); m = x[0]
    for i in range(len(x)):
        m += a * (x[i] - m); out[i] = m
    return out


rows = []
ids = np.where(cv["cls"] == CLS)[0]
ids = np.random.default_rng(3).choice(ids, min(NSEQ, len(ids)), replace=False)
for g in ids:
    t = pf.read_row_group(int(g), columns=cols, use_threads=False)
    s = int(t["seq_ix"][0].as_py())
    sc = t["is_scored"].to_numpy().astype(bool)
    y = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()]).astype(np.float64)
    q = PR[pos[s]].astype(np.float64)
    b = t["i0_p10"].to_numpy().astype(np.float64); a = t["i0_p14"].to_numpy().astype(np.float64)
    vb = [ndtr(t[f"i0_v{j}"].to_numpy().astype(np.float64)) for j in BID[:3]]
    va = [ndtr(t[f"i0_v{j}"].to_numpy().astype(np.float64)) for j in ASK[:3]]
    xb = t[f"i0_v{BID[0]}"].to_numpy().astype(np.float64); xa = t[f"i0_v{ASK[0]}"].to_numpy().astype(np.float64)
    tb, _ = since_change(b); ta, _ = since_change(a)
    mid = (ndtr(b) + ndtr(a)) / 2
    F = {
        "imb1_rank": vb[0] - va[0],
        "imb1_ratio": (vb[0] - va[0]) / (vb[0] + va[0] + 1e-9),
        "imb1_x": xb - xa,
        "imb3_rank": sum(vb) - sum(va),
        "log_tb": np.log1p(tb), "log_ta": np.log1p(ta), "tb_minus_ta": np.log1p(tb) - np.log1p(ta),
        "lastdir_b": last_dir(b), "lastdir_a": last_dir(a),
        "mid_dev_100": mid - ema(mid, 1 / 100), "mid_dev_1000": mid - ema(mid, 1 / 1000), "mid_dev_5000": mid - ema(mid, 1 / 5000),
        "imb1_ema50": ema(vb[0] - va[0], 1 / 50),
        "i1_imb": ndtr(t["i1_v10"].to_numpy().astype(np.float64)) - ndtr(t["i1_v14"].to_numpy().astype(np.float64)),
    }
    k = sc.copy(); k[:99] = False
    rows.append((y[k], q[k], np.column_stack([F[n][k] for n in F]), np.full(k.sum(), g)))
names = list(F)
Y = np.concatenate([r[0] for r in rows]); Q = np.concatenate([r[1] for r in rows]); X = np.concatenate([r[2] for r in rows])
G = np.concatenate([r[3] for r in rows])
print(f"inst {CLS}: {len(ids)} secuencias, {len(Y)} filas puntuadas, WP LR {score(Y, Q):.4f}")
Yc = np.clip(Y, -2, 2)


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum(); b = b - (w * b).sum() / w.sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())


for t in range(2):
    w = np.abs(Yc[:, t]); r = Yc[:, t] - Q[:, t]
    print(f"\nt{t}: corr(feature, y) | corr(feature, residuo) | corr(feature, pred)")
    for j, n in enumerate(names):
        print(f"  {n:14s} {wcorr(X[:, j], Yc[:, t], w):+.3f}  {wcorr(X[:, j], r, w):+.3f}  {wcorr(X[:, j], Q[:, t], w):+.3f}")
useq = np.unique(G); fold = np.random.default_rng(7).permutation(len(useq)) % 2
Fo = fold[np.searchsorted(useq, G)]
Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
Qcv = np.zeros_like(Q); Qb = np.zeros_like(Q)
for f in range(2):
    tr, te = Fo != f, Fo == f
    for t in range(2):
        w = np.abs(Yc[tr, t])
        for out, D in ((Qcv, np.column_stack([Q[:, t], Xs, np.ones(len(Q))])), (Qb, np.column_stack([Q[:, t], np.ones(len(Q))]))):
            A = (D[tr] * w[:, None]).T @ D[tr]; bb = D[tr].T @ (w * Yc[tr, t])
            out[te, t] = D[te] @ np.linalg.solve(A + 1e-6 * np.eye(D.shape[1]), bb)
print(f"\ncross-fit: solo pred {score(Y, Qb):.4f} | pred + features {score(Y, Qcv):.4f} | ganancia {score(Y, Qcv) - score(Y, Qb):+.4f}")
