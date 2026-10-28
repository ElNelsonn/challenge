import numpy as np
from instrumentos_comun import *
seq, row, Y = load_cache()
m = np.load(SC + r"\member_scored.npz")
SUBS = {"S7": ["y2", "E", "X", "Z"], "S8": ["E", "l1", "LR", "LA"], "S9": ["long", "LR", "LA"], "S10": ["mask", "LR", "LA", "long"]}
LB = {"S7": 0.6542, "S8": 0.6541, "S9": 0.6533, "S10": 0.6540}
useq, inv = np.unique(seq, return_inverse=True)
ns = len(useq)
Yc = np.clip(Y, -2, 2); W = np.abs(Yc)


def stats(Q):
    Qc = np.clip(Q, -2, 2)
    out = np.zeros((ns, 2, 6))
    for t in range(2):
        w, y, q = W[:, t], Yc[:, t], Qc[:, t]
        for j, v in enumerate([w, w * y, w * q, w * y * y, w * q * q, w * y * q]):
            out[:, t, j] = np.bincount(inv, v, ns)
    return out


def score_from(S, cnt):
    T = np.tensordot(cnt, S, axes=(1, 0))
    sw, sy, sq, syy, sqq, syq = [T[..., j] for j in range(6)]
    cov = syq / sw - sy * sq / sw ** 2
    vy = syy / sw - (sy / sw) ** 2; vq = sqq / sw - (sq / sw) ** 2
    return (cov / np.sqrt(vy * vq)).mean(-1)


ST = {k: stats(sum(m[n].astype(np.float64) for n in v) / len(v)) for k, v in SUBS.items()}
full = np.ones((1, ns))
for k in SUBS:
    print(k, "valid", round(float(score_from(ST[k], full)[0]), 4), "LB", LB[k])
rng = np.random.default_rng(0)
for N in (100, 200, 400, 800, 1873):
    cnt = np.zeros((1000, ns))
    for b in range(1000):
        cnt[b] = np.bincount(rng.choice(ns, N, replace=False if N < ns else True), minlength=ns)
    sc = {k: score_from(ST[k], cnt) for k in SUBS}
    d = sc["S10"] - sc["S7"]
    print(f"N={N:5d}: sd(score) {sc['S10'].std():.4f}  delta S10-S7 media {d.mean():+.4f} sd {d.std():.4f}  P(delta<=-0.0002) {np.mean(d <= -0.0002):.3f}  "
          f"sd S9-S8 {(sc['S9'] - sc['S8']).std():.4f}")
