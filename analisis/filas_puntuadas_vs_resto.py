import numpy as np, pyarrow.parquet as pq
from instrumentos_comun import *
pf = pq.ParquetFile(ROOT + r"\datasets\valid.parquet")
cv = np.load(SC + r"\classes_valid.npz"); cls = dict(zip(cv["seq"].tolist(), cv["cls"].tolist()))
z = np.load(ENS["LR"]); P = z["preds"]; pos = {int(s): i for i, s in enumerate(z["seq_ix"])}
Ys, Qs, Ss, Cs = [], [], [], []
for g in range(0, 1873, 3):
    t = pf.read_row_group(g, columns=["seq_ix", "is_scored", "t0", "t1"])
    s = int(t["seq_ix"][0].as_py())
    y = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()])[99:]
    Ys.append(y); Qs.append(P[pos[s], 99:].astype(np.float64)); Ss.append(t["is_scored"].to_numpy()[99:].astype(bool))
    Cs.append(np.full(len(y), cls[s]))
Y = np.concatenate(Ys); Q = np.concatenate(Qs); S = np.concatenate(Ss); C = np.concatenate(Cs)
np.savez(SC + r"\all_rows_LR.npz", Y=Y.astype(np.float32), Q=Q.astype(np.float32), S=S, C=C)


def slope(y, q):
    y = np.clip(y, -2, 2); w = np.abs(y)
    qc = q - (w * q).sum() / w.sum(); yc = y - (w * y).sum() / w.sum()
    return (w * qc * yc).sum() / (w * qc * qc).sum()


for c in range(4):
    k = C == c
    a, b = k & S, k & ~S
    print(f"inst {c}: scored WP {score(Y[a], Q[a]):.4f} slope {slope(Y[a,0],Q[a,0]):.3f}/{slope(Y[a,1],Q[a,1]):.3f} | "
          f"unscored WP {score(Y[b], Q[b]):.4f} slope {slope(Y[b,0],Q[b,0]):.3f}/{slope(Y[b,1],Q[b,1]):.3f} | "
          f"mean|y| sc {np.abs(np.clip(Y[a],-2,2)).mean():.4f} un {np.abs(np.clip(Y[b],-2,2)).mean():.4f} | std q sc {Q[a].std():.4f} un {Q[b].std():.4f}")
print("global scored", round(score(Y[S], Q[S]), 4), "unscored", round(score(Y[~S], Q[~S]), 4))
