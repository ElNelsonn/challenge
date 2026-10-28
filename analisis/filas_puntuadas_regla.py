import numpy as np, pyarrow.parquet as pq
from instrumentos_comun import ROOT, SC
pf = pq.ParquetFile(ROOT + r"\datasets\valid.parquet")
cv = np.load(SC + r"\classes_valid.npz")
cls = dict(zip(cv["seq"].tolist(), cv["cls"].tolist())) if "seq" in cv else None

L = range(-15, 16)
acc = {c: {"n": np.zeros(len(L)), "s": np.zeros(len(L))} for c in range(4)}
gaps = {c: [] for c in range(4)}
base = {c: [0, 0] for c in range(4)}
for g in range(0, 1873, 12):
    t = pf.read_row_group(g, columns=["seq_ix", "is_scored", "t0", "t1"])
    s = int(t["seq_ix"][0].as_py()); c = cls[s]
    sc = t["is_scored"].to_numpy().astype(bool)
    y = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()])
    ch = np.r_[False, np.any(np.abs(np.diff(y, axis=0)) > 1e-9, axis=1)]
    idx = np.where(sc)[0]
    gaps[c] += np.diff(idx).tolist()
    base[c][0] += sc[99:].sum(); base[c][1] += len(sc) - 99
    cp = np.where(ch)[0]
    n = len(sc)
    for j, l in enumerate(L):
        k = cp + l
        k = k[(k >= 99) & (k < n)]
        acc[c]["n"][j] += len(k); acc[c]["s"][j] += sc[k].sum()
for c in range(4):
    r = base[c][0] / base[c][1]
    print(f"inst {c}: base scored {r:.4f}  gaps median {np.median(gaps[c]):.0f} min {np.min(gaps[c])} p10 {np.percentile(gaps[c],10):.0f} p90 {np.percentile(gaps[c],90):.0f}")
    print("   lag:ratio " + " ".join(f"{l}:{acc[c]['s'][j]/max(acc[c]['n'][j],1)/r:.2f}" for j, l in enumerate(L)))
