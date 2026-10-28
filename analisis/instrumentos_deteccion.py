import numpy as np, pyarrow.parquet as pq
from collections import defaultdict
from instrumentos_comun import ROOT, SC
COLS = ["i0_p10", "i0_p14", "a7", "i0_dp1", "i1_p10", "i1_p14", "i0_p5", "i0_p21"]
Ct = np.load(SC + r"\classes_train.npz"); Cv = np.load(SC + r"\classes_valid.npz")
pt = pq.ParquetFile(ROOT + r"\datasets\train.parquet"); pv = pq.ParquetFile(ROOT + r"\datasets\valid.parquet")
NPER = 500
tab = {c: defaultdict(lambda: np.zeros(4)) for c in COLS}
for k in range(4):
    for g in np.where(Ct["cls"] == k)[0][:NPER]:
        t = pt.read_row_group(int(g), columns=COLS, use_threads=False)
        for c in COLS:
            for v in np.unique(t.column(c).to_numpy()):
                tab[c][float(v)][k] += 1
for c in COLS:
    n = len(tab[c]); amb = sum(1 for v in tab[c].values() if (v > 0).sum() > 1)
    print(f"{c}: {n} distinct values, {amb} appear in >1 class")
counts = np.bincount(Ct["cls"][: 0] if False else np.repeat(np.arange(4), NPER), minlength=4)
K = [1, 5, 20, 50, 99, 300, 1000]
for cset in [["a7"], ["i0_p10", "i0_p14"], ["i0_p10", "i0_p14", "a7", "i0_dp1"], COLS]:
    acc = {k: 0 for k in K}; nseq = 0
    for g in range(0, pv.metadata.num_row_groups, 3):
        t = pv.read_row_group(g, columns=cset, use_threads=False)
        true = Cv["cls"][g]; nseq += 1
        for k in K:
            lp = np.zeros(4)
            for c in cset:
                for v in np.unique(t.column(c).to_numpy()[:k]):
                    cnt = tab[c].get(float(v))
                    if cnt is None:
                        continue
                    lp += np.log((cnt + 0.5) / (NPER + 1))
            acc[k] += int(np.argmax(lp) == true)
    print(cset, {k: round(acc[k] / nseq, 3) for k in K})
