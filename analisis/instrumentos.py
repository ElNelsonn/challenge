import numpy as np, pyarrow.parquet as pq, sys
ROOT = r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack"
QV = np.array([0.12543, 0.24133, 0.46853, 0.65686])


def seq_class(y):
    u = np.unique(np.round(np.abs(y.ravel()), 5)); u = u[u > 1e-6]
    return int(np.argmin(np.abs(QV - u.min()))), u.min()


def run(split):
    pf = pq.ParquetFile(ROOT + rf"\datasets\{split}.parquet")
    n = pf.metadata.num_row_groups
    cols = ["i0_p10", "i0_p14", "i1_p10", "i1_p14", "t0", "t1", "seq_ix"]
    K = [100, 300, 1000, 2000, 5000, 20000]
    out = {"seq": np.zeros(n, np.int64), "cls": np.zeros(n, np.int8), "q": np.zeros(n), "nu": np.zeros((n, 4, len(K)), np.int32)}
    for g in range(n):
        t = pf.read_row_group(g, columns=cols, use_threads=False)
        y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)
        out["seq"][g] = t.column("seq_ix")[0].as_py()
        out["cls"][g], out["q"][g] = seq_class(y)
        for i, c in enumerate(cols[:4]):
            x = t.column(c).to_numpy()
            for j, k in enumerate(K):
                out["nu"][g, i, j] = len(np.unique(x[:k]))
        if g % 1000 == 0:
            print(split, g, flush=True)
    np.savez(f"classes_{split}.npz", **out)
    print(split, np.bincount(out["cls"]), "max q err", np.max(np.abs(QV[out["cls"]] - out["q"])))


run(sys.argv[1])
