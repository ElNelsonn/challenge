import sys, time, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from instrumentos_comun import ROOT, SC, score
sys.path.insert(0, ROOT)
from utils import FEATURE_COLUMNS
NVA = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
QT = 0.65686
I0 = [j for j, c in enumerate(FEATURE_COLUMNS) if c.startswith("i0_")]


def feats(x):
    out = [x]
    for lag in (1, 10, 100):
        d = np.zeros_like(x[:, I0]); d[lag:] = x[lag:, I0] - x[:-lag, I0]
        out.append(d)
    return np.hstack(out).astype(np.float32)


def states(T):
    s = np.full(len(T), 4)
    s[(T[:, 0] == 0) & (T[:, 1] == -1)] = 0
    s[(T[:, 0] > 0) & (T[:, 1] == -3)] = 1
    s[(T[:, 0] == -1) & (T[:, 1] == 0)] = 2
    s[(T[:, 0] == -3) & (T[:, 1] > 0)] = 3
    return s


def build(split, hid, seqs, rows_of):
    pf = pq.ParquetFile(ROOT + rf"\datasets\{split}.parquet")
    cl = np.load(SC + rf"\classes_{split}.npz")
    g_of = {int(s): i for i, s in enumerate(cl["seq"])}
    X, Y, S, Pl = [], [], [], []
    for s in seqs:
        k = np.where(hid["s"] == s)[0]
        t = pf.read_row_group(g_of[int(s)], columns=FEATURE_COLUMNS + (["is_scored"] if split == "valid" else []), use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        r = rows_of(t)
        assert len(r) == len(k), (s, len(r), len(k))
        F = feats(x)[r]
        X.append(np.hstack([F, hid["h"][k], hid["p"][k]])); Y.append(hid["y"][k].astype(np.float64)); Pl.append(hid["p"][k].astype(np.float64))
        S.append(states(np.round(Y[-1] / QT).astype(int)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(Pl)


def rows_train(t):
    r = np.arange(20000)
    return r[(r >= 99) & (r % 8 == 0)]


def rows_valid(t):
    sc = t["is_scored"].to_numpy().astype(bool); sc[:99] = False
    return np.where(sc)[0]


t0 = time.time()
ht = dict(np.load(SC + r"\hid_train_LR.npz"))
m3 = ht["c"] == 3
ht = {k: v[m3] for k, v in ht.items()}
seq_tr = np.unique(ht["s"])
Xt, Yt, St, Pt = build("train", ht, seq_tr, rows_train)
del ht
hv = np.load(SC + r"\hid_valid_LR.npz")
cv = hv["c"]; m3 = cv == 3
hv = {k: hv[k][m3] for k in ("h", "p", "y", "s")}
rng = np.random.default_rng(4)
seq_va = rng.choice(np.unique(hv["s"]), NVA, replace=False)
Xv, Yv, Sv, Pv = build("valid", hv, seq_va, rows_valid)
del hv
print(f"cargado en {time.time()-t0:.0f}s: train {len(Xt)} filas de {len(seq_tr)} secuencias, valid {len(Xv)} de {NVA}", flush=True)
Yc = np.clip(Yt, -2, 2)
M1 = np.array([[np.abs(Yc[St == s, k]).mean() for k in range(2)] for s in range(5)])
M2 = np.array([[(np.abs(Yc[St == s, k]) * Yc[St == s, k]).mean() for k in range(2)] for s in range(5)])
nraw = Xt.shape[1] - 130
sets = {"solo LSTM (h + p)": slice(nraw, None), "LSTM + features": slice(0, None)}
print(f"LSTM LR sola en estas filas: {score(Yv, Pv):.4f}")
for name, cols in sets.items():
    t0 = time.time()
    m = lgb.train(dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=63, bagging_fraction=0.5,
                       bagging_freq=1, feature_fraction=0.5, min_data_in_leaf=300, verbose=-1, num_threads=12),
                  lgb.Dataset(Xt[:, cols], St.astype(float)), num_boost_round=ROUNDS)
    P = m.predict(Xv[:, cols])
    qs = (P @ M2) / (P @ M1)
    line = f"{name:22s} estructural {score(Yv, qs):.4f}"
    for a in (0.25, 0.5, 0.75):
        line += f" | mezcla {a}: {score(Yv, (1-a)*Pv + a*qs):.4f}"
    print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
    np.save(SC + rf"\inst3_stack_{'f' if cols.start == 0 else 'h'}.npy", qs)
np.savez(SC + r"\inst3_stack_meta.npz", y=Yv, p=Pv, seq_va=seq_va)
