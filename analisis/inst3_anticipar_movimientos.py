import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from scipy.stats import rankdata
from instrumentos_comun import ROOT, SC, ENS
sys.path.insert(0, ROOT)
from utils import FEATURE_COLUMNS
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 150
QT = 0.65686
I0 = [j for j, c in enumerate(FEATURE_COLUMNS) if c.startswith("i0_")]


def feats(x):
    out = [x]
    for lag in (1, 10, 100):
        d = np.zeros_like(x[:, I0]); d[lag:] = x[lag:, I0] - x[:-lag, I0]
        out.append(d)
    return np.hstack(out).astype(np.float32)


def load(split, ids, step, scored_only):
    pf = pq.ParquetFile(ROOT + rf"\datasets\{split}.parquet")
    X, L, R, G = [], [], [], []
    for g in ids:
        cols = FEATURE_COLUMNS + ["t0", "t1"] + (["is_scored"] if scored_only else [])
        t = pf.read_row_group(int(g), columns=cols, use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        T = np.round(np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()]) / QT).astype(int)
        F = feats(x)
        sel = np.zeros(len(x), bool); sel[100:] = True
        if scored_only:
            sel &= t["is_scored"].to_numpy().astype(bool)
        else:
            sel[np.arange(len(x)) % step != 0] = False
        regA = ((T[:, 0] == 0) & (T[:, 1] == -1)) | ((T[:, 0] > 0) & (T[:, 1] == -3))
        regB = ((T[:, 0] == -1) & (T[:, 1] == 0)) | ((T[:, 0] == -3) & (T[:, 1] > 0))
        mv = (T[:, 0] > 0) | (T[:, 1] > 0)
        reg = np.where(regA, 0, np.where(regB, 1, -1))
        k = sel & (reg >= 0)
        X.append(F[k]); L.append(mv[k]); R.append(reg[k]); G.append(np.full(k.sum(), g))
    return np.concatenate(X), np.concatenate(L), np.concatenate(R), np.concatenate(G)


def auc(lab, s):
    r = rankdata(s); n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


ct = np.load(SC + r"\classes_train.npz"); cv = np.load(SC + r"\classes_valid.npz")
rng = np.random.default_rng(4)
tr = rng.choice(np.where(ct["cls"] == 3)[0], NTR, replace=False)
va = rng.choice(np.where(cv["cls"] == 3)[0], NVA, replace=False)
Xt, Lt, Rt, _ = load("train", tr, 4, False)
Xv, Lv, Rv, Gv = load("valid", va, 1, True)
print(f"train {len(Xt)} filas (mueve {Lt.mean():.3f}), valid {len(Xv)} filas puntuadas (mueve {Lv.mean():.3f})")
z = np.load(ENS["LR"]); pos = {int(s): i for i, s in enumerate(z["seq_ix"])}
pfv = pq.ParquetFile(ROOT + r"\datasets\valid.parquet")
Q = []
for g in va:
    t = pfv.read_row_group(int(g), columns=["seq_ix", "is_scored", "t0", "t1"], use_threads=False)
    s = int(t["seq_ix"][0].as_py()); sc = t["is_scored"].to_numpy().astype(bool); sc[:100] = False
    T = np.round(np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()]) / QT).astype(int)
    regA = ((T[:, 0] == 0) & (T[:, 1] == -1)) | ((T[:, 0] > 0) & (T[:, 1] == -3))
    regB = ((T[:, 0] == -1) & (T[:, 1] == 0)) | ((T[:, 0] == -3) & (T[:, 1] > 0))
    k = sc & (regA | regB)
    Q.append(z["preds"][pos[s]][k].astype(np.float64))
Q = np.concatenate(Q)
assert len(Q) == len(Xv)
for r, name in ((0, "regimen A: sube vs sigue"), (1, "regimen B: baja vs sigue")):
    mt, mv_ = Rt == r, Rv == r
    lr_score = Q[mv_, 0] - Q[mv_, 1] if r == 0 else Q[mv_, 1] - Q[mv_, 0]
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, bagging_fraction=0.5, bagging_freq=1,
                       feature_fraction=0.5, min_data_in_leaf=200, verbose=-1, num_threads=12),
                  lgb.Dataset(Xt[mt], Lt[mt].astype(float)), num_boost_round=400)
    p = m.predict(Xv[mv_])
    print(f"{name}: tasa {Lv[mv_].mean():.3f} | AUC LSTM LR {auc(Lv[mv_], lr_score):.3f} | AUC LightGBM features {auc(Lv[mv_], p):.3f}")
    imp = np.argsort(-m.feature_importance("gain"))[:12]
    names = FEATURE_COLUMNS + [f"d{l}_{FEATURE_COLUMNS[j]}" for l in (1, 10, 100) for j in I0]
    print("   top:", ", ".join(names[i] for i in imp))
