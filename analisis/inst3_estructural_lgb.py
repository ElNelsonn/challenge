import sys, time, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from instrumentos_comun import ROOT, SC, ENS, score
sys.path.insert(0, ROOT)
from utils import FEATURE_COLUMNS
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
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


def load(split, ids, step, scored):
    pf = pq.ParquetFile(ROOT + rf"\datasets\{split}.parquet")
    X, S, Y, SQ = [], [], [], []
    for g in ids:
        cols = ["seq_ix"] + FEATURE_COLUMNS + ["t0", "t1"] + (["is_scored"] if scored else [])
        t = pf.read_row_group(int(g), columns=cols, use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        y = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()]).astype(np.float64)
        sel = np.zeros(len(x), bool); sel[100:] = True
        if scored:
            sel &= t["is_scored"].to_numpy().astype(bool)
        else:
            sel[np.arange(len(x)) % step != 0] = False
        F = feats(x)
        X.append(F[sel]); Y.append(y[sel]); S.append(states(np.round(y[sel] / QT).astype(int)))
        SQ.append(np.stack([np.full(sel.sum(), int(t["seq_ix"][0].as_py())), np.where(sel)[0]], 1))
    return np.concatenate(X), np.concatenate(S), np.concatenate(Y), np.concatenate(SQ)


ct = np.load(SC + r"\classes_train.npz"); cv = np.load(SC + r"\classes_valid.npz")
rng = np.random.default_rng(4)
tr = rng.choice(np.where(ct["cls"] == 3)[0], NTR, replace=False)
va = rng.choice(np.where(cv["cls"] == 3)[0], NVA, replace=False)
t0 = time.time()
Xt, St, Yt, _ = load("train", tr, 4, False)
Xv, Sv, Yv, SQv = load("valid", va, 1, True)
print(f"cargado en {time.time()-t0:.0f}s: train {len(Xt)} filas, valid {len(Xv)} filas puntuadas; estados train {np.bincount(St, minlength=5)/len(St)}")
Yc = np.clip(Yt, -2, 2)
M1 = np.array([[np.abs(Yc[St == s, k]).mean() for k in range(2)] for s in range(5)])
M2 = np.array([[(np.abs(Yc[St == s, k]) * Yc[St == s, k]).mean() for k in range(2)] for s in range(5)])
print("E[|y|y]/E[|y|] por estado:\n", np.round(M2 / np.maximum(M1, 1e-9), 3))
t0 = time.time()
m = lgb.train(dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=63, bagging_fraction=0.5,
                   bagging_freq=1, feature_fraction=0.5, min_data_in_leaf=300, verbose=-1, num_threads=12),
              lgb.Dataset(Xt, St.astype(float)), num_boost_round=ROUNDS)
print(f"LightGBM {ROUNDS} rondas en {time.time()-t0:.0f}s")
P = m.predict(Xv)
qs = (P @ M2) / (P @ M1)
m.save_model(SC + r"\inst3_struct_lgb.txt")
np.savez(SC + r"\inst3_struct_P.npz", P=P, qs=qs, y=Yv, sq=SQv)


def member(name):
    zz = np.load(ENS[name]); pp = {int(s): i for i, s in enumerate(zz["seq_ix"])}
    A = zz["preds"]
    return A[np.array([pp[s] for s in SQv[:, 0]]), SQv[:, 1]].astype(np.float64)


ql = member("LR")
qe = (ql + member("LA") + member("long") + member("b256")) / 4
print(f"\nvalid inst 3 ({NVA} secuencias): LSTM LR {score(Yv, ql):.4f} | ensamble 4 {score(Yv, qe):.4f} | estructural LightGBM {score(Yv, qs):.4f}")
print(f"correlacion estructural vs ensamble: t0 {np.corrcoef(qs[:,0], qe[:,0])[0,1]:.3f} t1 {np.corrcoef(qs[:,1], qe[:,1])[0,1]:.3f}")
useq = np.unique(SQv[:, 0]); half = np.isin(SQv[:, 0], rng.permutation(useq)[: len(useq) // 2])
for base, name in ((ql, "LR"), (qe, "ensamble")):
    for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"   {name} mezcla a={a}: {score(Yv, (1-a)*base + a*qs):.4f}  (mitad1 {score(Yv[half], ((1-a)*base + a*qs)[half]):.4f} vs {score(Yv[half], base[half]):.4f} | mitad2 {score(Yv[~half], ((1-a)*base + a*qs)[~half]):.4f} vs {score(Yv[~half], base[~half]):.4f})")
np.savez(SC + r"\inst3_struct_valid.npz", qs=qs, ql=ql, qe=qe, y=Yv, sq=SQv, P=P)
m.save_model(SC + r"\inst3_struct_lgb.txt")
