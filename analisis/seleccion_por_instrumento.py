import numpy as np, json
from instrumentos_comun import *
from pool_modelos import POOL

seq, row, Y = load_cache()
cls = np.load(SC + r"\cls_scored.npy")
P = np.load(SC + r"\pool_scored.npy", mmap_mode="r")
names = json.load(open(SC + r"\pool_names.json"))
cost = np.array([POOL[n][2] for n in names])
BASE = [names.index(n) for n in ["masks", "LR", "LA", "long"]]
BUDGET, MAXK = 55.0, 6
seq_cls = {}
for s, c in zip(seq, cls):
    seq_cls.setdefault(int(s), int(c))
useq = np.array(sorted(seq_cls)); ucls = np.array([seq_cls[s] for s in useq])


def greedy(rows):
    Pr = np.asarray(P[:, rows], dtype=np.float32); Yr = Y[rows]
    chosen, tot, cur, best = [], 0.0, np.zeros((len(rows), 2)), -1
    while len(chosen) < MAXK:
        cand = None
        for m in range(len(names)):
            if m in chosen or tot + cost[m] > BUDGET:
                continue
            s = score(Yr, (cur + Pr[m]) / (len(chosen) + 1))
            if cand is None or s > cand[0]:
                cand = (s, m)
        if cand is None or cand[0] <= best + 1e-5:
            break
        best = cand[0]; chosen.append(cand[1]); tot += cost[cand[1]]; cur += Pr[cand[1]]
    return chosen


def mean_of(idx, rows):
    return np.mean([np.asarray(P[m, rows], dtype=np.float64) for m in idx], 0)


rng = np.random.default_rng(7)
res = {"perclass": [], "global": []}
for rep in range(3):
    A = set()
    for c in range(4):
        s = rng.permutation(useq[ucls == c]); A |= set(s[: len(s) // 2].tolist())
    inA = np.isin(seq, list(A))
    for tr, te in [(inA, ~inA), (~inA, inA)]:
        te_rows = np.where(te)[0]
        base = mean_of(BASE, te_rows)
        sb = score(Y[te_rows], base)
        st = np.zeros((len(te_rows), 2))
        sel = {}
        for c in range(4):
            ch = greedy(np.where(tr & (cls == c))[0])
            sel[c] = [names[m] for m in ch]
            mrows = cls[te_rows] == c
            st[mrows] = mean_of(ch, te_rows[mrows])
        gch = greedy(np.where(tr)[0])
        sg = score(Y[te_rows], mean_of(gch, te_rows))
        sp = score(Y[te_rows], st)
        res["perclass"].append(sp - sb); res["global"].append(sg - sb)
        pc = [round(score(Y[te_rows][cls[te_rows] == c], st[cls[te_rows] == c]) - score(Y[te_rows][cls[te_rows] == c], base[cls[te_rows] == c]), 4) for c in range(4)]
        print(f"rep {rep}: base {sb:.4f} | per-class sel {sp:.4f} ({sp-sb:+.4f}, by class {pc}) | global sel {sg:.4f} ({sg-sb:+.4f})", flush=True)
        print("   per-class sets:", sel, "| global set:", [names[m] for m in gch], flush=True)
print("mean honest gain per-class %+.5f  global %+.5f" % (np.mean(res["perclass"]), np.mean(res["global"])))
