import numpy as np
from instrumentos_comun import *
seq, row, Y = load_cache()
cls = np.load(SC + r"\cls_scored.npy")
Q = np.load(SC + r"\ens_scored.npy").astype(np.float64)
k = cls == 3
y, q = Y[k], Q[k]
T = np.round(y / 0.65686).astype(int)
st = np.full(len(T), "otro", dtype=object)
st[(T[:, 0] == 0) & (T[:, 1] == -1)] = "A(0,-1)"
st[(T[:, 0] == -1) & (T[:, 1] == 0)] = "B(-1,0)"
st[(T[:, 0] > 0) & (T[:, 1] == -3)] = "sube(+k,-3)"
st[(T[:, 0] == -3) & (T[:, 1] > 0)] = "baja(-3,+k)"
yc = np.clip(y, -2, 2); w = np.abs(yc)
print(f"inst 3: {len(y)} filas, WP {score(y, q):.4f}")
print(f"{'estado':12s} {'filas':>7s} {'peso t0':>8s} {'peso t1':>8s} {'pred t0':>8s} {'pred t1':>8s} {'sd t0':>6s} {'sd t1':>6s}")
for s in ["A(0,-1)", "B(-1,0)", "sube(+k,-3)", "baja(-3,+k)", "otro"]:
    m = st == s
    print(f"{s:12s} {m.mean():7.3f} {w[m,0].sum()/w[:,0].sum():8.3f} {w[m,1].sum()/w[:,1].sum():8.3f} {q[m,0].mean():+8.3f} {q[m,1].mean():+8.3f} {q[m,0].std():6.3f} {q[m,1].std():6.3f}")
print("k de los movimientos (t0 en sube):", np.bincount(np.clip(T[st == 'sube(+k,-3)', 0], 0, 8)))
m = st == "otro"
u, c = np.unique([tuple(r) for r in T[m]], axis=0, return_counts=True)
print("otros estados mas comunes:", [(tuple(a), int(b)) for a, b in sorted(zip(u.tolist(), c), key=lambda z: -z[1])[:8]])


def oracle(mask_states, how):
    q2 = q.copy()
    for s in mask_states:
        m = st == s
        if how == "media":
            q2[m] = y[m].mean(0)
        else:
            q2[m] = q[m].mean(0)
    return score(y, q2)


print("\nOraculos (reemplazan la prediccion de esos estados por el target medio del estado):")
for ss in [["A(0,-1)", "B(-1,0)"], ["A(0,-1)"], ["B(-1,0)"], ["sube(+k,-3)", "baja(-3,+k)"], ["sube(+k,-3)"], ["baja(-3,+k)"]]:
    print(f"  {'+'.join(ss):28s} {oracle(ss, 'media'):.4f}")
qq = q.copy(); nm = np.isin(st, ["A(0,-1)", "B(-1,0)"])
qq[nm] = q[nm].mean(0)
print(f"  sin distinguir A de B (constante comun en no-movimiento): {score(y, qq):.4f}")
