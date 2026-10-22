"""¿Destilar nos mató el ensamble?

Nuestros 4 modelos del ensamble están TODOS destilados de los MISMOS 4 maestros. Correlacionan 0.98 entre sí
y por eso promediarlos vale apenas +0.002 sobre el mejor de ellos. En una competencia normal, promediar 4
modelos distintos vale entre +0.005 y +0.015.

Nunca medí lo que corresponde: cuánto correlacionan entre sí los modelos SIN destilar. Son peores solos
(0.678 contra 0.684) pero si correlacionan 0.96 en vez de 0.985, el promedio de 4 puede terminar más arriba.

La cuenta: k modelos de puntaje s que correlacionan r entre sí dan un promedio de  s * sqrt(k / (1+(k-1)r)).
  4 destilados:    0.684 con r=0.985 -> 0.6879
  4 sin destilar:  0.679 con r=0.96  -> 0.6894
  4 sin destilar:  0.679 con r=0.94  -> 0.6936
"""
import itertools
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
R = CH / "model" / "official_gpu" / "results"

SIN = {"l1": R / "sprint4_lstm/results/l1/valid_preds_best.npz",
       "l2": R / "sprint4_lstm/results/l2/valid_preds_best.npz",
       "l3": R / "sprint4_lstm/results/l3/valid_preds_best.npz",
       "y1": R / "sprint4_x/results/y1/valid_preds_best.npz",
       "y2": R / "sprint4_x/results/y2/valid_preds_best.npz",
       "y3": R / "sprint4_x/results/y3/valid_preds_best.npz",
       "seqwp": R / "sprint10_notebookd0e4cb9378/results/student_seqwp/valid_preds_best.npz"}
CON = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
       "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
       "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
       "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz",
       "mask": R / "s15_notebookfd82629297/results/student_mask/valid_preds_swa.npz"}

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Y = np.clip(z["y"], -2, 2).astype(np.float64)
W = np.abs(Y)
print(f"{len(Y):,} filas puntuadas, {len(np.unique(seq)):,} secuencias de valid\n")

orden = np.argsort(seq, kind="stable")
bordes = np.flatnonzero(np.diff(seq[orden])) + 1
tramos = np.split(orden, bordes)

P = {}
for grupo, nombre in ((SIN, "sin destilar"), (CON, "destilados")):
    for k, p in grupo.items():
        if not p.exists():
            print(f"  (falta {k}: {p.name})")
            continue
        d = np.load(p)
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
        preds = d["preds"]
        out = np.zeros((len(Y), 2), np.float32)
        for tr in tramos:
            s_ = int(seq[tr[0]])
            if s_ in pos:
                out[tr] = preds[pos[s_]][row[tr]]
        P[k] = np.clip(out.astype(np.float64), -2, 2)
        print(f"  {k} ({nombre}) listo", flush=True)


def wp(p):
    v = []
    for j in (0, 1):
        w, y, q = W[:, j], Y[:, j], p[:, j]
        s = w.sum()
        my, mq = (w * y).sum() / s, (w * q).sum() / s
        cov = (w * (y - my) * (q - mq)).sum() / s
        vy, vq = (w * (y - my) ** 2).sum() / s, (w * (q - mq) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vq))
    return (v[0] + v[1]) / 2


sub = slice(None, None, 5)
for grupo, nombre in ((SIN, "SIN DESTILAR"), (CON, "DESTILADOS")):
    ks = [k for k in grupo if k in P]
    print(f"\n{'='*58}\n{nombre}\n{'='*58}")
    for k in ks:
        print(f"  {k:8s} solo: {wp(P[k]):.4f}")
    rs = [np.corrcoef(P[a][sub, 0], P[b][sub, 0])[0, 1] for a, b in itertools.combinations(ks, 2)]
    print(f"\n  correlacion ENTRE ELLOS: min {min(rs):.4f} | media {np.mean(rs):.4f} | max {max(rs):.4f}")
    mejor4 = max(itertools.combinations(ks, min(4, len(ks))),
                 key=lambda c: wp(np.mean([P[k] for k in c], 0)))
    e = wp(np.mean([P[k] for k in mejor4], 0))
    solo = max(wp(P[k]) for k in mejor4)
    print(f"  promedio de {len(mejor4)} ({'+'.join(mejor4)}): {e:.4f}")
    print(f"  el mejor solo:                 {solo:.4f}")
    print(f"  lo que gana promediar:         {e - solo:+.4f}")

print(f"\n{'='*58}\nMEZCLAR LAS DOS FAMILIAS\n{'='*58}")
ks_s = [k for k in SIN if k in P]
ks_c = [k for k in CON if k in P]
if ks_s and ks_c:
    cru = [np.corrcoef(P[a][sub, 0], P[b][sub, 0])[0, 1] for a in ks_s for b in ks_c]
    print(f"  correlacion entre familias: media {np.mean(cru):.4f}")
    mejor, mejorc = -9, None
    for n in (4, 5, 6):
        for c in itertools.combinations(ks_s + ks_c, n):
            v = wp(np.mean([P[k] for k in c], 0))
            if v > mejor:
                mejor, mejorc = v, c
    print(f"  mejor mezcla de las dos familias: {'+'.join(mejorc)} = {mejor:.4f}")
    base = wp(np.mean([P[k] for k in ("LR", "LA", "long", "b256") if k in P], 0))
    print(f"  nuestro ensamble actual (LR+LA+long+b256): {base:.4f}")
    print(f"  diferencia: {mejor - base:+.4f}")
