"""LA BUSQUEDA QUE NUNCA HICE BIEN: ensamblar TODO lo que entrenamos en dos meses, con presupuesto.

Hasta ahora siempre compare cuaternas de modelos parecidos. Pero tenemos ~60 modelos guardados de TODAS las
ramas: las alumnas destiladas, los sin destilar del sprint 4, las convolucionales (TCN), la mascara, seqwp,
cross, ema, half1/half2, los grandes 192/256, los del sprint 21, los div del 24 y la normalizacion adaptativa.

Y el presupuesto no se mide en "4 modelos": se mide en MINUTOS. Un GRU cuesta 3/4 de un LSTM, uno de 256
cuesta 4 veces uno de 128. Asi que puede convenir cambiar una red grande por dos chicas distintas.

Mide:
  1. el puntaje individual de cada modelo
  2. la mejor combinacion SIN presupuesto (si aca no hay nada, no hay nada)
  3. la mejor combinacion que entra en 60 min de inferencia, con el costo real de cada arquitectura

Busqueda sobre una submuestra de filas para que entre en memoria; los finalistas se vuelven a medir con
TODAS las filas puntuadas.
"""
import itertools
import json
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")
PASO = 3        # 1 de cada 3 filas puntuadas para la busqueda

# ---------------------------------------------------------------- costo de inferencia
# Referencia medida: 4 x LSTM 128x2 = 41 min de los 60. O sea 10.25 min por LSTM 128x2.
# Un GRU tiene 3 puertas contra 4 -> 0.75. El costo va con hidden^2.
BASE_LSTM = 41.0 / 4


def costo(cell, hidden, layers, arch=None):
    if arch and "tcn" in str(arch).lower():
        return BASE_LSTM * 1.3      # medido en el sprint 7: una TCN sale mas caro que una LSTM 128x2
    g = 0.75 if cell == "gru" else 1.0
    return BASE_LSTM * g * (hidden / 128.0) ** 2 * (layers / 2.0)


# ---------------------------------------------------------------- candidatos
cands = {}
for p in sorted(CH.glob("model/official_gpu/results/**/valid_preds_*.npz")) + \
         sorted(SC.glob("k2*/**/valid_preds_*.npz")):
    etiqueta = ("swa" if "swa" in p.name else "best")
    nombre = f"{p.parent.name}:{etiqueta}"
    cands[nombre] = p

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Yf = np.clip(z["y"], -2, 2).astype(np.float32)
orden = np.argsort(seq, kind="stable")
tramos = np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)
sel = np.sort(np.concatenate(tramos))[::PASO]      # submuestra estable
Y = Yf[sel].astype(np.float64)
W = np.abs(Y)
print(f"{len(Yf):,} filas puntuadas | submuestra de busqueda: {len(Y):,}", flush=True)

ARQ = {}
P = {}
for nombre, p in cands.items():
    try:
        d = np.load(p)
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
        preds = d["preds"]
    except Exception as e:
        print(f"  (salteo {nombre}: {e})")
        continue
    out = np.zeros((len(Yf), 2), np.float32)
    vistas = 0
    for t in tramos:
        s_ = int(seq[t[0]])
        if s_ in pos:
            out[t] = preds[pos[s_]][row[t]]
            vistas += 1
    if vistas < len(tramos):
        print(f"  (salteo {nombre}: solo {vistas}/{len(tramos)} secuencias)")
        continue
    P[nombre] = np.clip(out[sel], -2, 2).astype(np.float32)
    m = p.parent / "metrics.json"
    if m.exists():
        c = json.loads(m.read_text(encoding="utf-8", errors="replace")).get("config", {})
        ARQ[nombre] = (c.get("cell", "lstm"), int(c.get("hidden", 128)),
                       int(c.get("layers", 2)), c.get("arch"))
    else:
        ARQ[nombre] = ("lstm", 128, 2, None)
    del out
print(f"\n{len(P)} modelos cargados con las {len(tramos)} secuencias completas\n", flush=True)


def wp(q):
    v = []
    for j in (0, 1):
        w, y, p_ = W[:, j], Y[:, j], q[:, j].astype(np.float64)
        s = w.sum()
        my, mp = (w * y).sum() / s, (w * p_).sum() / s
        cov = (w * (y - my) * (p_ - mp)).sum() / s
        vy, vp = (w * (y - my) ** 2).sum() / s, (w * (p_ - mp) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vp))
    return (v[0] + v[1]) / 2


SOLO = {k: wp(P[k]) for k in P}
COSTO = {k: costo(*ARQ[k]) for k in P}
print("=" * 86)
print("PUNTAJE INDIVIDUAL Y COSTO (los 18 mejores)")
print("=" * 86)
for k in sorted(SOLO, key=SOLO.get, reverse=True)[:18]:
    c = ARQ[k]
    print(f"  {k:34s} {SOLO[k]:.4f}   {c[0]} h{c[1]} x{c[2]}{' tcn' if c[3] and 'tcn' in str(c[3]).lower() else ''}"
          f"   {COSTO[k]:5.1f} min")

NUESTRO = [k for k in P if k.startswith(("student_lr5e-4", "student_LA", "student_long", "student_big256"))]
NUESTRO = [k for k in NUESTRO if (k.endswith(":swa") if "big256" in k else k.endswith(":best"))]
if len(NUESTRO) == 4:
    S = wp(np.mean([P[k] for k in NUESTRO], 0))
    print(f"\n  NUESTRO ENSAMBLE ACTUAL: {S:.4f}  ({sum(COSTO[k] for k in NUESTRO):.0f} min)")
else:
    S = max(SOLO.values()); print(f"\n  (no identifique la cuaterna actual: {NUESTRO})")


def avanzar(permitidos, limite_min=None, k_max=8):
    """Seleccion hacia adelante: agrega el modelo que mas suba, respetando el presupuesto."""
    eleg, suma, mejor = [], None, -9
    for _ in range(k_max):
        cand, cval = None, mejor
        for k in permitidos:
            if k in eleg:
                continue
            if limite_min is not None and sum(COSTO[x] for x in eleg) + COSTO[k] > limite_min:
                continue
            s2 = P[k].astype(np.float64) if suma is None else suma + P[k]
            v = wp(s2 / (len(eleg) + 1))
            if v > cval:
                cand, cval = k, v
        if cand is None:
            break
        eleg.append(cand)
        suma = P[cand].astype(np.float64) if suma is None else suma + P[cand]
        mejor = cval
    return mejor, eleg


print("\n" + "=" * 86)
print("2. LA MEJOR COMBINACION SIN PRESUPUESTO  (si aca no hay nada, no hay nada)")
print("=" * 86)
v_libre, e_libre = avanzar(list(P), None, 8)
print(f"  {v_libre:.4f} con {len(e_libre)} modelos ({sum(COSTO[k] for k in e_libre):.0f} min, "
      f"{'NO ENTRA' if sum(COSTO[k] for k in e_libre) > 55 else 'entra'})")
for k in e_libre:
    print(f"    {k}")
print(f"  sobre nuestro {S:.4f}: {v_libre-S:+.4f}")

print("\n" + "=" * 86)
print("3. LA MEJOR QUE ENTRA EN PRESUPUESTO (55 min de los 60, margen para la carga)")
print("=" * 86)
v_pres, e_pres = avanzar(list(P), 55.0, 8)
print(f"  {v_pres:.4f} con {len(e_pres)} modelos ({sum(COSTO[k] for k in e_pres):.1f} min)")
for k in e_pres:
    print(f"    {k:34s} {SOLO[k]:.4f}  {COSTO[k]:5.1f} min")
print(f"  sobre nuestro {S:.4f}: {v_pres-S:+.4f}")

lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos_, sco = np.array(sorted(lb)), np.array([lb[p] for p in sorted(lb)])
print("\n" + "=" * 86)
print("  (ojo: estos numeros son de la submuestra de 1 cada 3 filas, hay que reconfirmar los finalistas)")
for nombre, val in (("ahora", S), ("mejor con presupuesto", v_pres), ("mejor sin presupuesto", v_libre)):
    x = val - 0.0323
    pu = ">420" if x <= sco[-1] else int(round(np.interp(-x, -sco, pos_)))
    print(f"  {nombre:24s} {val:.4f} -> LB {x:.4f} -> puesto ~{pu}")
print("=" * 86)
np.save(SC / "ens_elegidos.npy", np.array(e_pres, dtype=object), allow_pickle=True)
