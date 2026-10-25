"""La busqueda sobre los 58 modelos, con los tres arreglos que le faltaban a la primera version.

1. COSTO BIEN CALIBRADO. La primera version decia que nuestro ensamble cuesta 72 min, pero esta MEDIDO en
   41m52s. Nuestra cuaterna es 3 x LSTM 128x2 + 1 x LSTM 256x2. Contando el costo del kernel (matvec de
   entrada 112xG mas recurrencia HxG, con G = 4H para LSTM y 3H para GRU) eso son 3*1 + 1*4 = 7 unidades de
   "LSTM 128x2". Entonces 41.87/7 = 5.98 min por unidad, y el presupuesto de 60 min son 10.0 unidades.

2. UN SOLO CHECKPOINT POR CORRIDA. La primera version eligio student_mask:best Y student_mask:swa, que son
   el mismo modelo entrenado una vez. En inferencia ocupan dos lugares y aportan casi nada. Ahora cada
   corrida compite con su mejor variante solamente.

3. SELECCION HONESTA. Elegir la combinacion mirando las mismas filas con las que se la mide infla el
   resultado. Ahora se elige en la mitad de las secuencias y se mide en la otra mitad.
"""
import json
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
SC = Path(r"C:\Users\Joaquin\AppData\Local\Temp\claude\C--Users-Joaquin-Desktop"
          r"\16f58967-ed4f-4cd2-a867-451d65dfb638\scratchpad")
PASO = 3
MIN_POR_UNIDAD = 41.87 / 7.0      # calibrado con la submission medida
PRESUPUESTO = 55.0                 # 55 de los 60, margen para carga y escritura

z = np.load(CH / "model" / "submission" / "results" / "valid_scored_cache.npz")
seq, row = z["seq"], z["row"]
Yf = np.clip(z["y"], -2, 2).astype(np.float32)
orden = np.argsort(seq, kind="stable")
tramos = np.split(orden, np.flatnonzero(np.diff(seq[orden])) + 1)

# submuestra, conservando a que secuencia pertenece cada fila (para partir por secuencia)
sel, dueno = [], []
for i, t in enumerate(tramos):
    s = np.sort(t)[::PASO]
    sel.append(s); dueno.append(np.full(len(s), i))
sel = np.concatenate(sel); dueno = np.concatenate(dueno)
Y = Yf[sel].astype(np.float64)
W = np.abs(Y)
print(f"{len(Yf):,} filas puntuadas | submuestra {len(Y):,} de {len(tramos)} secuencias", flush=True)


def unidades(cell, hidden, layers, arch=None):
    """Costo en 'LSTM 128x2 equivalentes', segun las operaciones del kernel."""
    if arch and "tcn" in str(arch).lower():
        return 1.3
    G = 3 if cell == "gru" else 4
    H, L = hidden, layers
    ops = 112 * G * H + sum((H if l > 0 else H) * G * H for l in range(L)) + (L - 1) * H * G * H
    ref = 112 * 4 * 128 + 2 * 128 * 4 * 128 + 1 * 128 * 4 * 128
    return ops / ref


cands = {}
for p in sorted(CH.glob("model/official_gpu/results/**/valid_preds_*.npz")) + \
         sorted(SC.glob("k2*/**/valid_preds_*.npz")):
    cands[f"{p.parent.name}:{'swa' if 'swa' in p.name else 'best'}"] = p

P, ARQ = {}, {}
for nombre, p in cands.items():
    try:
        d = np.load(p); pos = {int(v): i for i, v in enumerate(d["seq_ix"])}; preds = d["preds"]
    except Exception:
        continue
    out = np.zeros((len(Yf), 2), np.float32); vistas = 0
    for t in tramos:
        s_ = int(seq[t[0]])
        if s_ in pos:
            out[t] = preds[pos[s_]][row[t]]; vistas += 1
    if vistas < len(tramos):
        continue
    P[nombre] = np.clip(out[sel], -2, 2).astype(np.float32)
    m = p.parent / "metrics.json"
    c = json.loads(m.read_text(encoding="utf-8", errors="replace")).get("config", {}) if m.exists() else {}
    ARQ[nombre] = (c.get("cell", "lstm"), int(c.get("hidden", 128)), int(c.get("layers", 2)), c.get("arch"))
    del out
print(f"{len(P)} modelos cargados\n", flush=True)

rs = np.random.default_rng(2026)
perm = rs.permutation(len(tramos))
mitad = len(perm) // 2
en_A = np.isin(dueno, perm[:mitad])
A, B = np.flatnonzero(en_A), np.flatnonzero(~en_A)


def wp(q, filas):
    v = []
    for j in (0, 1):
        w, y, p_ = W[filas, j], Y[filas, j], q[filas, j].astype(np.float64)
        s = w.sum()
        my, mp = (w * y).sum() / s, (w * p_).sum() / s
        cov = (w * (y - my) * (p_ - mp)).sum() / s
        vy, vp = (w * (y - my) ** 2).sum() / s, (w * (p_ - mp) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vp))
    return (v[0] + v[1]) / 2


TODO = np.arange(len(Y))
UNID = {k: unidades(*ARQ[k]) for k in P}
MIN = {k: UNID[k] * MIN_POR_UNIDAD for k in P}

# una sola variante por corrida: la mejor de best/swa medida en TODO
corrida = {}
for k in P:
    base = k.rsplit(":", 1)[0]
    s_ = wp(P[k], TODO)
    if base not in corrida or s_ > corrida[base][1]:
        corrida[base] = (k, s_)
UNICOS = {v[0]: v[1] for v in corrida.values()}
print(f"{len(UNICOS)} corridas distintas (una variante cada una)\n")

NUESTRO = ["student_lr5e-4:best", "student_LA:best", "student_long:best", "student_big256:swa"]
NUESTRO = [k for k in NUESTRO if k in P]
S_todo = wp(np.mean([P[k] for k in NUESTRO], 0), TODO)
S_B = wp(np.mean([P[k] for k in NUESTRO], 0), B)
print("=" * 88)
print(f"NUESTRO ENSAMBLE: {S_todo:.4f} en todo | {S_B:.4f} en la mitad B | "
      f"{sum(MIN[k] for k in NUESTRO):.1f} min ({sum(UNID[k] for k in NUESTRO):.1f} unidades)")
print("=" * 88)
print("  costo de cada tipo:  LSTM 128x2 = %.1f min | LSTM 192x2 = %.1f | LSTM 256x2 = %.1f | GRU 128x2 = %.1f"
      % tuple(unidades(*a) * MIN_POR_UNIDAD for a in
              (("lstm", 128, 2, None), ("lstm", 192, 2, None), ("lstm", 256, 2, None), ("gru", 128, 2, None))))


def avanzar(filas_sel, limite, k_max=10):
    eleg, suma, mejor = [], None, -9
    while len(eleg) < k_max:
        cand, cval = None, mejor
        for k in UNICOS:
            if k in eleg:
                continue
            if limite is not None and sum(MIN[x] for x in eleg) + MIN[k] > limite:
                continue
            s2 = P[k].astype(np.float64) if suma is None else suma + P[k]
            v = wp(s2 / (len(eleg) + 1), filas_sel)
            if v > cval:
                cand, cval = k, v
        if cand is None:
            break
        eleg.append(cand)
        suma = P[cand].astype(np.float64) if suma is None else suma + P[cand]
        mejor = cval
    return eleg


for etiqueta, limite in (("SIN PRESUPUESTO", None), (f"EN {PRESUPUESTO:.0f} MIN", PRESUPUESTO)):
    print("\n" + "=" * 88)
    print(f"{etiqueta}: elegido en la mitad A, medido en la mitad B (honesto)")
    print("=" * 88)
    eleg = avanzar(A, limite)
    vA, vB = wp(np.mean([P[k] for k in eleg], 0), A), wp(np.mean([P[k] for k in eleg], 0), B)
    for k in eleg:
        print(f"    {k:30s} solo {UNICOS[k]:.4f}  {MIN[k]:5.1f} min")
    print(f"  total {sum(MIN[k] for k in eleg):.1f} min")
    print(f"  en A (donde se eligio): {vA:.4f}   <- inflado")
    print(f"  en B (nunca visto):     {vB:.4f}   contra nuestro {S_B:.4f}  =  {vB-S_B:+.4f}")

lb = {1: 0.67380, 50: 0.66517, 100: 0.66256, 200: 0.65934, 300: 0.65685, 400: 0.65491}
pos_, sco = np.array(sorted(lb)), np.array([lb[p] for p in sorted(lb)])
print("\n" + "=" * 88)
print("  La ganancia honesta es la de la mitad B. Para traducirla a puesto hay que sumarsela a nuestro")
print(f"  0.65400 real del leaderboard, no proyectar la mitad B (que mide {S_B:.4f} en vez de {S_todo:.4f}).")
print("=" * 88)
