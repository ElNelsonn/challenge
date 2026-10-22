"""¿Cada secuencia tiene su propia escala, y se la estamos escondiendo a la red?

En septiembre medimos algo que nunca explotamos: si pudiéramos multiplicar las predicciones de cada secuencia
por el factor correcto (el "oráculo"), ganaríamos **+0.047**. Eso prueba que las secuencias tienen escalas muy
distintas entre sí. Intentamos estimar ese factor DESPUÉS, desde las features, y no se pudo (R² 0.05-0.15).

Pero nunca probamos lo otro: **normalizamos las 112 features con una media y un desvío GLOBALES**, calculados
una vez sobre una muestra de todo train. Si la secuencia A se mueve 3 veces más que la B, a la red le llegan
números 3 veces más grandes en A, y tiene que aprender sola a compensarlo gastando capacidad.

La alternativa es normalizar **dentro de cada secuencia**, con un promedio y un desvío móviles que solo miran
el pasado (causal, así que es legal). Así cada secuencia le llega a la red en la misma escala.

Acá se mide cuán grande es el problema: cuánto varía el desvío de cada feature de una secuencia a otra.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
DATOS = CH.parent / "datasets" / "valid.parquet"
N_SEQ = 150

f = pq.ParquetFile(str(DATOS))
sch = f.schema_arrow
FEATS = [n for n, t in zip(sch.names, sch.types) if pa.types.is_floating(t) and n not in ("t0", "t1")]

desvios, escalas_y = [], []
for g in range(min(N_SEQ, f.num_row_groups)):
    t = f.read_row_group(g, columns=FEATS + ["t0", "t1"], use_threads=False)
    x = np.column_stack([t[c].to_numpy().astype(np.float64) for c in FEATS])
    x[~np.isfinite(x)] = np.nan
    desvios.append(np.nanstd(x, axis=0))
    y = np.column_stack([np.clip(t[c].to_numpy().astype(np.float64), -2, 2) for c in ("t0", "t1")])
    escalas_y.append(np.abs(y).mean())
D = np.array(desvios)          # (secuencias, features)
print(f"{len(D)} secuencias, {D.shape[1]} features\n")

# cuanto varia el desvio de cada feature entre secuencias
rel = D / np.nanmedian(D, axis=0)
p10, p90 = np.nanpercentile(rel, [10, 90], axis=0)
razon = p90 / np.maximum(p10, 1e-9)
orden = np.argsort(-razon)
print("features cuya escala MAS cambia de una secuencia a otra")
print(f"  {'feature':10s} {'p10':>8s} {'p90':>8s} {'p90/p10':>9s}")
for i in orden[:12]:
    print(f"  {FEATS[i]:10s} {p10[i]:8.3f} {p90[i]:8.3f} {razon[i]:9.2f}x")
print(f"\n  mediana de p90/p10 sobre las 112 features: {np.nanmedian(razon):.2f}x")
print(f"  features con mas de 2x de variacion: {int((razon > 2).sum())} de {len(razon)}")
print(f"  features con mas de 3x: {int((razon > 3).sum())}")

ey = np.array(escalas_y)
print(f"\nescala del TARGET por secuencia (|y| promedio):")
print(f"  p10 {np.percentile(ey,10):.4f} | mediana {np.median(ey):.4f} | p90 {np.percentile(ey,90):.4f} "
      f"| p90/p10 {np.percentile(ey,90)/np.percentile(ey,10):.2f}x")

# ¿la escala del target se puede anticipar desde la escala de las features? (causalmente seria con el pasado)
vol = np.nanmean(rel[:, orden[:20]], axis=1)
r = np.corrcoef(vol, ey)[0, 1]
print(f"\ncorrelacion entre la escala de las features y la escala del target: {r:.3f}")
print("(si es alta, normalizar las entradas por secuencia tambien normaliza implicitamente la salida,")
print(" que es justo lo que el oraculo de +0.047 decia que faltaba)")
