r"""Mide µs/fila de varios builds de submissions/ con el callback oficial, núcleo fijo, y proyecta el tiempo
de servidor anclado a un build cuyo tiempo real en wundernn se conoce (por defecto S4 = 33 min).

Uso (desde challenge/):  python model/submission/kernel_bench/time_anchored.py ens2_B_Cswa=33 ens4_y2_Eswa_X_Zswa_int8 ...
El primer argumento con "=min" es el ancla. Round-robin de 3 repeticiones sobre 3 secuencias reales.
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow.parquet as pq

CH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CH.parent))
from utils import DataPoint, FEATURE_COLUMNS, SEQUENCE_LENGTH, validate_sequence  # noqa: E402


def load(build):
    d = str(CH / "submissions" / build)
    sys.path.insert(0, d)
    sys.modules.pop("solution", None)
    m = importlib.import_module("solution")
    sys.path.pop(0)
    return m.PredictionModel()


def main():
    p = psutil.Process()
    p.cpu_affinity([6])
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    specs = [a.split("=") for a in sys.argv[1:]]
    anchor_name, anchor_min = next((n, float(m)) for n, *m in specs if m for m in m)
    pf = pq.ParquetFile(CH.parent / "datasets" / "valid.parquet")
    seqs = []
    for g in range(3):
        tb = pf.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(tb)
        seqs.append((seq, need, np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)))
    models = [(n, load(n)) for n, *_ in specs]
    best = {n: 1e9 for n, _ in models}
    for _ in range(3):
        for n, pm in models:
            for seq, need, X in seqs:
                t = time.perf_counter()
                for s in range(SEQUENCE_LENGTH):
                    pm.predict(DataPoint(seq, s, bool(need[s]), X[s]))
                best[n] = min(best[n], (time.perf_counter() - t) / SEQUENCE_LENGTH * 1e6)
    a = best[anchor_name]
    for n, _ in models:
        print(f"{n:34s} {best[n]:6.1f} us/fila -> servidor ~ {anchor_min * best[n] / a:4.1f} min (anclado a {anchor_name} = {anchor_min:.0f} min)")


if __name__ == "__main__":
    main()
