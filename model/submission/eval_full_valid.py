r"""Evalúa modelos sobre TODAS las secuencias de valid (o las primeras N) con la métrica GLOBAL
(y la vieja por secuencia), para tener números comparables con el leaderboard.

  --baseline          baseline.onnx oficial (GRU numba de connectome/src/gru_numba.py)
  --gru RUTA.pt       best_model.pt de train_official_gpu_v1 (PyTorch en CPU, por batches)

Uso (desde challenge/):
  python model/submission/eval_full_valid.py --baseline --gru model/official_gpu/results/full_gru128x2_10ep/best_model.pt --n 1873
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
ROOT = CHALLENGE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHALLENGE / "model" / "lgbm_deltas"))
sys.path.insert(0, str(ROOT.parents[1] / "connectome" / "src"))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, BlockAccumulator, validate_sequence, DataPoint  # noqa: E402
from train_lgbm_deltas_v1 import GlobalWP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--gru", default=None)
    ap.add_argument("--n", type=int, default=1873)
    ap.add_argument("--out", default=str(HERE / "results" / "eval_full_valid.json"))
    args = ap.parse_args()
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    n = min(args.n, pf.num_row_groups)
    models = {}
    if args.baseline:
        from gru_numba import NumbaGRU2, load_weights_from_onnx
        models["baseline"] = ("numba", NumbaGRU2(load_weights_from_onnx(ROOT / "baseline" / "baseline.onnx")))
    if args.gru:
        import torch
        from torch import nn
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) - 2))
        ck = torch.load(args.gru, map_location="cpu", weights_only=False)
        net = nn.ModuleDict({"core": nn.GRU(ck["n_inputs"], ck["config"]["hidden"], num_layers=ck["config"]["layers"], batch_first=True),
                             "head": nn.Linear(ck["config"]["hidden"], 2)})
        net.load_state_dict(ck["state_dict"])
        net.eval()
        models["gru"] = ("torch", (net, ck["pipeline"]["mean"], ck["pipeline"]["std"]))
    accs = {k: (GlobalWP(), BlockAccumulator()) for k in models}
    t0 = time.time()
    for g in range(n):
        tb = pf.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(tb)
        X = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        m = need & tb["is_scored"].to_numpy().astype(bool)
        for name, (kind, obj) in models.items():
            if kind == "numba":
                P = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
                for s in range(SEQUENCE_LENGTH):
                    v = obj.predict(DataPoint(seq, s, bool(need[s]), X[s]))
                    if need[s]:
                        P[s] = v
            else:
                import torch
                net, mean, std = obj
                with torch.no_grad():
                    z = torch.from_numpy(((X - mean) / std).astype(np.float32)).unsqueeze(0)
                    P = net["head"](net["core"](z)[0]).squeeze(0).numpy()
            accs[name][0].add(Y, P, m)
            accs[name][1].add(Y, P, m)
        if (g + 1) % 100 == 0 or g + 1 == n:
            el = time.time() - t0
            line = "  ".join(f"{k}: G={a[0].result()['weighted_pearson']:.4f}" for k, a in accs.items())
            print(f"{g+1}/{n} seqs ({el/60:.1f} min, ~{el/(g+1)*(n-g-1)/60:.1f} min restantes)   {line}", flush=True)
    out = {}
    for k, (ga, ba) in accs.items():
        g, b = ga.result(), ba.result()
        out[k] = {"global": g, "per_sequence": {kk: b[kk] for kk in ("weighted_pearson", "t0", "t1", "eligible_blocks", "blocks")}}
        print(f"\n{k}: GLOBAL WP={g['weighted_pearson']:.4f} (t0={g['t0']:.4f}, t1={g['t1']:.4f})   por secuencia={b['weighted_pearson']:.4f}  [{n} seqs]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_seqs": n, "models": out}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
