r"""Evalúa varias GRU (y todas sus combinaciones promediadas) sobre valid con la métrica GLOBAL.

Acepta dos formatos de checkpoint:
  * train_official_gpu_v1: {state_dict, config{cell,hidden,layers}, pipeline{mean,std}, n_inputs}
  * temporal_experiment (Santiago): {model, architecture{width,hidden,layers}, normalizer{mean,std,names}}
Solo modelos de 112 entradas crudas. GPU si hay, si no CPU por batches.

Uso:
  python eval_ensemble.py --valid /kaggle/input/.../valid.parquet --n all \
      --models A=/path/a.pt B=/path/b.pt C=/path/c.pt
Imprime WP global de cada modelo y de cada combinación (promedio simple), y el tiempo estimado
de inferencia en 1 CPU (µs/fila) de cada combinación para saber si entra en 60 min.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

FEATURES = [f"i{i}_{g}{j}" for i in range(2) for g, n in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4)) for j in range(n)] + [f"a{i}" for i in range(8)]
SEQ, WARMUP = 20000, 99


class GlobalWP:
    def __init__(self):
        self.s = np.zeros((2, 6), np.float64)

    def add(self, y, p, mask):
        y = np.clip(y[mask].astype(np.float64), -2, 2)
        p = np.clip(p[mask].astype(np.float64), -2, 2)
        w = np.abs(y)
        for j in range(2):
            wj, yj, pj = w[:, j], y[:, j], p[:, j]
            self.s[j] += [wj.sum(), (wj * yj).sum(), (wj * pj).sum(), (wj * yj * yj).sum(), (wj * pj * pj).sum(), (wj * yj * pj).sum()]

    def result(self):
        out = {}
        for j, name in enumerate(("t0", "t1")):
            W, wy, wp, wyy, wpp, wyp = self.s[j]
            my, mp = wy / W, wp / W
            cov, vy, vp = wyp / W - my * mp, wyy / W - my * my, wpp / W - mp * mp
            out[name] = float(np.clip(cov / np.sqrt(vy * vp), -1, 1)) if vy > 1e-16 and vp > 1e-16 else 0.0
        out["wp"] = (out["t0"] + out["t1"]) / 2
        return out


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        sd, H, L, n_in = ck["state_dict"], ck["config"]["hidden"], ck["config"]["layers"], ck["n_inputs"]
        mean, std = ck["pipeline"]["mean"], ck["pipeline"]["std"]
    else:
        sd, H, L, n_in = ck["model"], ck["architecture"]["hidden"], ck["architecture"]["layers"], ck["architecture"]["width"]
        mean, std = ck["normalizer"]["mean"], ck["normalizer"]["std"]
    if n_in != 112:
        raise ValueError(f"{path}: {n_in} entradas; solo se soportan 112 crudas")
    net = nn.ModuleDict({"core": nn.GRU(112, H, num_layers=L, batch_first=True), "head": nn.Linear(H, 2)}).to(device)
    net.load_state_dict(sd)
    net.eval()
    cost = 3 * H * (112 + H) + 3 * H * (H + H) * (L - 1) + 2 * H     # MACs por fila
    return dict(net=net, mean=torch.as_tensor(np.asarray(mean, np.float32), device=device),
                std=torch.as_tensor(np.asarray(std, np.float32), device=device), hidden=H, layers=L, macs=cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--n", default="all")
    ap.add_argument("--models", nargs="+", required=True, help="NOMBRE=ruta.pt ...")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-combo", type=int, default=3)
    ap.add_argument("--out", default="ensemble_eval.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = load_model(path, device)
        print(f"{name}: GRU h={models[name]['hidden']} x{models[name]['layers']}  ({path})", flush=True)
    names = list(models)
    combos = [c for r in range(1, min(args.max_combo, len(names)) + 1) for c in itertools.combinations(names, r)]
    accs = {c: GlobalWP() for c in combos}
    pf = pq.ParquetFile(args.valid)
    n = pf.num_row_groups if args.n == "all" else int(args.n)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch):
            ids = list(range(start, min(start + args.batch, n)))
            tbs = [pf.read_row_group(g, use_threads=False) for g in ids]
            X = np.stack([np.column_stack([tb[c].to_numpy() for c in FEATURES]).astype(np.float32) for tb in tbs])
            Y = np.stack([np.column_stack([tb[c].to_numpy() for c in ("t0", "t1")]).astype(np.float32) for tb in tbs])
            M = np.stack([tb["need_prediction"].to_numpy().astype(bool) & tb["is_scored"].to_numpy().astype(bool) for tb in tbs])
            xt = torch.from_numpy(X).to(device)
            preds = {}
            for nm, m in models.items():
                z = (xt - m["mean"]) / m["std"]
                preds[nm] = m["net"]["head"](m["net"]["core"](z)[0]).float().cpu().numpy()
            for c in combos:
                p = np.mean([preds[nm] for nm in c], axis=0)
                for b in range(len(ids)):
                    accs[c].add(Y[b], p[b], M[b])
            done = start + len(ids)
            if done % (args.batch * 10) == 0 or done == n:
                print(f"  {done}/{n} seqs ({(time.time()-t0)/60:.1f} min)", flush=True)
    # µs/fila estimados en 1 CPU con el kernel v2 (int8 traspuesto): ~0.091 µs por 1k MACs + 2 µs fijos
    # (medido en laptop Ryzen 7735U: 3x GRU128x2 = 570k MACs -> 54 µs). El servidor es ~1.15x más lento.
    rows = []
    for c in combos:
        r = accs[c].result()
        macs = sum(models[nm]["macs"] for nm in c)
        us = 2 + macs / 1000 * 0.091
        rows.append({"models": "+".join(c), "wp": r["wp"], "t0": r["t0"], "t1": r["t1"], "us_per_row_est": round(us, 1),
                     "fits_60min": us * 1.15 * 37.46e6 / 60e6 < 55})   # con margen: servidor 1.15x, tope 55 min
    rows.sort(key=lambda r: -r["wp"])
    print(f"\n{'modelos':28s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra")
    for r in rows:
        print(f"{r['models']:28s} {r['wp']:10.4f} {r['t0']:8.4f} {r['t1']:8.4f} {r['us_per_row_est']:8.1f}  {'sí' if r['fits_60min'] else 'NO'}")
    Path(args.out).write_text(json.dumps({"n_seqs": n, "results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
