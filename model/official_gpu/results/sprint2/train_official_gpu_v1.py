r"""Entrenamiento oficial GPU/CPU (v2): GRU/LSTM por batches, métrica GLOBAL, dropout/weight decay, SWA.

Mismo script para la laptop/PC (CPU) y para Kaggle (GPU): detecta `cuda` solo.

Qué hace distinto a los FirstTry:
  * Entrena B secuencias en paralelo (batches) con BPTT truncado por chunks.
  * Loss alineada a la métrica (targets clip ±2, peso |y|), o -WeightedPearson (`--loss wp`).
  * Gradient clipping, lr configurable, checkpoints por época y `--resume`.
  * Evalúa con la métrica oficial (BlockAccumulator de utils.py) en grupos de valid fijos:
    con `--valid-groups 10 --seed 2026` son los MISMOS 10 grupos que EXPERIMENTS.md
    (baseline = 0.568 ahí).
  * Opcional: lags/deltas de `model/feature_select/features.py` (`--top-k`), flags de centinela.

Uso local (desde challenge/):
    python model/official_gpu/train_official_gpu_v1.py --train-groups 500 --epochs 3
Uso Kaggle:
    python train_official_gpu_v1.py --train-parquet /kaggle/input/connectome-train-part1/train_part1.parquet \
        /kaggle/input/connectome-train-part2/train_part2.parquet \
        --valid-parquet /kaggle/input/connectome-valid/valid.parquet --train-groups all --epochs 5 \
        --output-dir /kaggle/working/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE   # starter pack root en local; en Kaggle, la misma carpeta
for p in (ROOT, HERE, HERE.parent / "feature_select"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from utils import (BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS,  # noqa: E402
                   WARMUP, validate_sequence, weighted_pearson)

SENTINEL = -5.1993
SENTINEL_COLS = ("a5", "a6", "a7")


# ----------------------------------------------------------------------------- datos
class ParquetGroups:
    """Varios parquet vistos como una sola lista de row groups (1 row group = 1 secuencia)."""

    def __init__(self, paths: list[str]):
        self.files = [pq.ParquetFile(p) for p in paths]
        self.index = [(f, g) for f in self.files for g in range(f.num_row_groups)]
        self.has_scored = all("is_scored" in f.schema_arrow.names for f in self.files)

    def __len__(self):
        return len(self.index)

    def read(self, i: int):
        f, g = self.index[i]
        cols = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        if self.has_scored:
            cols.append("is_scored")
        t = f.read_row_group(g, columns=cols, use_threads=False)
        seq_ix, need = validate_sequence(t)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        x[~np.isfinite(x)] = 0.0
        scored = t["is_scored"].to_numpy().astype(bool) if self.has_scored else None
        return seq_ix, need, x, y, scored


def select_groups(total: int, count: int, seed: int) -> list[int]:
    """Idéntico a first_try_common.select_groups (mismos grupos que EXPERIMENTS.md)."""
    if not 1 <= count <= total:
        raise ValueError(f"count debe estar entre 1 y {total}")
    return sorted(np.random.default_rng(seed).choice(total, count, replace=False).tolist())


class FeaturePipeline:
    """112 crudas (+ flags de centinela) (+ lags/deltas de features.py). Guarda su normalizador."""

    def __init__(self, top_k: int, sentinel_flags: bool, ranking: Path | None):
        self.top_k = top_k
        self.sentinel_flags = sentinel_flags
        self.specs = []
        if top_k > 0:
            from features import load_rankings, select_temporal_features  # model/feature_select
            self.specs = select_temporal_features(load_rankings(ranking), top_k)
        self.sent_idx = [FEATURE_COLUMNS.index(c) for c in SENTINEL_COLS]
        self.mean = None
        self.std = None

    @property
    def names(self) -> list[str]:
        out = list(FEATURE_COLUMNS)
        if self.sentinel_flags:
            out += [f"{c}_is_sentinel" for c in SENTINEL_COLS]
        out += [s.name for s in self.specs]
        return out

    def raw_to_features(self, x: np.ndarray) -> np.ndarray:
        parts = [x]
        if self.sentinel_flags:
            flags = np.isclose(x[:, self.sent_idx], SENTINEL, atol=2e-4).astype(np.float32)
            x = x.copy()
            x[:, self.sent_idx] = np.where(flags > 0, 0.0, x[:, self.sent_idx])
            parts = [x, flags]
        if self.specs:
            from features import build_sequence
            derived = build_sequence(x, self.specs)[:, len(FEATURE_COLUMNS):]
            parts.append(derived)
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def fit_normalizer(self, groups: ParquetGroups, ids: list[int], rows_per_group: int, seed: int):
        rng = np.random.default_rng(seed)
        sample = []
        for i in ids:
            _, need, x, _, _ = groups.read(i)
            f = self.raw_to_features(x)[need]
            take = min(rows_per_group, len(f))
            sample.append(f[rng.choice(len(f), take, replace=False)])
        s = np.concatenate(sample)
        self.mean = np.nanmean(s, axis=0).astype(np.float32)
        self.std = np.nanstd(s, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        if self.sentinel_flags:   # flags binarias: no estandarizar
            a, b = len(FEATURE_COLUMNS), len(FEATURE_COLUMNS) + len(SENTINEL_COLS)
            self.mean[a:b] = 0.0
            self.std[a:b] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        f = (self.raw_to_features(x) - self.mean) / self.std
        return np.nan_to_num(f, nan=0.0).astype(np.float32)   # sin historia -> 0 estandarizado

    def state(self) -> dict:
        return {"top_k": self.top_k, "sentinel_flags": self.sentinel_flags,
                "specs": [(s.base_feature, s.transform, s.lag) for s in self.specs],
                "mean": self.mean, "std": self.std, "names": self.names}



# ----------------------------------------------------------------------------- métrica GLOBAL (nueva, 2026-09)
class GlobalWP:
    """Weighted Pearson global: junta TODAS las filas scoreadas de todas las secuencias y calcula
    una sola correlación por target (clip ±2, peso |y|), luego promedia t0 y t1.
    Reemplaza a la métrica por secuencia (utils.BlockAccumulator), que quedó obsoleta."""

    def __init__(self):
        self.s = np.zeros((2, 6), dtype=np.float64)   # por target: sum w, wy, wp, wyy, wpp, wyp

    def add(self, targets, predictions, mask):
        y = np.clip(targets[mask].astype(np.float64), -2.0, 2.0)
        p = np.clip(predictions[mask].astype(np.float64), -2.0, 2.0)
        w = np.abs(y)
        for j in range(2):
            wj, yj, pj = w[:, j], y[:, j], p[:, j]
            self.s[j] += [wj.sum(), (wj * yj).sum(), (wj * pj).sum(),
                          (wj * yj * yj).sum(), (wj * pj * pj).sum(), (wj * yj * pj).sum()]

    def result(self):
        out = {}
        for j, name in enumerate(TARGET_COLUMNS):
            W, wy, wp, wyy, wpp, wyp = self.s[j]
            if W < 1e-8:
                out[name] = 0.0
                continue
            my, mp = wy / W, wp / W
            cov = wyp / W - my * mp
            vy, vp = wyy / W - my * my, wpp / W - mp * mp
            out[name] = float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0)) if vy > 1e-16 and vp > 1e-16 else 0.0
        out["weighted_pearson"] = (out["t0"] + out["t1"]) / 2
        return out


# ----------------------------------------------------------------------------- modelo
class RNNModel(nn.Module):
    def __init__(self, n_in: int, hidden: int, layers: int, cell: str, dropout: float = 0.0):
        super().__init__()
        core = nn.GRU if cell == "gru" else nn.LSTM
        # dropout solo entre capas (PyTorch lo ignora con layers=1); no cambia la inferencia
        self.core = core(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, state=None):
        out, state = self.core(x, state)
        return self.head(out), state


def detach(state):
    if state is None:
        return None
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()


def loss_wmse(pred, target, mask, gamma=1.0):
    y = target.clamp(-2.0, 2.0)
    w = y.abs() * mask.unsqueeze(-1)
    if gamma != 1.0:
        w = w ** gamma
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-8)


def loss_gwp(pred, target, mask):
    """1 - WeightedPearson GLOBAL sobre todo el batch x tiempo (como la métrica), promedio de t0/t1."""
    y = target.clamp(-2.0, 2.0)
    w = (y.abs() * mask.unsqueeze(-1)).reshape(-1, 2)
    y = y.reshape(-1, 2)
    p = pred.reshape(-1, 2)
    ws = w.sum(0).clamp_min(1e-8)
    my, mp = (w * y).sum(0) / ws, (w * p).sum(0) / ws
    yc, pc = y - my, p - mp
    cov = (w * yc * pc).sum(0) / ws
    sy = ((w * yc * yc).sum(0) / ws).clamp_min(1e-8).sqrt()
    sp = ((w * pc * pc).sum(0) / ws).clamp_min(1e-8).sqrt()
    return (1.0 - cov / (sy * sp)).mean()


def load_teacher(path, device):
    """Carga una GRU (ambos formatos) como teacher congelado. Devuelve (net, mean, std, clip)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        sd, H, L = ck["state_dict"], ck["config"]["hidden"], ck["config"]["layers"]
        mean, std, clip = ck["pipeline"]["mean"], ck["pipeline"]["std"], float("inf")
    else:
        sd, H, L = ck["model"], ck["architecture"]["hidden"], ck["architecture"]["layers"]
        mean, std, clip = ck["normalizer"]["mean"], ck["normalizer"]["std"], 20.0
    net = nn.ModuleDict({"core": nn.GRU(112, H, num_layers=L, batch_first=True), "head": nn.Linear(H, 2)}).to(device)
    net.load_state_dict(sd)
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net, torch.as_tensor(np.asarray(mean, np.float32), device=device), torch.as_tensor(np.asarray(std, np.float32), device=device), clip


def loss_wp(pred, target, mask):
    """-WeightedPearson por secuencia y target dentro del chunk (promedio)."""
    y = target.clamp(-2.0, 2.0)
    w = y.abs() * mask.unsqueeze(-1)                     # (B, T, 2)
    ws = w.sum(dim=1).clamp_min(1e-8)                     # (B, 2)
    my = (w * y).sum(1) / ws
    mp = (w * pred).sum(1) / ws
    yc, pc = y - my.unsqueeze(1), pred - mp.unsqueeze(1)
    cov = (w * yc * pc).sum(1) / ws
    sy = ((w * yc * yc).sum(1) / ws).clamp_min(1e-8).sqrt()
    sp = ((w * pc * pc).sum(1) / ws).clamp_min(1e-8).sqrt()
    rho = cov / (sy * sp)
    valid = (ws > 1e-6) & (sy > 1e-4)
    return -(rho * valid).sum() / valid.sum().clamp_min(1)


# ----------------------------------------------------------------------------- evaluación
@torch.no_grad()
def evaluate(model, pipe, groups: ParquetGroups, ids: list[int], device, batch: int):
    model.eval()
    acc, gacc = BlockAccumulator(), GlobalWP()
    rows = []
    for start in range(0, len(ids), batch):
        chunk_ids = ids[start:start + batch]
        data = [groups.read(i) for i in chunk_ids]
        x = torch.from_numpy(np.stack([pipe.transform(d[2]) for d in data])).to(device)
        pred = model(x)[0].float().cpu().numpy()
        for (seq_ix, need, _, y, scored), p in zip(data, pred):
            mask = need if scored is None else (need & scored)
            acc.add(y, p, mask)
            gacc.add(y, p, mask)
            yy, pp = y[mask], p[mask]
            if all(np.any(yy[:, j] > 0) and np.any(yy[:, j] < 0) for j in range(2)):
                w0, w1 = weighted_pearson(yy[:, 0], pp[:, 0]), weighted_pearson(yy[:, 1], pp[:, 1])
                rows.append({"seq_ix": seq_ix, "wp_t0": w0, "wp_t1": w1, "wp": (w0 + w1) / 2})
            else:
                rows.append({"seq_ix": seq_ix, "wp_t0": None, "wp_t1": None, "wp": None})
    model.train()
    res = acc.result()
    g = gacc.result()
    res.update(global_wp=g["weighted_pearson"], global_t0=g["t0"], global_t1=g["t1"], per_sequence_wp=res["weighted_pearson"])
    res["weighted_pearson"] = g["weighted_pearson"]   # la métrica GLOBAL es la oficial desde 2026-09
    res["t0"], res["t1"] = g["t0"], g["t1"]
    return res, rows


@torch.no_grad()
def _save_valid_preds(model, pipe, groups, ids, device, batch, path):
    model.eval()
    seqs, preds = [], []
    for start in range(0, len(ids), batch):
        data = [groups.read(i) for i in ids[start:start + batch]]
        x = torch.from_numpy(np.stack([pipe.transform(d[2]) for d in data])).to(device)
        p = model(x)[0].float().cpu().numpy().astype(np.float16)
        seqs += [d[0] for d in data]
        preds.append(p)
    model.train()
    np.savez_compressed(path, seq_ix=np.array(seqs, np.int64), preds=np.concatenate(preds))


# ----------------------------------------------------------------------------- entrenamiento
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-parquet", nargs="+", default=[str(ROOT / "datasets" / "train.parquet")])
    ap.add_argument("--valid-parquet", nargs="+", default=[str(ROOT / "datasets" / "valid.parquet")])
    ap.add_argument("--train-groups", default="500", help="N o 'all'")
    ap.add_argument("--valid-groups", default="10", help="N o 'all' (las 1.873)")
    ap.add_argument("--seed", type=int, default=2026, help="selección de grupos (valid usa seed+1, como FirstTry)")
    ap.add_argument("--model-seed", type=int, default=2026)
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None, help="secuencias por batch (default: 32 GPU / 8 CPU)")
    ap.add_argument("--chunk", type=int, default=None, help="pasos por chunk de BPTT (default: 512 GPU / 256 CPU)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", type=float, default=1.0, help="factor multiplicativo por época")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient clipping (0 = off)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout entre capas de la GRU (regularización)")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (regularización)")
    ap.add_argument("--swa-start", type=int, default=0, help="promediar pesos desde esta época (0 = off); evalúa y guarda swa_model.pt")
    ap.add_argument("--save-epochs", action="store_true", help="guardar epoch_N.pt de cada época (para promediar offline)")
    ap.add_argument("--loss", choices=["wmse", "wp"], default="wmse")
    ap.add_argument("--gwp-weight", type=float, default=0.0, help="suma gwp_weight * (1 - WP global del batch) a la loss")
    ap.add_argument("--weight-power", type=float, default=1.0, help="pesos |y|^gamma en la WMSE (1 = métrica)")
    ap.add_argument("--lr-schedule", choices=["step", "cosine"], default="step", help="step: lr*decay^epoch; cosine: de lr a lr-min")
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--init-from", default=None, help="checkpoint (best_model.pt / swa_model.pt) para inicializar pesos (fine-tuning)")
    ap.add_argument("--distill-teachers", nargs="*", default=[], help="rutas .pt de teachers; sus predicciones promediadas son target blando")
    ap.add_argument("--distill-weight", type=float, default=1.0, help="peso de la loss blanda (WMSE contra el teacher)")
    ap.add_argument("--save-valid-preds", action="store_true", help="guardar predicciones del mejor modelo en valid (npz float16)")
    ap.add_argument("--top-k", type=int, default=0, help="lags/deltas por target de features.py (0,3,5,10,20)")
    ap.add_argument("--ranking", default=str((HERE.parents[1] if len(HERE.parents) > 1 else HERE) / "explore" / "feature_select" / "results" / "top_20_by_target.csv"))
    ap.add_argument("--sentinel-flags", action="store_true", help="a5/a6/a7: -5.1993 -> 0 + columna indicadora")
    ap.add_argument("--norm-rows", type=int, default=1000, help="filas por grupo (máx 200 grupos) para el normalizador")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=0, help="además evaluar cada N batches (0 = solo por época)")
    ap.add_argument("--workers", type=int, default=2, help="hilos de lectura de parquet")
    ap.add_argument("--threads", type=int, default=0, help="hilos de torch en CPU (0 = auto)")
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="carpeta de un run para retomar (usa checkpoint.pt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    batch = args.batch or (32 if on_gpu else 8)
    chunk = args.chunk or (512 if on_gpu else 256)
    if not on_gpu:
        torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) - 2))
    print(f"device={device}  batch={batch}  chunk={chunk}  torch={torch.__version__}", flush=True)

    train = ParquetGroups(args.train_parquet)
    valid = ParquetGroups(args.valid_parquet)
    if not valid.has_scored:
        raise ValueError("valid necesita is_scored para la métrica oficial")
    train_ids = list(range(len(train))) if args.train_groups == "all" else select_groups(len(train), int(args.train_groups), args.seed)
    valid_ids = list(range(len(valid))) if str(args.valid_groups) == "all" else select_groups(len(valid), int(args.valid_groups), args.seed + 1)

    # ---- run dir / resume
    if args.resume:
        run_dir = Path(args.resume)
        ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = ck["config"]
        pipe = FeaturePipeline(cfg["top_k"], cfg["sentinel_flags"], Path(cfg["ranking"]))
        pipe.mean, pipe.std = ck["pipeline"]["mean"], ck["pipeline"]["std"]
        print(f"Retomando {run_dir} desde época {ck['epoch']} batch {ck['batch']}", flush=True)
    else:
        name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = {**vars(args), "device": str(device), "batch": batch, "chunk": chunk,
               "train_ids": train_ids, "valid_ids": valid_ids, "n_train": len(train_ids)}
        pipe = FeaturePipeline(args.top_k, args.sentinel_flags, Path(args.ranking))
        t0 = time.time()
        if len(train_ids) > 200:
            norm_ids = [train_ids[i] for i in select_groups(len(train_ids), 200, args.seed + 7)]
        else:
            norm_ids = train_ids
        pipe.fit_normalizer(train, norm_ids, args.norm_rows, args.seed + 2)
        print(f"Normalizador: {len(norm_ids)} grupos, {len(pipe.names)} entradas ({time.time()-t0:.0f}s)", flush=True)
        ck = None
        (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    torch.manual_seed(cfg["model_seed"])
    model = RNNModel(len(pipe.names), cfg["hidden"], cfg["layers"], cfg["cell"], cfg.get("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = loss_wp if cfg["loss"] == "wp" else loss_wmse
    if cfg.get("init_from") and not ck:
        init = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        model.load_state_dict(init["state_dict"] if "state_dict" in init else init["model"])
        if "pipeline" in init:   # usar el normalizador del modelo inicial (los pesos lo asumen)
            pipe.mean, pipe.std = init["pipeline"]["mean"], init["pipeline"]["std"]
        print(f"Pesos inicializados desde {cfg['init_from']}", flush=True)
    teachers = [load_teacher(t, device) for t in cfg.get("distill_teachers", []) or []]
    if teachers:
        print(f"Destilación: {len(teachers)} teachers, peso {cfg['distill_weight']}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['cell'].upper()} h={cfg['hidden']} x{cfg['layers']}  entradas={len(pipe.names)}  params={n_params:,}", flush=True)

    start_epoch, start_batch, history, best = 0, 0, [], -1.0
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, start_batch, history, best = ck["epoch"], ck["batch"], ck["history"], ck["best"]
        torch.set_rng_state(ck["rng"]["torch"])

    def save_checkpoint(epoch, b_idx):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "batch": b_idx,
                    "history": history, "best": best, "config": cfg, "pipeline": pipe.state(),
                    "rng": {"torch": torch.get_rng_state()}}, run_dir / "checkpoint.pt")

    def save_model(path, state_dict=None, extra=None):
        sd = state_dict if state_dict is not None else model.state_dict()
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in sd.items()},
                    "config": {k: cfg[k] for k in ("cell", "hidden", "layers")},
                    "pipeline": pipe.state(), "n_inputs": len(pipe.names), **(extra or {})}, path)

    swa_sum, swa_n = None, 0

    def run_eval(tag, epoch, b_idx):
        nonlocal best
        t = time.time()
        res, rows = evaluate(model, pipe, valid, valid_ids, device, cfg["eval_batch"])
        wp = res["weighted_pearson"]
        print(f"  [{tag}] valid GLOBAL WP={wp:.4f} (t0={res['t0']:.4f}, t1={res['t1']:.4f})   "
              f"[por secuencia: {res['per_sequence_wp']:.4f}]  ({time.time()-t:.0f}s)", flush=True)
        history.append({"epoch": epoch, "batch": b_idx, "tag": tag, "wp": wp, "t0": res["t0"], "t1": res["t1"], "time": time.time()})
        if wp > best:
            best = wp
            save_model(run_dir / "best_model.pt")
            if cfg.get("save_valid_preds"):
                _save_valid_preds(model, pipe, valid, valid_ids, device, cfg["eval_batch"], run_dir / "valid_preds_best.npz")
            with (run_dir / "best_by_rowgroup.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        (run_dir / "metrics.json").write_text(json.dumps({"best_valid_wp": best, "last": res, "history": history,
                                                          "config": cfg, "params": n_params}, indent=2, default=str), encoding="utf-8")
        return wp

    # ---- loop
    pool = ThreadPoolExecutor(max_workers=cfg["workers"])
    n_batches = math.ceil(len(train_ids) / batch)
    warm = torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32)
    warm[WARMUP:] = 1.0
    t_start = time.time()
    done_batches = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        order = np.random.default_rng(cfg["seed"] + 100 + epoch).permutation(len(train_ids))
        batches = [[train_ids[i] for i in order[s:s + batch]] for s in range(0, len(order), batch)]
        if cfg.get("lr_schedule", "step") == "cosine":
            lr_now = cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * epoch / max(1, cfg["epochs"] - 1)))
        else:
            lr_now = cfg["lr"] * (cfg["lr_decay"] ** epoch)
        for g in opt.param_groups:
            g["lr"] = lr_now
        print(f"época {epoch+1}: lr={lr_now:.2e}", flush=True)

        def load(bi):
            return [train.read(i) for i in batches[bi]]

        first = start_batch if epoch == start_epoch else 0
        futures = {bi: pool.submit(load, bi) for bi in range(first, min(first + cfg["workers"], n_batches))}
        loss_acc, loss_n = 0.0, 0
        for bi in range(first, n_batches):
            data = futures.pop(bi).result()
            nxt = bi + cfg["workers"]
            if nxt < n_batches:
                futures[nxt] = pool.submit(load, nxt)
            x = torch.from_numpy(np.stack([pipe.transform(d[2]) for d in data])).to(device, non_blocking=True)
            y = torch.from_numpy(np.stack([d[3] for d in data])).to(device, non_blocking=True)
            m = warm.unsqueeze(0).expand(len(data), -1).to(device)
            state = None
            if teachers:
                xraw = torch.from_numpy(np.stack([d[2] for d in data])).to(device, non_blocking=True)
                t_states = [None] * len(teachers)
            for s in range(0, SEQUENCE_LENGTH, chunk):
                e = min(s + chunk, SEQUENCE_LENGTH)
                pred, state = model(x[:, s:e], state)
                if teachers:
                    with torch.no_grad():
                        soft = 0.0
                        for ti, (tnet, tmean, tstd, tclip) in enumerate(teachers):
                            zt = ((xraw[:, s:e] - tmean) / tstd).clamp(-tclip, tclip)
                            out_t, t_states[ti] = tnet["core"](zt, t_states[ti])
                            soft = soft + tnet["head"](out_t)
                        soft = soft / len(teachers)
                if e > WARMUP:
                    if cfg["loss"] == "wmse":
                        loss = loss_wmse(pred, y[:, s:e], m[:, s:e], cfg.get("weight_power", 1.0))
                    else:
                        loss = loss_fn(pred, y[:, s:e], m[:, s:e])
                    if cfg.get("gwp_weight", 0.0) > 0:
                        loss = loss + cfg["gwp_weight"] * loss_gwp(pred, y[:, s:e], m[:, s:e])
                    if teachers:
                        wsoft = y[:, s:e].clamp(-2.0, 2.0).abs() * m[:, s:e].unsqueeze(-1)   # mismos pesos que la métrica
                        loss = loss + cfg["distill_weight"] * ((wsoft * (pred - soft) ** 2).sum() / wsoft.sum().clamp_min(1e-8))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg["clip"] > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                    opt.step()
                    loss_acc += float(loss.detach())
                    loss_n += 1
                state = detach(state)
            done_batches += 1
            if bi % max(1, n_batches // 20) == 0 or bi == n_batches - 1:
                el = time.time() - t_start
                total = (cfg["epochs"] - start_epoch) * n_batches - (start_batch if epoch == start_epoch else 0)
                eta = el / max(1, done_batches) * (total - done_batches)
                print(f"época {epoch+1}/{cfg['epochs']} batch {bi+1}/{n_batches}  loss={loss_acc/max(1,loss_n):.4f}  "
                      f"transcurrido {el/60:.1f} min  restante ~{eta/60:.1f} min", flush=True)
                loss_acc, loss_n = 0.0, 0
            if cfg["eval_every"] and (bi + 1) % cfg["eval_every"] == 0 and bi + 1 < n_batches:
                run_eval(f"e{epoch+1}b{bi+1}", epoch, bi + 1)
                save_checkpoint(epoch, bi + 1)
        run_eval(f"época {epoch+1}", epoch + 1, 0)
        save_checkpoint(epoch + 1, 0)
        if cfg.get("save_epochs"):
            save_model(run_dir / f"epoch_{epoch+1:02d}.pt", extra={"epoch": epoch + 1})
        if cfg.get("swa_start", 0) and epoch + 1 >= cfg["swa_start"]:
            cur = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
            if swa_sum is None:
                swa_sum = cur
            else:
                for k in swa_sum:
                    swa_sum[k] += cur[k]
            swa_n += 1
        start_batch = 0
    pool.shutdown()
    save_model(run_dir / "last_model.pt")
    if swa_sum is not None and swa_n > 1:
        swa_sd = {k: (v / swa_n).to(device) for k, v in swa_sum.items()}
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(swa_sd)
        res, _ = evaluate(model, pipe, valid, valid_ids, device, cfg["eval_batch"])
        print(f"  [SWA épocas {cfg['swa_start']}-{cfg['epochs']} ({swa_n} modelos)] valid GLOBAL WP={res['weighted_pearson']:.4f} "
              f"(t0={res['t0']:.4f}, t1={res['t1']:.4f})", flush=True)
        save_model(run_dir / "swa_model.pt", extra={"swa": {"start": cfg["swa_start"], "n": swa_n, "valid": res}})
        history.append({"epoch": cfg["epochs"], "batch": 0, "tag": "swa", "wp": res["weighted_pearson"], "t0": res["t0"], "t1": res["t1"], "time": time.time()})
        (run_dir / "metrics.json").write_text(json.dumps({"best_valid_wp": best, "swa_valid_wp": res["weighted_pearson"], "history": history,
                                                          "config": cfg, "params": n_params}, indent=2, default=str), encoding="utf-8")
        model.load_state_dict(backup)
    print(f"Mejor valid WP: {best:.4f}   Resultados en {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
