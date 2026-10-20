r"""v4 (2026-09-18): --distill-cache precalcula los soft targets de los teachers una vez (fp16) en vez de correrlos en cada chunk.
Entrenamiento oficial GPU/CPU (v2): GRU/LSTM por batches, métrica GLOBAL, dropout/weight decay, SWA.

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

_IX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
BIDP = [_IX[f"i0_p{j}"] for j in range(11)]        # 11 niveles de compra (precio)
ASKP = [_IX[f"i0_p{j}"] for j in range(11, 22)]    # 11 niveles de venta
BIDV = [_IX[f"i0_v{j}"] for j in range(11)]
ASKV = [_IX[f"i0_v{j}"] for j in range(11, 22)]
BIDP1 = [_IX[f"i1_p{j}"] for j in range(11)]
ASKP1 = [_IX[f"i1_p{j}"] for j in range(11, 22)]
BIDV1 = [_IX[f"i1_v{j}"] for j in range(11)]
ASKV1 = [_IX[f"i1_v{j}"] for j in range(11, 22)]
CROSS_NAMES = ["deseq_top", "deseq_3", "deseq_total", "deseq_vol_frac", "desvio_bid_p", "desvio_ask_p",
               "desvio_bid_v", "desvio_ask_v", "prof_bid_rel", "prof_ask_rel", "spread_rel", "mid_dif_rel",
               "max_bid_v", "max_ask_v", "conc_bid_v", "conc_ask_v", "deseq_i1_top", "deseq_dif",
               "spread_x_deseq", "deseq_top_x_3"]


def cross_features(x: np.ndarray) -> np.ndarray:
    """(T, 112) -> (T, 20): combinaciones NO LINEALES dentro de la misma fila.

    Solo cosas que la primera capa de la red no puede construir sola: cocientes (desequilibrios),
    desvíos y máximos entre niveles, y productos. Denominadores con +1.0 para que no exploten
    (las entradas ya vienen normalizadas, valores del orden de 1). Causales: solo usan la fila actual.
    """
    bp, ap_, bv, av = x[:, BIDP], x[:, ASKP], x[:, BIDV], x[:, ASKV]
    bv1, av1 = x[:, BIDV1], x[:, ASKV1]
    spread = ap_[:, 0] - bp[:, 0]
    mid = (ap_[:, 0] + bp[:, 0]) * 0.5
    mid1 = (x[:, ASKP1][:, 0] + x[:, BIDP1][:, 0]) * 0.5
    d_top = (bv[:, 0] - av[:, 0]) / (np.abs(bv[:, 0]) + np.abs(av[:, 0]) + 1.0)
    d_3 = (bv[:, :3].sum(1) - av[:, :3].sum(1)) / (np.abs(bv[:, :3]).sum(1) + np.abs(av[:, :3]).sum(1) + 1.0)
    d_all = (bv.sum(1) - av.sum(1)) / (np.abs(bv).sum(1) + np.abs(av).sum(1) + 1.0)
    d_i1 = (bv1[:, 0] - av1[:, 0]) / (np.abs(bv1[:, 0]) + np.abs(av1[:, 0]) + 1.0)
    out = [
        d_top, d_3, d_all,
        np.abs(bv).sum(1) / (np.abs(bv).sum(1) + np.abs(av).sum(1) + 1.0),
        bp.std(1), ap_.std(1), bv.std(1), av.std(1),
        (bp[:, 0] - bp[:, 10]) / (np.abs(spread) + 1.0),
        (ap_[:, 10] - ap_[:, 0]) / (np.abs(spread) + 1.0),
        spread / (np.abs(mid) + 1.0),
        (mid - mid1) / (np.abs(spread) + 1.0),
        bv.max(1), av.max(1),
        bv.max(1) / (np.abs(bv).sum(1) + 1.0),        # concentración: cuánto del volumen está en un solo nivel
        av.max(1) / (np.abs(av).sum(1) + 1.0),
        d_i1, d_top - d_i1,
        spread * d_top, d_top * d_3,
    ]
    return np.clip(np.stack(out, axis=1), -10.0, 10.0).astype(np.float32)


try:
    from scipy.signal import lfilter as _lfilter     # importar UNA vez: adentro de la función cuesta 2,5 s la primera
except ImportError:
    _lfilter = None

EMA_COLS = ["i0_p0", "i0_p11", "i0_v0", "i0_v11", "i1_p0", "i1_p11", "i1_v0", "i1_v11", "a0", "a1", "a2", "a4"]
EMA_IX = [_IX[c] for c in EMA_COLS]
EMA_SPANS = (100, 1000, 10000)      # filas de memoria de cada promedio
EMA_VOL_COLS = ["i0_p0", "i0_p11", "i0_v0", "i1_p0"]
EMA_VOL_IX = [_IX[c] for c in EMA_VOL_COLS]
EMA_NAMES = ([f"{c}_dif_ema{s}" for s in EMA_SPANS for c in EMA_COLS] +
             [f"{c}_vol_lenta" for c in EMA_VOL_COLS])


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Promedio móvil exponencial causal por columna (filtro IIR de primer orden). ema[0] = x[0]."""
    a = 2.0 / (span + 1.0)
    if _lfilter is not None:
        zi = (x[0] * (1 - a))[None, :]                      # estado inicial: arranca en x[0]
        return _lfilter([a], [1.0, -(1 - a)], x, axis=0, zi=zi)[0]
    else:                                                    # sin scipy: bucle (lento, solo para pruebas)
        out = np.empty_like(x)
        m = x[0].copy()
        for t in range(len(x)):
            m += a * (x[t] - m)
            out[t] = m
        return out


def ema_features(x: np.ndarray) -> np.ndarray:
    """(T, 112) -> (T, len(EMA_NAMES)): desvío respecto de promedios móviles largos + volatilidad lenta."""
    sub = x[:, EMA_IX].astype(np.float32)
    parts = []
    e100 = None
    for span in EMA_SPANS:
        e = _ema(sub, span).astype(np.float32)
        if span == EMA_SPANS[0]:
            e100 = e
        parts.append(sub - e)
    vol = _ema(np.abs(sub[:, [EMA_COLS.index(c) for c in EMA_VOL_COLS]] -
                      e100[:, [EMA_COLS.index(c) for c in EMA_VOL_COLS]]), 1000).astype(np.float32)
    parts.append(vol)
    return np.clip(np.concatenate(parts, axis=1), -10.0, 10.0).astype(np.float32)


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

    def __init__(self, top_k: int, sentinel_flags: bool, ranking: Path | None, cross: bool = False, ema: bool = False):
        self.top_k = top_k
        self.sentinel_flags = sentinel_flags
        self.cross = cross
        self.ema = ema
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
        if self.cross:
            out += CROSS_NAMES
        if self.ema:
            out += EMA_NAMES
        out += [s.name for s in self.specs]
        return out

    def raw_to_features(self, x: np.ndarray) -> np.ndarray:
        parts = [x]
        if self.sentinel_flags:
            flags = np.isclose(x[:, self.sent_idx], SENTINEL, atol=2e-4).astype(np.float32)
            x = x.copy()
            x[:, self.sent_idx] = np.where(flags > 0, 0.0, x[:, self.sent_idx])
            parts = [x, flags]
        if self.cross:
            parts.append(cross_features(x))
        if self.ema:
            parts.append(ema_features(x))
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
        return {"top_k": self.top_k, "sentinel_flags": self.sentinel_flags, "cross": self.cross, "ema": self.ema,
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
    """Red rápida (GRU/LSTM por fila) + opcionalmente una red LENTA que corre 1 de cada `slow_every` filas.

    La lenta recibe el promedio de los estados de la rápida durante la ventana y devuelve un vector de
    contexto que se concatena a la salida. Las filas de una ventana usan el contexto de la ventana ANTERIOR
    (causal, y calculable online en el kernel de inferencia).
    """

    def __init__(self, n_in: int, hidden: int, layers: int, cell: str, dropout: float = 0.0,
                 slow_hidden: int = 0, slow_every: int = 32, n_aux: int = 0):
        super().__init__()
        core = nn.GRU if cell == "gru" else nn.LSTM
        # dropout solo entre capas (PyTorch lo ignora con layers=1); no cambia la inferencia
        self.core = core(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.slow_hidden, self.slow_every = slow_hidden, slow_every
        self.n_aux = n_aux
        if slow_hidden:
            self.slow = nn.GRU(hidden, slow_hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden + slow_hidden, 2 + n_aux)   # las salidas extra solo se usan al entrenar

    def forward(self, x, state=None, with_aux=False):
        core_state, slow_state, s_prev = state if state is not None else (None, None, None)
        out, core_state = self.core(x, core_state)
        if not self.slow_hidden:
            y = self.head(out)
            return (y if with_aux else y[..., :2]), (core_state, None, None)
        B, T, H = out.shape
        k = self.slow_every
        assert T % k == 0, f"el chunk ({T}) debe ser múltiplo de --slow-every ({k})"
        if s_prev is None:
            s_prev = out.new_zeros(B, self.slow_hidden)
        means = out.reshape(B, T // k, k, H).mean(2)                  # (B, W, H): resumen de cada ventana
        s_seq, slow_state = self.slow(means, slow_state)              # (B, W, S)
        s_use = torch.cat([s_prev.unsqueeze(1), s_seq[:, :-1]], 1)    # cada ventana usa el estado de la anterior
        s_broad = s_use.unsqueeze(2).expand(B, T // k, k, self.slow_hidden).reshape(B, T, self.slow_hidden)
        y = self.head(torch.cat([out, s_broad], -1))
        return (y if with_aux else y[..., :2]), (core_state, slow_state, s_seq[:, -1])


class TCNModel(nn.Module):
    """Convolucional causal dilatada. Mira una ventana larga de pasos con filtros aprendidos.

    Cada bloque es residual: h = h + gelu(norm(conv(h))). Las dilataciones crecen (1, 4, 16, 64, 256) para
    cubrir cientos de filas con pocas capas. El estado que se pasa entre trozos son las últimas
    (kernel-1)*dilatación entradas de cada bloque, que es lo mismo que el buffer circular de la inferencia.
    """

    def __init__(self, n_in: int, channels: int = 128, layers: int = 5, kernel: int = 3,
                 dropout: float = 0.0, n_aux: int = 0, dil_base: int = 4):
        super().__init__()
        self.kernel, self.n_aux = kernel, n_aux
        self.dils = [dil_base ** i for i in range(layers)]
        self.proj = nn.Linear(n_in, channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel, dilation=d) for d in self.dils])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in self.dils])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 2 + n_aux)
        self.receptive = 1 + (kernel - 1) * sum(self.dils)

    def forward(self, x, state=None, with_aux=False):
        h = self.proj(x)
        nuevo = []
        for i, (conv, d) in enumerate(zip(self.convs, self.dils)):
            pad = (self.kernel - 1) * d
            prev = state[i] if state is not None else h.new_zeros(h.shape[0], pad, h.shape[2])
            hin = torch.cat([prev, h], 1)
            nuevo.append(hin[:, hin.shape[1] - pad:])
            z = conv(hin.transpose(1, 2)).transpose(1, 2)
            h = h + self.drop(torch.nn.functional.gelu(self.norms[i](z)))
        y = self.head(h)
        return (y if with_aux else y[..., :2]), nuevo


def detach(state):
    if state is None:
        return None
    if isinstance(state, list):                                       # convolucional: un buffer por capa
        return [t.detach() for t in state]
    if isinstance(state, tuple) and len(state) == 3:                  # (estado rápido, estado lento, contexto)
        return tuple(detach(s) for s in state)
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()


AUX_NAMES = ["t0_suavizado20", "t1_suavizado20", "t0_suavizado100", "t1_suavizado100", "i1_movimiento10"]


def aux_targets(y, xraw, bidp1_0, askp1_0):
    """Targets auxiliares a partir de los targets y las features crudas de la secuencia COMPLETA.

    y: (B, T, 2) targets. xraw: (B, T, 112). Devuelve (B, T, 5), ya estandarizados.
    Mirar al futuro acá es legítimo: son targets, no entradas.
    """
    B, T, _ = y.shape
    outs = []
    for k in (20, 100):
        c = torch.cumsum(y.double(), 1)
        fut = (c[:, k:] - c[:, :-k]) / k                       # fut[t] = promedio de y[t+1 .. t+k]
        fut = torch.cat([fut, fut[:, -1:].expand(B, k, 2)], 1).float()
        outs.append(fut)
    mid1 = (xraw[:, :, askp1_0] + xraw[:, :, bidp1_0]) * 0.5    # precio medio del segundo instrumento
    mov = torch.cat([mid1[:, 10:] - mid1[:, :-10], torch.zeros(B, 10, device=y.device)], 1)
    outs.append(mov.unsqueeze(-1))
    a = torch.cat(outs, -1)
    return (a - a.mean((0, 1), keepdim=True)) / a.std((0, 1), keepdim=True).clamp_min(1e-6)


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


def loss_seqwp(pred, target, mask, stats, decay=0.98):
    """1 - WeightedPearson acumulado a lo largo de la secuencia (como la métrica), por target.

    stats: tensor (6, 2) con los estadísticos de los trozos anteriores (sin gradiente), o None al empezar
    una secuencia nueva. Devuelve (loss, stats_nuevos). decay<1 = promedio exponencial: el trozo actual
    pesa lo suficiente como para que el gradiente no se diluya al final de la secuencia.
    """
    y = target.clamp(-2.0, 2.0)
    w = (y.abs() * mask.unsqueeze(-1)).reshape(-1, 2)      # mismos pesos que la métrica: |y|
    y = y.reshape(-1, 2)
    p = pred.reshape(-1, 2)
    cur = torch.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0),
                       (w * y * y).sum(0), (w * p * p).sum(0), (w * y * p).sum(0)])
    tot = cur if stats is None else cur + decay * stats
    W = tot[0].clamp_min(1e-8)
    my, mp = tot[1] / W, tot[2] / W
    cov = tot[5] / W - my * mp
    vy = (tot[3] / W - my * my).clamp_min(1e-10)
    vp = (tot[4] / W - mp * mp).clamp_min(1e-10)
    rho = cov / (vy * vp).sqrt()
    return (1.0 - rho).mean(), tot.detach()


def load_teacher(path, device):
    """Carga una GRU/LSTM de N capas (ambos formatos) como teacher congelado. Devuelve (net, mean, std, clip)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        sd, H, L, cell = ck["state_dict"], ck["config"]["hidden"], ck["config"]["layers"], ck["config"].get("cell", "gru")
        mean, std, clip = ck["pipeline"]["mean"], ck["pipeline"]["std"], float("inf")
    else:
        sd, H, L, cell = ck["model"], ck["architecture"]["hidden"], ck["architecture"]["layers"], "gru"
        mean, std, clip = ck["normalizer"]["mean"], ck["normalizer"]["std"], 20.0
    core = (nn.GRU if cell == "gru" else nn.LSTM)(112, H, num_layers=L, batch_first=True)
    net = nn.ModuleDict({"core": core, "head": nn.Linear(H, 2)}).to(device)
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


@torch.no_grad()
def precompute_soft(teachers, teacher_paths, groups, ids, batch, workers, device, cache_path, chunk=2000):
    """Predicción promedio de los teachers para cada secuencia de ids -> dict id -> (SEQUENCE_LENGTH, 2) float16.
    Si cache_path existe y corresponde a los mismos teachers e ids, se carga; si no, se calcula y se guarda."""
    key = "|".join(Path(t).name for t in teacher_paths)
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            z = np.load(cp, allow_pickle=False)
            if str(z["key"]) == key and z["ids"].tolist() == list(ids):
                soft, zids = z["soft"], z["ids"]   # leer UNA vez: NpzFile relee el archivo entero en cada acceso z["soft"]
                print(f"Soft targets cargados de {cp} ({soft.nbytes/1e6:.0f} MB)", flush=True)
                return {int(i): soft[k] for k, i in enumerate(zids)}
            print(f"Cache {cp} no coincide (teachers o ids distintos): se recalcula", flush=True)
    t0 = time.time()
    soft = np.zeros((len(ids), SEQUENCE_LENGTH, 2), np.float16)
    pool = ThreadPoolExecutor(max_workers=workers)
    chunks = [ids[s:s + batch] for s in range(0, len(ids), batch)]
    futures = {bi: pool.submit(lambda b: [groups.read(i) for i in b], chunks[bi]) for bi in range(min(workers, len(chunks)))}
    pos = 0
    for bi in range(len(chunks)):
        data = futures.pop(bi).result()
        if bi + workers < len(chunks):
            futures[bi + workers] = pool.submit(lambda b: [groups.read(i) for i in b], chunks[bi + workers])
        xraw = torch.from_numpy(np.stack([d[2] for d in data])).to(device)
        acc = torch.zeros(len(data), SEQUENCE_LENGTH, 2, device=device)
        for tnet, tmean, tstd, tclip in teachers:
            zt = ((xraw - tmean) / tstd).clamp(-tclip, tclip)
            state = None
            for s0 in range(0, SEQUENCE_LENGTH, chunk):   # por trozos con estado: 20.000 pasos de una vez pide >7 GB en GPU
                out_t, state = tnet["core"](zt[:, s0:s0 + chunk], state)
                acc[:, s0:s0 + chunk] += tnet["head"](out_t)
        soft[pos:pos + len(data)] = (acc / len(teachers)).half().cpu().numpy()
        pos += len(data)
        if bi % max(1, len(chunks) // 10) == 0 or bi == len(chunks) - 1:
            print(f"  soft targets {pos}/{len(ids)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    pool.shutdown()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, soft=soft, ids=np.asarray(ids, np.int64), key=np.array(key))
        print(f"Soft targets guardados en {cache_path} ({soft.nbytes/1e6:.0f} MB, {(time.time()-t0)/60:.1f} min)", flush=True)
    return {int(i): soft[k] for k, i in enumerate(ids)}


# ----------------------------------------------------------------------------- entrenamiento
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-parquet", nargs="+", default=None, help="default: datasets/train.parquet local; con --resume, la ruta del checkpoint")
    ap.add_argument("--valid-parquet", nargs="+", default=None, help="default: datasets/valid.parquet local; con --resume, la ruta del checkpoint")
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
    ap.add_argument("--mask-coef", default=None, help="JSON con el modelo que estima la probabilidad de que una fila "
                    "sea puntuada por la métrica (mean/std/w/b). Pesa el entrenamiento hacia esas filas")
    ap.add_argument("--mask-power", type=float, default=1.0, help="con --mask-coef: exponente del peso (0 = sin efecto)")
    ap.add_argument("--arch", choices=["rnn", "tcn"], default="rnn", help="rnn: GRU/LSTM (lo de siempre); "
                    "tcn: convolucional causal dilatada, otra clase de modelo")
    ap.add_argument("--tcn-channels", type=int, default=128)
    ap.add_argument("--tcn-layers", type=int, default=5, help="dilataciones 1, 4, 16, 64, 256: campo receptivo 683 filas")
    ap.add_argument("--tcn-kernel", type=int, default=3)
    ap.add_argument("--train-half", type=int, choices=[1, 2], default=0, help="entrenar solo con la mitad 1 o la "
                    "mitad 2 de las secuencias de train (disjuntas): fuerza diversidad entre modelos")
    ap.add_argument("--aux-targets", action="store_true", help="cabezas extra al entrenar: targets suavizados a 20 y "
                    "100 filas + movimiento del segundo instrumento. Dan una señal mucho menos ruidosa que el target "
                    "crudo; se recortan al guardar, así que no cuestan nada en inferencia")
    ap.add_argument("--aux-weight", type=float, default=0.3, help="peso de las tareas auxiliares en la loss")
    ap.add_argument("--slow-hidden", type=int, default=0, help="capa recurrente LENTA de N unidades que corre 1 de cada "
                    "--slow-every filas (0 = off): memoria larga barata y errores distintos a los de la familia actual")
    ap.add_argument("--slow-every", type=int, default=32, help="cada cuántas filas corre la capa lenta (divisor del chunk)")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (regularización)")
    ap.add_argument("--swa-start", type=int, default=0, help="promediar pesos desde esta época (0 = off); evalúa y guarda swa_model.pt")
    ap.add_argument("--save-epochs", action="store_true", help="guardar epoch_N.pt de cada época (para promediar offline)")
    ap.add_argument("--loss", choices=["wmse", "wp", "gwp", "seqwp"], default="wmse",
                    help="wmse: error cuadrático alineado; wp: 1-WP por secuencia (chunk); gwp: 1-WP GLOBAL del batch; "
                         "seqwp: 1-WP acumulado a lo largo de toda la secuencia (lo más parecido a la métrica)")
    ap.add_argument("--seqwp-decay", type=float, default=0.98, help="con --loss seqwp: olvido de los estadísticos por trozo")
    ap.add_argument("--wmse-anchor", type=float, default=0.0, help="con --loss seqwp: suma este peso por WMSE, para que la "
                    "escala de las predicciones no quede libre (la correlación es invariante a escala)")
    ap.add_argument("--gwp-weight", type=float, default=0.0, help="suma gwp_weight * (1 - WP global del batch) a la loss")
    ap.add_argument("--weight-power", type=float, default=1.0, help="pesos |y|^gamma en la WMSE (1 = métrica)")
    ap.add_argument("--lr-schedule", choices=["step", "cosine"], default="step", help="step: lr*decay^epoch; cosine: de lr a lr-min")
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--init-from", default=None, help="checkpoint (best_model.pt / swa_model.pt) para inicializar pesos (fine-tuning)")
    ap.add_argument("--distill-teachers", nargs="*", default=[], help="rutas .pt de teachers; sus predicciones promediadas son target blando")
    ap.add_argument("--distill-weight", type=float, default=1.0, help="peso de la loss blanda (WMSE contra el teacher)")
    ap.add_argument("--distill-cache", default=None, help="archivo .npz para precalcular UNA vez las predicciones de los teachers "
                    "(fp16, ~850 MB para todo train) y reutilizarlas en todas las épocas/corridas; '' = precalcular solo en RAM. "
                    "Sin esta opción los teachers corren en cada chunk de cada época (2x más lento)")
    ap.add_argument("--save-valid-preds", action="store_true", help="guardar predicciones del mejor modelo en valid (npz float16)")
    ap.add_argument("--top-k", type=int, default=0, help="lags/deltas por target de features.py (0,3,5,10,20)")
    ap.add_argument("--ranking", default=str((HERE.parents[1] if len(HERE.parents) > 1 else HERE) / "explore" / "feature_select" / "results" / "top_20_by_target.csv"))
    ap.add_argument("--sentinel-flags", action="store_true", help="a5/a6/a7: -5.1993 -> 0 + columna indicadora")
    ap.add_argument("--ema-features", action="store_true", help="memoria larga: desvío respecto de promedios móviles de "
                    "100/1.000/10.000 filas (12 columnas) + volatilidad lenta. El BPTT truncado en 512 pasos impide que "
                    "la red aprenda esto sola; en inferencia cuesta una cuenta por fila")
    ap.add_argument("--cross-features", action="store_true", help="agrega 24 combinaciones entre columnas de la misma fila "
                    "(spread, mid, huecos entre niveles, desequilibrio de volumen, i0 vs i1); causales y baratas en inferencia")
    ap.add_argument("--norm-rows", type=int, default=1000, help="filas por grupo (máx 200 grupos) para el normalizador")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=0, help="además evaluar cada N batches (0 = solo por época)")
    ap.add_argument("--workers", type=int, default=2, help="hilos de lectura de parquet")
    ap.add_argument("--threads", type=int, default=0, help="hilos de torch en CPU (0 = auto)")
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="carpeta de un run para retomar (usa checkpoint.pt)")
    ap.add_argument("--resume-epochs", type=int, default=None, help="con --resume: seguir hasta esta época (pisa el epochs del checkpoint)")
    ap.add_argument("--resume-swa-start", type=int, default=None, help="con --resume: SWA desde esta época (el promedio anterior no se conserva)")
    ap.add_argument("--resume-cache", default=None, help="con --resume: ruta del cache de soft targets en esta sesión (pisa distill_cache del checkpoint)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    batch = args.batch or (32 if on_gpu else 8)
    chunk = args.chunk or (512 if on_gpu else 256)
    if not on_gpu:
        torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) - 2))
    print(f"device={device}  batch={batch}  chunk={chunk}  torch={torch.__version__}", flush=True)

    if args.resume:   # las rutas de datos del run original, salvo que se pasen explícitas (en Kaggle no existe la ruta local)
        ck0 = torch.load(Path(args.resume) / "checkpoint.pt", map_location="cpu", weights_only=False)["config"]
        args.train_parquet = args.train_parquet or ck0["train_parquet"]
        args.valid_parquet = args.valid_parquet or ck0["valid_parquet"]
        del ck0
    args.train_parquet = args.train_parquet or [str(ROOT / "datasets" / "train.parquet")]
    args.valid_parquet = args.valid_parquet or [str(ROOT / "datasets" / "valid.parquet")]
    train = ParquetGroups(args.train_parquet)
    valid = ParquetGroups(args.valid_parquet)
    if not valid.has_scored:
        raise ValueError("valid necesita is_scored para la métrica oficial")
    train_ids = list(range(len(train))) if args.train_groups == "all" else select_groups(len(train), int(args.train_groups), args.seed)
    if args.train_half:     # mitades disjuntas, repartidas al azar pero con semilla fija para que sean reproducibles
        orden = np.random.default_rng(777).permutation(len(train_ids))
        corte = len(train_ids) // 2
        elegidas = orden[:corte] if args.train_half == 1 else orden[corte:]
        train_ids = [train_ids[i] for i in sorted(elegidas)]
        print(f"Mitad {args.train_half}: {len(train_ids)} secuencias de entrenamiento", flush=True)
    valid_ids = list(range(len(valid))) if str(args.valid_groups) == "all" else select_groups(len(valid), int(args.valid_groups), args.seed + 1)

    # ---- run dir / resume
    if args.resume:
        run_dir = Path(args.resume)
        ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = ck["config"]
        pipe = FeaturePipeline(cfg["top_k"], cfg["sentinel_flags"], Path(cfg["ranking"]), cfg.get("cross_features", False),
                               cfg.get("ema_features", False))
        pipe.mean, pipe.std = ck["pipeline"]["mean"], ck["pipeline"]["std"]
        if args.resume_epochs is not None:
            cfg["epochs"] = args.resume_epochs
        if args.resume_swa_start is not None:
            cfg["swa_start"] = args.resume_swa_start
        if args.resume_cache is not None:
            cfg["distill_cache"] = args.resume_cache
        train_ids, valid_ids = list(cfg["train_ids"]), list(cfg["valid_ids"])   # mismos datos que el run original
        batch, chunk = cfg["batch"], cfg["chunk"]                                 # mismo batch/chunk que el run original
        print(f"Retomando {run_dir} desde época {ck['epoch']} batch {ck['batch']} hasta la época {cfg['epochs']} (SWA desde {cfg.get('swa_start', 0)})", flush=True)
    else:
        name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = {**vars(args), "device": str(device), "batch": batch, "chunk": chunk,
               "train_ids": train_ids, "valid_ids": valid_ids, "n_train": len(train_ids)}
        pipe = FeaturePipeline(args.top_k, args.sentinel_flags, Path(args.ranking), args.cross_features, args.ema_features)
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
    n_aux = len(AUX_NAMES) if cfg.get("aux_targets") else 0
    if cfg.get("arch", "rnn") == "tcn":
        model = TCNModel(len(pipe.names), cfg["tcn_channels"], cfg["tcn_layers"], cfg["tcn_kernel"],
                         cfg.get("dropout", 0.0), n_aux).to(device)
    else:
        model = RNNModel(len(pipe.names), cfg["hidden"], cfg["layers"], cfg["cell"], cfg.get("dropout", 0.0),
                         cfg.get("slow_hidden", 0), cfg.get("slow_every", 32), n_aux).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = {"wmse": loss_wmse, "wp": loss_wp, "gwp": loss_gwp, "seqwp": None}[cfg["loss"]]
    if cfg.get("init_from") and not ck:
        init = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
        model.load_state_dict(init["state_dict"] if "state_dict" in init else init["model"])
        if "pipeline" in init:   # usar el normalizador del modelo inicial (los pesos lo asumen)
            pipe.mean, pipe.std = init["pipeline"]["mean"], init["pipeline"]["std"]
        print(f"Pesos inicializados desde {cfg['init_from']}", flush=True)
    teachers = [load_teacher(t, device) for t in cfg.get("distill_teachers", []) or []]
    soft_cache = None
    if teachers:
        print(f"Destilación: {len(teachers)} teachers, peso {cfg['distill_weight']}", flush=True)
        if cfg.get("distill_cache") is not None:
            soft_cache = precompute_soft(teachers, cfg["distill_teachers"], train, train_ids, batch, cfg["workers"], device, cfg["distill_cache"] or None)
            del teachers
            teachers = []
            if device.type == "cuda":
                torch.cuda.empty_cache()
    mask_mu = mask_sd = mask_w = mask_b = None
    if cfg.get("mask_coef"):
        mc = json.loads(Path(cfg["mask_coef"]).read_text(encoding="utf-8"))
        mask_mu = torch.as_tensor(np.asarray(mc["mean"], np.float32), device=device)
        mask_sd = torch.as_tensor(np.asarray(mc["std"], np.float32), device=device)
        mask_w = torch.as_tensor(np.asarray(mc["w"], np.float32), device=device)
        mask_b = float(mc["b"])
        print(f"Pesado por probabilidad de ser puntuada (AUC {mc.get('auc', 0):.3f}), potencia {cfg['mask_power']}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    if cfg.get("arch", "rnn") == "tcn":
        macs = len(pipe.names) * cfg["tcn_channels"] + cfg["tcn_layers"] * cfg["tcn_kernel"] * cfg["tcn_channels"] ** 2
        print(f"TCN {cfg['tcn_channels']} canales x{cfg['tcn_layers']} capas  campo receptivo={model.receptive} filas  "
              f"entradas={len(pipe.names)}  params={n_params:,}  ~{macs:,} operaciones por fila "
              f"(la LSTM 128x2 usa 254.208)", flush=True)
    else:
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
        sd = dict(state_dict if state_dict is not None else model.state_dict())
        if cfg.get("aux_targets"):      # el modelo guardado tiene solo las 2 salidas reales
            sd["head.weight"], sd["head.bias"] = sd["head.weight"][:2], sd["head.bias"][:2]
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in sd.items()},
                    "config": {k: cfg[k] for k in ("cell", "hidden", "layers")} |
                              {k: cfg.get(k) for k in ("arch", "tcn_channels", "tcn_layers", "tcn_kernel")} |
                              {k: cfg.get(k, 0 if k == "slow_hidden" else 32) for k in ("slow_hidden", "slow_every")},
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
            if mask_w is not None:          # pesar hacia las filas que la métrica va a mirar
                xr0 = torch.from_numpy(np.stack([d[2] for d in data])).to(device, non_blocking=True)
                pr = torch.sigmoid(((xr0 - mask_mu) / mask_sd) @ mask_w + mask_b)
                m = m * (pr / pr.mean()).pow(cfg["mask_power"])
                del xr0
            state = None
            seq_stats = None                      # estadísticos acumulados de la secuencia (--loss seqwp)
            aux_full = None
            if cfg.get("aux_targets"):            # se calculan sobre la secuencia entera, antes de cortarla en trozos
                xr = torch.from_numpy(np.stack([d[2] for d in data])).to(device, non_blocking=True)
                aux_full = aux_targets(y, xr, BIDP1[0], ASKP1[0])
                del xr
            if teachers:
                xraw = torch.from_numpy(np.stack([d[2] for d in data])).to(device, non_blocking=True)
                t_states = [None] * len(teachers)
            if soft_cache is not None:
                soft_full = torch.from_numpy(np.stack([soft_cache[i] for i in batches[bi]])).to(device, non_blocking=True).float()
            for s in range(0, SEQUENCE_LENGTH, chunk):
                e = min(s + chunk, SEQUENCE_LENGTH)
                pred, state = model(x[:, s:e], state, with_aux=aux_full is not None)
                if aux_full is not None:
                    pred, pred_aux = pred[..., :2], pred[..., 2:]
                if soft_cache is not None:
                    soft = soft_full[:, s:e]
                elif teachers:
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
                    elif cfg["loss"] == "seqwp":
                        loss, seq_stats = loss_seqwp(pred, y[:, s:e], m[:, s:e], seq_stats, cfg.get("seqwp_decay", 0.98))
                        if cfg.get("wmse_anchor", 0.0) > 0:
                            loss = loss + cfg["wmse_anchor"] * loss_wmse(pred, y[:, s:e], m[:, s:e])
                    else:
                        loss = loss_fn(pred, y[:, s:e], m[:, s:e])
                    if cfg.get("gwp_weight", 0.0) > 0:
                        loss = loss + cfg["gwp_weight"] * loss_gwp(pred, y[:, s:e], m[:, s:e])
                    if teachers or soft_cache is not None:
                        wsoft = y[:, s:e].clamp(-2.0, 2.0).abs() * m[:, s:e].unsqueeze(-1)   # mismos pesos que la métrica
                        loss = loss + cfg["distill_weight"] * ((wsoft * (pred - soft) ** 2).sum() / wsoft.sum().clamp_min(1e-8))
                    if aux_full is not None:
                        ma = m[:, s:e].unsqueeze(-1)
                        loss = loss + cfg["aux_weight"] * ((ma * (pred_aux - aux_full[:, s:e]) ** 2).sum()
                                                           / ma.sum().clamp_min(1e-8) / len(AUX_NAMES))
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
        if cfg.get("save_valid_preds"):   # permite armar tablas de ensambles en la laptop, sin GPU
            _save_valid_preds(model, pipe, valid, valid_ids, device, cfg["eval_batch"], run_dir / "valid_preds_swa.npz")
        history.append({"epoch": cfg["epochs"], "batch": 0, "tag": "swa", "wp": res["weighted_pearson"], "t0": res["t0"], "t1": res["t1"], "time": time.time()})
        (run_dir / "metrics.json").write_text(json.dumps({"best_valid_wp": best, "swa_valid_wp": res["weighted_pearson"], "history": history,
                                                          "config": cfg, "params": n_params}, indent=2, default=str), encoding="utf-8")
        model.load_state_dict(backup)
    print(f"Mejor valid WP: {best:.4f}   Resultados en {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
