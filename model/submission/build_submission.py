r"""Arma un zip de submission: solution.py + weights.npz con un ensamble de N redes recurrentes (GRU o LSTM, promedio).

Acepta checkpoints de train_official_gpu_v1 ({state_dict, config, pipeline}) y de
temporal_experiment ({model, architecture, normalizer}). Todas las redes deben tener 2 capas y 112 entradas;
el hidden puede ser distinto (se rellena a Hmax) y la celda también (GRU: reordena compuertas PyTorch
(r,z,n) -> kernel (z,r,n); LSTM: orden PyTorch i,f,g,o sin cambios; kernel v3).
Exporta las matrices traspuestas y cuantizadas por fila (--bits 8 por defecto; 16 o 32 disponibles).
Verifica sobre N secuencias de valid: paridad con PyTorch float32 (con int8 la diferencia esperable es
~1e-2 en la salida; la métrica no cambia), métrica GLOBAL, µs/fila.

Uso (desde challenge/):
  python model/submission/build_submission.py --name ens3 --models A.pt B.pt C.pt [--weights ...] [--bits 8] --verify-seqs 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
ROOT = CHALLENGE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from utils import DataPoint, FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, validate_sequence  # noqa: E402
from eval_ensemble import GlobalWP  # noqa: E402


def load_any(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        sd, H, L, n_in = ck["state_dict"], ck["config"]["hidden"], ck["config"]["layers"], ck["n_inputs"]
        mean, std = ck["pipeline"]["mean"], ck["pipeline"]["std"]
        cell, clip = ck["config"].get("cell", "gru"), float("inf")        # train_official_gpu_v1 no recorta
    else:
        sd, H, L, n_in = ck["model"], ck["architecture"]["hidden"], ck["architecture"]["layers"], ck["architecture"]["width"]
        mean, std, cell = ck["normalizer"]["mean"], ck["normalizer"]["std"], "gru"
        clip = 20.0                                             # temporal_experiment.py: NORMALIZED_CLIP = 20
    assert cell in ("gru", "lstm") and L == 2 and n_in == 112, f"{path}: se requiere GRU/LSTM 2 capas 112 entradas (tiene {cell} {L} capas {n_in})"
    return sd, H, np.asarray(mean, np.float32), np.asarray(std, np.float32), clip, cell


def make_net(sd, H, cell):
    """Red PyTorch de referencia (mismo formato que el trainer)."""
    from torch import nn
    core = (nn.GRU if cell == "gru" else nn.LSTM)(112, H, num_layers=2, batch_first=True)
    net = nn.ModuleDict({"core": core, "head": nn.Linear(H, 2)})
    net.load_state_dict(sd)
    net.eval()
    return net


def quantize_rows(w, bits):
    """(3h, k) float32 -> (entero (3h, k), escala (3h,)) simétrico por fila. bits=32: sin cuantizar, escala 1."""
    if bits == 32:
        return w.astype(np.float32), np.ones(w.shape[0], np.float32)
    qmax = 2 ** (bits - 1) - 1
    sc = (np.abs(w).max(axis=1) / qmax).clip(1e-12).astype(np.float32)
    q = np.clip(np.rint(w / sc[:, None]), -qmax, qmax).astype(np.int8 if bits == 8 else np.int16)
    return q, sc


def export_weights(paths, weights, out, bits=8):
    models = [load_any(p) for p in paths]
    hs = [m[1] for m in models]
    H = max(hs)   # los arrays se rellenan con ceros hasta el H máximo; el kernel usa el H real de cada modelo
    qdt = {8: np.int8, 16: np.int16, 32: np.float32}[bits]

    def pad_rows(w, h):  # (h, k) -> (H, k)
        w = np.asarray(w, np.float32)
        out = np.zeros((H,) + w.shape[1:], np.float32)
        out[:h] = w
        return out

    def to_np(t):
        return t.numpy() if hasattr(t, "numpy") else np.asarray(t)

    def gate_order(w, h, cell):
        """GRU: PyTorch (r,z,n) -> kernel (z,r,n). LSTM: PyTorch (i,f,g,o) tal cual."""
        if cell == "gru":
            return np.concatenate([w[h:2 * h], w[:h], w[2 * h:]], axis=0)
        return w

    W = {}
    # matrices: ordenar compuertas, cuantizar por fila, trasponer a (K, G*h) y rellenar a (K_pad, 4*H_max)
    for k, v, kin in (("W1", "core.weight_ih_l0", 112), ("R1", "core.weight_hh_l0", None),
                      ("W2", "core.weight_ih_l1", None), ("R2", "core.weight_hh_l1", None)):
        mats, scales = [], []
        for m in models:
            h, cell = m[1], m[5]
            w = gate_order(to_np(m[0][v]).astype(np.float32), h, cell)                # (G*h, K)
            q, sc = quantize_rows(w, bits)
            K_pad = 112 if kin == 112 else H
            buf = np.zeros((K_pad, 4 * H), qdt)
            buf[: q.shape[1], : q.shape[0]] = q.T
            sbuf = np.zeros(4 * H, np.float32)
            sbuf[: q.shape[0]] = sc
            mats.append(np.ascontiguousarray(buf))
            scales.append(sbuf)
        W[k + "T"] = np.stack(mats)
        W["s" + k] = np.stack(scales)
    for k, v in (("Wb1", "core.bias_ih_l0"), ("Rb1", "core.bias_hh_l0"), ("Wb2", "core.bias_ih_l1"), ("Rb2", "core.bias_hh_l1")):
        mats = []
        for m in models:
            h, cell = m[1], m[5]
            b = gate_order(to_np(m[0][v]).astype(np.float32), h, cell)
            buf = np.zeros(4 * H, np.float32)
            buf[: len(b)] = b
            mats.append(buf)
        W[k] = np.stack(mats)
    W["Wo"] = np.stack([pad_rows(m[0]["head.weight"].numpy().T, m[1]) for m in models])
    W["bo"] = np.stack([m[0]["head.bias"].numpy().astype(np.float32) for m in models])
    W["hidden_per_model"] = np.array(hs, np.int64)
    W["cell_per_model"] = np.array([0 if m[5] == "gru" else 1 for m in models], np.int64)
    W["mean"] = np.stack([m[2] for m in models])
    W["std"] = np.stack([m[3] for m in models])
    W["input_clip"] = np.array([m[4] for m in models], np.float32)
    W["hidden"] = np.int64(H)
    W["n_models"] = np.int64(len(models))
    w = np.asarray(weights if weights else [1.0 / len(models)] * len(models), np.float32)
    assert len(w) == len(models) and abs(w.sum() - 1) < 1e-6, "--weights debe tener un peso por modelo y sumar 1"
    W["model_weights"] = w
    W["bits"] = np.int64(bits)
    np.savez(out, **W)
    return H, len(models)


def verify(build_dir, paths, weights, n_seqs):
    sys.path.insert(0, str(build_dir))
    import importlib
    import torch
    import pyarrow.parquet as pq
    from torch import nn
    sol = importlib.import_module("solution")
    importlib.reload(sol)
    model = sol.PredictionModel()
    refs = []
    for p in paths:
        sd, Hm, mean, std, clip, cell = load_any(p)
        refs.append((make_net(sd, Hm, cell), mean, std, clip))
    w = np.asarray(weights if weights else [1.0 / len(paths)] * len(paths), np.float32)
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    gacc = GlobalWP()
    max_diff, t_pred, rows = 0.0, 0.0, 0
    for g in range(n_seqs):
        tb = pf.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(tb)
        X = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        P = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
        t = time.perf_counter()
        for s in range(SEQUENCE_LENGTH):
            v = model.predict(DataPoint(seq, s, bool(need[s]), X[s]))
            if need[s]:
                P[s] = v
        t_pred += time.perf_counter() - t
        rows += SEQUENCE_LENGTH
        with torch.no_grad():
            R = sum(wi * net["head"](net["core"](torch.from_numpy(np.clip((X - mean) / std, -clip, clip).astype(np.float32)).unsqueeze(0))[0]).squeeze(0).numpy()
                    for wi, (net, mean, std, clip) in zip(w, refs))
        max_diff = max(max_diff, float(np.abs(P[need] - R[need]).max()))
        gacc.add(Y, P, need & tb["is_scored"].to_numpy().astype(bool))
    r = gacc.result()
    us = t_pred / rows * 1e6
    print(f"paridad: max |solution - PyTorch| = {max_diff:.2e}")
    print(f"{n_seqs} seqs de valid: GLOBAL WP={r['wp']:.4f} (t0 {r['t0']:.4f}, t1 {r['t1']:.4f})")
    print(f"tiempo: {us:.1f} us/fila  ->  {us * 37_460_000 / 60e6:.1f} min proyectados para 37.46M filas (límite 60)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="rutas a .pt (1 o más GRU/LSTM de 2 capas; hidden puede variar)")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="pesos del promedio (default: iguales)")
    ap.add_argument("--verify-seqs", type=int, default=10)
    ap.add_argument("--bits", type=int, choices=[8, 16, 32], default=8, help="precisión de las matrices GRU (int8 por fila = sin pérdida medible, 1.6x más rápido)")
    ap.add_argument("--out", default=str(CHALLENGE / "submissions"))
    args = ap.parse_args()
    build_dir = Path(args.out) / args.name
    build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "solution.py", build_dir / "solution.py")
    H, n = export_weights(args.models, args.weights, build_dir / "weights.npz", args.bits)
    zip_path = Path(args.out) / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(build_dir / "solution.py", "solution.py")
        z.write(build_dir / "weights.npz", "weights.npz")
    cells = [load_any(p)[5].upper() for p in args.models]
    (build_dir / "BUILD.json").write_text(json.dumps({"models": args.models, "cells": cells, "weights": args.weights, "hidden": H, "n_models": n, "bits": args.bits}, indent=2))
    print(f"zip: {zip_path}  ({zip_path.stat().st_size / 1e6:.2f} MB)   {n} redes ({'+'.join(cells)}), hidden máx {H}, "
          + (f"matrices int{args.bits}" if args.bits < 32 else "float32"))
    if args.verify_seqs:
        verify(build_dir, args.models, args.weights, args.verify_seqs)


if __name__ == "__main__":
    main()
