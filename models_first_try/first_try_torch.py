"""Simple causal GRU, LSTM and TCN experiments using the common evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from first_try_common import (
    TRAIN_PATH, VALID_PATH, FitProgress, build_parser, evaluate, prepare_run, read_group,
    sample_training_rows, save_metrics,
)
from utils import N_FEATURES, SEQUENCE_LENGTH, WARMUP


def run_torch_models(names: list[str], folder: Path) -> None:
    parser = build_parser("FirstTry secuencial: GRU, LSTM y TCN causales, sin K-fold.")
    parser.add_argument("--epochs", type=int, default=1, help="Épocas de entrenamiento (default: 1).")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser positivo")
    try:
        import torch
        from torch import nn
    except ImportError:
        parser.error("Falta PyTorch. Instala: python -m pip install torch")

    import os
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))  # dejar 2 hilos libres para el sistema
    train = pq.ParquetFile(TRAIN_PATH)
    valid = pq.ParquetFile(VALID_PATH)
    train_groups, valid_groups, run_dir = prepare_run(folder, args, train, valid)
    x_sample, _ = sample_training_rows(train, train_groups, args.rows_per_group, args.seed + 2)
    mean = x_sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_sample.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    class RecurrentModel(nn.Module):
        def __init__(self, kind: str):
            super().__init__()
            core = nn.GRU if kind == "GRU" else nn.LSTM
            self.core = core(N_FEATURES, 64, batch_first=True)
            self.head = nn.Linear(64, 2)

        def forward(self, x, state=None):
            hidden, next_state = self.core(x, state)
            return self.head(hidden), next_state

    class CausalTCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = N_FEATURES
            for dilation in (1, 2, 4):
                layers.extend([
                    nn.ConstantPad1d((2 * dilation, 0), 0.0),
                    nn.Conv1d(in_channels, 64, kernel_size=3, dilation=dilation),
                    nn.ReLU(),
                ])
                in_channels = 64
            self.network = nn.Sequential(*layers)
            self.head = nn.Linear(64, 2)

        def forward(self, x):
            hidden = self.network(x.transpose(1, 2)).transpose(1, 2)
            return self.head(hidden)

    for name in names:
        torch.manual_seed(args.seed)
        model = CausalTCN() if name == "TCN" else RecurrentModel(name)
        optimizer = torch.optim.Adam(model.parameters())
        def loss_fn(pred, target):
            # Alineada con la métrica del challenge: targets recortados a [-2, 2]
            # y cada fila pesa |target| (las filas con target ~0 no cuentan).
            y = target.clamp(-2.0, 2.0)
            w = y.abs()
            return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-8)
        chunks_per_group = (SEQUENCE_LENGTH + 127) // 128
        total_chunks = args.epochs * len(train_groups) * chunks_per_group
        progress = FitProgress(name, total_chunks, every=chunks_per_group)
        completed_chunks = 0

        for epoch in range(args.epochs):
            model.train()
            for position, group in enumerate(train_groups, start=1):
                _, _, x, y, _ = read_group(train, group)
                normalized = ((x - mean) / scale).astype(np.float32)
                inputs = torch.from_numpy(normalized).unsqueeze(0)
                targets = torch.from_numpy(y).unsqueeze(0)
                state = None
                for start in range(0, len(x), 128):
                    end = min(start + 128, len(x))
                    optimizer.zero_grad()
                    if name == "TCN":
                        # Three kernel-3 layers with dilations 1, 2, 4 have a
                        # 15-step receptive field. Keep only past context.
                        context_start = max(0, start - 14)
                        prediction_chunk = model(inputs[:, context_start:end])[:, start - context_start:]
                    else:
                        prediction_chunk, state = model(inputs[:, start:end], state)
                    first_scored = max(0, WARMUP - start)
                    if first_scored < end - start:
                        loss = loss_fn(prediction_chunk[:, first_scored:], targets[:, start + first_scored:end])
                        loss.backward()
                        optimizer.step()
                    if state is not None:
                        state = tuple(part.detach() for part in state) if isinstance(state, tuple) else state.detach()
                    completed_chunks += 1
                    progress.update(
                        completed_chunks,
                        f"época {epoch + 1}/{args.epochs}, grupo {position}/{len(train_groups)}",
                    )

        model.eval()

        def predict_group(features: np.ndarray) -> np.ndarray:
            normalized = ((features - mean) / scale).astype(np.float32)
            with torch.no_grad():
                tensor = torch.from_numpy(normalized).unsqueeze(0)
                if name == "TCN":
                    prediction = model(tensor)
                else:
                    prediction, _ = model(tensor)
            return prediction.squeeze(0).numpy().astype(np.float32)

        metrics, by_group = evaluate(predict_group, train, valid, train_groups, valid_groups)
        metrics["model"] = name
        metrics["training"] = {
            "epochs": args.epochs, "optimizer": "Adam defaults", "loss": "weighted MSE (clip ±2, w=|y|)",
            "hidden_or_channels": 64, "train_rows_per_group": SEQUENCE_LENGTH,
            "normalizer_sample_rows": len(x_sample),
            "truncated_bptt_chunk": 128 if name != "TCN" else None,
        }
        save_metrics(run_dir, name, metrics, by_group)
        model_dir = run_dir / name
        torch.save({
            "architecture": name,
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "training": metrics["training"],
        }, model_dir / "model.pt")
    print(f"Resultados en {run_dir.resolve()}")
