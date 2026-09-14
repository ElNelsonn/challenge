r"""Cinco rondas: GRU base frente a GRU con feedback causal de t0/t1.

Desde la raiz del proyecto, en PowerShell (ejecutar manualmente):

    .\.venv\Scripts\python.exe my_solution\model\model_target\train_gru_target_feedback.py

Para retomar una corrida interrumpida:

    .\.venv\Scripts\python.exe my_solution\model\model_target\train_gru_target_feedback.py --resume my_solution\model\model_target\results\run_YYYYMMDD_HHMMSS_ffffff

Cada ronda usa 100 grupos nuevos de train y los mismos 50 grupos nuevos de valid.
Los targets reales nunca se usan como entradas. El feedback sale de predicciones
anteriores del mismo modelo y se calcula igual en train y validacion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "my_solution" / "model" / "feature_select"))
from train_gru_five_group_pairs import choose_new_groups, previously_used_groups  # noqa: E402
from train_gru import normalize, train_variant, evaluate_variant  # noqa: E402
from first_try_common import FitProgress, group_wp, read_group  # noqa: E402
from utils import (  # noqa: E402
    BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP,
)
from feedback import FEATURE_NAMES, LAGS, PredictionHistory  # noqa: E402


ROUNDS = 5
TRAIN_GROUPS_PER_ROUND = 100
VALID_GROUPS = 50
SUMMARY_FIELDS = (
    "round", "train_groups", "valid_groups", "base_t0", "base_t1", "base_mean",
    "feedback_t0", "feedback_t1", "feedback_mean", "delta_t0", "delta_t1",
    "delta_mean", "eligible_blocks",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-path", type=Path, default=ROOT / "datasets" / "train.parquet")
    result.add_argument("--valid-path", type=Path, default=ROOT / "datasets" / "valid.parquet")
    result.add_argument("--output-dir", type=Path, default=HERE / "results")
    result.add_argument("--selection-seed", type=int, default=2028)
    result.add_argument("--model-seed", type=int, default=2026)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--chunk-size", type=int, default=128)
    result.add_argument("--resume", type=Path, default=None)
    return result


def fit_scales(train: pq.ParquetFile, groups: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estadisticas de los grupos de train, sin retener secuencias en memoria."""
    x_sum = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    x_sq_sum = np.zeros_like(x_sum)
    y_sum = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    y_sq_sum = np.zeros_like(y_sum)
    count = 0
    for index, group in enumerate(groups, 1):
        _, need, x, y, _ = read_group(train, group)
        xs, ys = x[need], y[need]
        x_sum += xs.sum(axis=0, dtype=np.float64)
        x_sq_sum += np.square(xs, dtype=np.float64).sum(axis=0)
        y_sum += ys.sum(axis=0, dtype=np.float64)
        y_sq_sum += np.square(ys, dtype=np.float64).sum(axis=0)
        count += len(xs)
        print(f"Normalizacion: {index}/{len(groups)} grupos", flush=True)
    x_mean = x_sum / count
    x_scale = np.sqrt(np.maximum(x_sq_sum / count - np.square(x_mean), 0.0))
    x_scale[x_scale < 1e-6] = 1.0
    y_mean = y_sum / count
    y_scale = np.sqrt(np.maximum(y_sq_sum / count - np.square(y_mean), 0.0))
    y_scale[y_scale < 1e-6] = 1.0
    return x_mean.astype(np.float32), x_scale.astype(np.float32), y_scale.astype(np.float32)


def feedback_sequence(model, x: np.ndarray, target_scale: np.ndarray, torch) -> np.ndarray:
    """Inferencia online: la prediccion del paso actual se guarda al terminarlo."""
    result = np.empty((len(x), len(TARGET_COLUMNS)), dtype=np.float32)
    hidden = None
    history = PredictionHistory(target_scale)
    model.eval()
    with torch.inference_mode():
        for step in range(len(x)):
            row = np.concatenate((x[step], history.features()))
            output, hidden = model(torch.from_numpy(row).view(1, 1, -1), hidden)
            prediction = output.view(-1).numpy()
            if not np.isfinite(prediction).all():
                raise ValueError(f"Prediccion no finita en paso {step}")
            result[step] = prediction
            history.append(prediction)
    return result


def train_feedback(model, optimizer, train: pq.ParquetFile, groups: list[int],
                   x_mean: np.ndarray, x_scale: np.ndarray, target_scale: np.ndarray,
                   epochs: int, chunk_size: int, torch) -> list[float]:
    """BPTT truncado; feedback de predicciones separado del grafo autograd."""
    chunks_per_group = (SEQUENCE_LENGTH + chunk_size - 1) // chunk_size
    progress = FitProgress("target_feedback", epochs * len(groups) * chunks_per_group,
                           every=chunks_per_group)
    completed = 0
    losses = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for group_index, group in enumerate(groups, 1):
            _, need, raw_x, y, _ = read_group(train, group)
            x = normalize(raw_x, x_mean, x_scale)
            hidden = None
            history = PredictionHistory(target_scale)
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                optimizer.zero_grad(set_to_none=True)
                outputs = []
                for step in range(start, end):
                    row = np.concatenate((x[step], history.features()))
                    output, hidden = model(torch.from_numpy(row).view(1, 1, -1), hidden)
                    prediction = output.view(-1)
                    if not torch.isfinite(prediction).all().item():
                        raise ValueError(f"Prediccion no finita en train grupo {group}, paso {step}")
                    history.append(prediction.detach().numpy())
                    if need[step]:
                        outputs.append(prediction)
                if outputs:
                    actual = torch.from_numpy(y[max(start, WARMUP):end])
                    loss = torch.nn.functional.mse_loss(torch.stack(outputs), actual)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.item()) * len(outputs)
                    row_count += len(outputs)
                hidden = hidden.detach()
                completed += 1
                progress.update(completed, f"epoca {epoch + 1}/{epochs}, grupo {group_index}/{len(groups)}")
        losses.append(loss_sum / row_count)
        print(f"target_feedback: epoca {epoch + 1}/{epochs}, MSE={losses[-1]:.6f}", flush=True)
    return losses


def evaluate_feedback(model, valid: pq.ParquetFile, groups: list[int],
                      x_mean: np.ndarray, x_scale: np.ndarray,
                      target_scale: np.ndarray, torch) -> tuple[dict, list[dict]]:
    accumulator = BlockAccumulator()
    rows = []
    for index, group in enumerate(groups, 1):
        seq_ix, need, raw_x, y, scored = read_group(valid, group, validation=True)
        prediction = feedback_sequence(model, normalize(raw_x, x_mean, x_scale), target_scale, torch)
        mask = need & scored
        accumulator.add(y, prediction, mask)
        rows.append({"row_group": group, "seq_ix": seq_ix,
                     "selected_rows": int(mask.sum()), **group_wp(y, prediction, mask)})
        print(f"Validacion feedback: {index}/{len(groups)} grupos", flush=True)
    return accumulator.result(), rows


def save_variant(folder: Path, model, feature_names: list[str], losses: list[float],
                 wp: dict, by_group: list[dict], x_mean: np.ndarray,
                 x_scale: np.ndarray, target_scale: np.ndarray | None, torch) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {"input_features": feature_names, "train_mse_by_epoch": losses,
               "validation_wp_official_mask": wp}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (folder / "by_rowgroup.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_group[0]))
        writer.writeheader()
        writer.writerows(by_group)
    torch.save({"model_state_dict": model.state_dict(), "input_features": feature_names,
                "x_mean": x_mean, "x_scale": x_scale, "target_scale": target_scale,
                "architecture": {"kind": "GRU", "layers": 1, "hidden_size": 32,
                                 "outputs": list(TARGET_COLUMNS)}}, folder / "model.pt")


def completed_wp(folder: Path, feature_names: list[str]) -> dict | None:
    if not all((folder / name).exists() for name in ("metrics.json", "by_rowgroup.csv", "model.pt")):
        return None
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if saved["input_features"] != feature_names:
        raise ValueError(f"Entradas incompatibles en {folder}")
    return saved["validation_wp_official_mask"]


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def manifest_groups(args, train_total: int, valid_total: int) -> tuple[Path, list[list[int]], list[int]]:
    if args.resume is not None:
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        expected = {"train_dataset": str(args.train_path.resolve()),
                    "validation_dataset": str(args.valid_path.resolve()),
                    "selection_seed": args.selection_seed, "model_seed": args.model_seed,
                    "epochs": args.epochs, "chunk_size": args.chunk_size,
                    "train_groups_per_round": TRAIN_GROUPS_PER_ROUND,
                    "validation_groups_per_round": VALID_GROUPS}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("El manifiesto no coincide con estos argumentos")
        rounds = [record["train_row_groups"] for record in manifest["rounds"]]
        valid_groups = manifest["validation_row_groups"]
    else:
        used_train, used_valid, sources = previously_used_groups(train_total, valid_total)
        rounds, valid_groups = choose_new_groups(
            train_total, valid_total, used_train, used_valid, args.selection_seed,
            rounds=ROUNDS, train_per_round=TRAIN_GROUPS_PER_ROUND, valid_count=VALID_GROUPS,
        )
        run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "train_dataset": str(args.train_path.resolve()),
            "validation_dataset": str(args.valid_path.resolve()),
            "selection_seed": args.selection_seed, "model_seed": args.model_seed,
            "epochs": args.epochs, "chunk_size": args.chunk_size,
            "train_groups_per_round": TRAIN_GROUPS_PER_ROUND,
            "validation_groups_per_round": VALID_GROUPS,
            "train_row_groups": [group for selected in rounds for group in selected],
            "validation_row_groups": valid_groups,
            "rounds": [{"round": i, "train_row_groups": selected,
                        "validation_row_groups": valid_groups}
                       for i, selected in enumerate(rounds, 1)],
            "variants": ["base", "target_feedback"],
            "target_feedback_features": list(FEATURE_NAMES),
            "target_feedback_lags": list(LAGS),
            "feedback_rule": "lag_k[t]=pred[t-k]; delta_k[t]=pred[t-1]-pred[t-1-k]",
            "previous_train_groups_excluded": len(used_train),
            "previous_validation_groups_excluded": len(used_valid),
            "previous_group_sources": sources,
        }
        # Reservar grupos antes de entrenar, incluso si se interrumpe la corrida.
        (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    flat = [group for selected in rounds for group in selected]
    if (len(rounds) != ROUNDS or any(len(selected) != TRAIN_GROUPS_PER_ROUND for selected in rounds)
            or len(flat) != len(set(flat)) or any(not 0 <= group < train_total for group in flat)
            or len(valid_groups) != VALID_GROUPS or len(valid_groups) != len(set(valid_groups))
            or any(not 0 <= group < valid_total for group in valid_groups)
            or any(record["validation_row_groups"] != valid_groups for record in manifest["rounds"])):
        raise ValueError("Grupos incompletos, repetidos o fuera de rango")
    return run_dir, rounds, valid_groups


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.chunk_size < 1:
        raise SystemExit("--epochs y --chunk-size deben ser positivos")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("Falta PyTorch: python -m pip install torch") from exc
    torch.set_num_threads(1)
    train = pq.ParquetFile(args.train_path)
    valid = pq.ParquetFile(args.valid_path)
    run_dir, rounds, valid_groups = manifest_groups(args, train.num_row_groups, valid.num_row_groups)
    print(f"Manifiesto: {run_dir / 'run_config.json'}", flush=True)

    class GRUModel(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.gru = nn.GRU(width, 32, num_layers=1, batch_first=True)
            self.head = nn.Linear(32, len(TARGET_COLUMNS))

        def forward(self, inputs, hidden=None):
            output, next_hidden = self.gru(inputs, hidden)
            return self.head(output), next_hidden

    rows = []
    for round_number, train_groups in enumerate(rounds, 1):
        print(f"Ronda {round_number}/{ROUNDS}", flush=True)
        round_dir = run_dir / f"round_{round_number:02d}"
        base_dir = round_dir / "base"
        feedback_dir = round_dir / "target_feedback"
        base_features = list(FEATURE_COLUMNS)
        feedback_features = base_features + list(FEATURE_NAMES)
        base_wp = completed_wp(base_dir, base_features) if args.resume else None
        feedback_wp = completed_wp(feedback_dir, feedback_features) if args.resume else None
        if base_wp is None or feedback_wp is None:
            x_mean, x_scale, target_scale = fit_scales(train, train_groups)
        if base_wp is None:
            torch.manual_seed(args.model_seed)
            model = GRUModel(len(base_features))
            optimizer = torch.optim.Adam(model.parameters())
            losses = train_variant(model, optimizer, train, train_groups, [], x_mean, x_scale,
                                   args.epochs, args.chunk_size, "base", torch)
            base_wp, by_group = evaluate_variant(model, valid, valid_groups, [], x_mean, x_scale,
                                                 args.chunk_size, torch)
            save_variant(base_dir, model, base_features, losses, base_wp, by_group,
                         x_mean, x_scale, None, torch)
        if feedback_wp is None:
            torch.manual_seed(args.model_seed)
            model = GRUModel(len(feedback_features))
            optimizer = torch.optim.Adam(model.parameters())
            losses = train_feedback(model, optimizer, train, train_groups, x_mean, x_scale,
                                    target_scale, args.epochs, args.chunk_size, torch)
            feedback_wp, by_group = evaluate_feedback(model, valid, valid_groups, x_mean,
                                                       x_scale, target_scale, torch)
            save_variant(feedback_dir, model, feedback_features, losses, feedback_wp,
                         by_group, x_mean, x_scale, target_scale, torch)
        row = {
            "round": round_number, "train_groups": len(train_groups), "valid_groups": len(valid_groups),
            "base_t0": base_wp["t0"], "base_t1": base_wp["t1"],
            "base_mean": base_wp["weighted_pearson"],
            "feedback_t0": feedback_wp["t0"], "feedback_t1": feedback_wp["t1"],
            "feedback_mean": feedback_wp["weighted_pearson"],
            "delta_t0": feedback_wp["t0"] - base_wp["t0"],
            "delta_t1": feedback_wp["t1"] - base_wp["t1"],
            "delta_mean": feedback_wp["weighted_pearson"] - base_wp["weighted_pearson"],
            "eligible_blocks": feedback_wp["eligible_blocks"],
        }
        rows.append(row)
        write_summary(run_dir / "summary.csv", rows)
        print(f"Ronda {round_number}: delta WP medio={row['delta_mean']:+.6f}", flush=True)
    aggregate = {}
    for target in ("t0", "t1", "mean"):
        deltas = np.asarray([row[f"delta_{target}"] for row in rows], dtype=np.float64)
        aggregate[target] = {"mean_delta": float(deltas.mean()),
                             "median_delta": float(np.median(deltas)),
                             "rounds_improved": int(np.count_nonzero(deltas > 0)),
                             "rounds": len(rows)}
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Resumen: {run_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
