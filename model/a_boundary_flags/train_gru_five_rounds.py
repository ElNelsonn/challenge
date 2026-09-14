r"""Cinco GRU con flags, comparadas con cinco GRU base ya entrenadas.

Ejecutar manualmente desde la raiz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\a_boundary_flags\train_gru_five_rounds.py

Para retomar una corrida:

    .\.venv\Scripts\python.exe my_solution\model\a_boundary_flags\train_gru_five_rounds.py --resume RUTA_DEL_RUN
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "my_solution" / "models_first_try"))
sys.path.insert(0, str(HERE))
from first_try_common import FitProgress, group_wp, read_group  # noqa: E402
from utils import (  # noqa: E402
    BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP,
)
from flags import A_COLUMNS, ATOL, BOUNDARY, FEATURE_NAMES, model_inputs  # noqa: E402


ROUNDS = 5
TRAIN_PER_ROUND = 100
VALID_GROUPS = 50
DEFAULT_REFERENCE_RUN = (
    ROOT / "my_solution" / "model" / "feature_select" / "results" / "group_repeats"
    / "run_20260913_205201_146618"
)
SUMMARY_FIELDS = (
    "round", "train_groups", "valid_groups", "base_t0", "base_t1", "base_mean",
    "flags_t0", "flags_t1", "flags_mean", "delta_t0", "delta_t1", "delta_mean",
    "eligible_blocks",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-path", type=Path, default=ROOT / "datasets" / "train.parquet")
    result.add_argument("--valid-path", type=Path, default=ROOT / "datasets" / "valid.parquet")
    result.add_argument("--output-dir", type=Path, default=HERE / "results")
    result.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN,
                        help="Corrida previa con las cinco GRU base completas")
    result.add_argument("--model-seed", type=int, default=2026)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--chunk-size", type=int, default=128)
    result.add_argument("--resume", type=Path, default=None,
                        help="Continuar sin repetir modelos terminados")
    return result


def train_with_flags(model, optimizer, train: pq.ParquetFile, groups: list[int],
                     mean: np.ndarray, scale: np.ndarray, epochs: int, chunk_size: int,
                     torch) -> list[float]:
    chunks_per_group = (SEQUENCE_LENGTH + chunk_size - 1) // chunk_size
    progress = FitProgress("a_boundary_flags", epochs * len(groups) * chunks_per_group,
                           every=chunks_per_group)
    completed = 0
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        rows = 0
        for position, group in enumerate(groups, 1):
            _, need, x, y, _ = read_group(train, group)
            inputs = model_inputs(x, mean, scale)
            hidden = None  # No cruzar secuencias.
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                optimizer.zero_grad(set_to_none=True)
                output, hidden = model(torch.from_numpy(inputs[start:end]).unsqueeze(0), hidden)
                first = max(0, WARMUP - start)
                if first < end - start:
                    targets = torch.from_numpy(y[start + first:end]).unsqueeze(0)
                    loss = torch.nn.functional.mse_loss(output[:, first:], targets)
                    loss.backward()
                    optimizer.step()
                    count = end - start - first
                    loss_sum += float(loss.item()) * count
                    rows += count
                hidden = hidden.detach()
                completed += 1
                progress.update(completed, f"epoca {epoch + 1}/{epochs}, grupo {position}/{len(groups)}")
        history.append(loss_sum / rows)
        print(f"a_boundary_flags: epoca {epoch + 1}/{epochs}, MSE={history[-1]:.6f}", flush=True)
    return history


def evaluate_with_flags(model, valid: pq.ParquetFile, groups: list[int],
                        mean: np.ndarray, scale: np.ndarray, chunk_size: int,
                        torch) -> tuple[dict, list[dict]]:
    accumulator = BlockAccumulator()
    by_group = []
    model.eval()
    for position, group in enumerate(groups, 1):
        seq_ix, need, x, y, scored = read_group(valid, group, validation=True)
        inputs = model_inputs(x, mean, scale)
        prediction = np.empty((SEQUENCE_LENGTH, len(TARGET_COLUMNS)), dtype=np.float32)
        hidden = None
        with torch.inference_mode():
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                output, hidden = model(torch.from_numpy(inputs[start:end]).unsqueeze(0), hidden)
                prediction[start:end] = output.squeeze(0).numpy()
        mask = need & scored
        accumulator.add(y, prediction, mask)
        by_group.append({"row_group": group, "seq_ix": seq_ix,
                         "selected_rows": int(mask.sum()), **group_wp(y, prediction, mask)})
        print(f"Validacion flags: {position}/{len(groups)} grupos", flush=True)
    return accumulator.result(), by_group


def save_variant(folder: Path, model, input_features: list[str], losses: list[float],
                 wp: dict, by_group: list[dict], mean: np.ndarray, scale: np.ndarray,
                 torch) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {"input_features": input_features, "train_mse_by_epoch": losses,
               "validation_wp_official_mask": wp}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (folder / "by_rowgroup.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_group[0]))
        writer.writeheader()
        writer.writerows(by_group)
    torch.save({"model_state_dict": model.state_dict(), "input_features": input_features,
                "mean": mean, "scale": scale, "boundary": BOUNDARY, "atol": ATOL,
                "architecture": {"kind": "GRU", "layers": 1, "hidden_size": 32,
                                 "outputs": list(TARGET_COLUMNS)}}, folder / "model.pt")


def completed_wp(folder: Path, expected_features: list[str]) -> dict | None:
    if not all((folder / name).exists() for name in ("metrics.json", "by_rowgroup.csv", "model.pt")):
        return None
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if metrics["input_features"] != expected_features:
        raise ValueError(f"Entradas incompatibles en {folder}")
    return metrics["validation_wp_official_mask"]


def prepare_groups(args, train_total: int, valid_total: int, torch):
    """Validar las cinco bases y reutilizar exactamente sus grupos y normalizadores."""
    reference = args.reference_run.resolve()
    source = json.loads((reference / "run_config.json").read_text(encoding="utf-8"))
    expected_source = {
        "train_dataset": str(args.train_path.resolve()),
        "validation_dataset": str(args.valid_path.resolve()),
        "model_seed": args.model_seed, "epochs": args.epochs,
        "chunk_size": args.chunk_size,
        "train_groups_per_round": TRAIN_PER_ROUND,
        "validation_groups_per_round": VALID_GROUPS,
        "sequence_length": SEQUENCE_LENGTH,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValueError("La corrida de referencia no coincide con datos o parametros")
    rounds = [record["train_row_groups"] for record in source["rounds"]]
    validation = source["validation_row_groups"]
    flat = [group for groups in rounds for group in groups]
    if (len(rounds) != ROUNDS or any(len(groups) != TRAIN_PER_ROUND for groups in rounds)
            or len(flat) != len(set(flat)) or source["train_row_groups"] != flat
            or any(not 0 <= group < train_total for group in flat)
            or len(validation) != VALID_GROUPS or len(validation) != len(set(validation))
            or any(not 0 <= group < valid_total for group in validation)
            or any(record["validation_row_groups"] != validation for record in source["rounds"])):
        raise ValueError("La referencia tiene grupos incompletos, repetidos o fuera de rango")

    references = []
    for number, groups in enumerate(rounds, 1):
        round_dir = reference / f"round_{number:02d}"
        round_config = json.loads((round_dir / "run_config.json").read_text(encoding="utf-8"))
        if (round_config.get("train_row_groups") != groups
                or round_config.get("validation_row_groups") != validation
                or round_config.get("model_seed") != args.model_seed
                or round_config.get("epochs") != args.epochs
                or round_config.get("chunk_size") != args.chunk_size
                or round_config.get("architecture") != {
                    "kind": "GRU", "layers": 1, "hidden_size": 32,
                    "outputs": list(TARGET_COLUMNS),
                }):
            raise ValueError(f"Parametros distintos en la base de ronda {number}")
        base_dir = round_dir / "base"
        metrics = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("input_features") != list(FEATURE_COLUMNS):
            raise ValueError(f"La base de ronda {number} no usa las 112 entradas originales")
        wp = metrics["validation_wp_official_mask"]
        with (base_dir / "by_rowgroup.csv").open(newline="", encoding="utf-8") as stream:
            observed_groups = [int(row["row_group"]) for row in csv.DictReader(stream)]
        if observed_groups != validation or wp["blocks"] != VALID_GROUPS:
            raise ValueError(f"La base de ronda {number} no evaluo los grupos esperados")
        checkpoint = torch.load(base_dir / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("input_features") != list(FEATURE_COLUMNS):
            raise ValueError(f"Checkpoint base incompatible en ronda {number}")
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        if (mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError(f"Normalizador base invalido en ronda {number}")
        references.append({"base_wp": wp, "mean": mean, "scale": scale})

    expected_run = {**expected_source, "reference_run": str(reference),
                    "boundary": BOUNDARY, "atol": ATOL,
                    "flag_features": list(FEATURE_NAMES)}
    if args.resume:
        run_dir = args.resume.resolve()
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if (any(config.get(key) != value for key, value in expected_run.items())
                or config.get("train_row_groups") != flat
                or config.get("validation_row_groups") != validation
                or config.get("rounds") != source["rounds"]):
            raise ValueError("El manifiesto de reanudacion no coincide con la referencia")
    else:
        run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        config = {**expected_run, "train_row_groups": flat,
                  "validation_row_groups": validation, "rounds": source["rounds"],
                  "variants": ["a_boundary_flags"],
                  "baseline": "WP y normalizador leidos de los checkpoints base de reference_run",
                  "a_columns": list(A_COLUMNS),
                  "normalization": "112 originales con la media/desviacion de la base; flags binarios"}
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir, rounds, validation, references


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
    run_dir, rounds, validation, references = prepare_groups(
        args, train.num_row_groups, valid.num_row_groups, torch,
    )
    print(f"Manifiesto: {run_dir / 'run_config.json'}", flush=True)

    class GRUModel(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.gru = nn.GRU(width, 32, num_layers=1, batch_first=True)
            self.head = nn.Linear(32, len(TARGET_COLUMNS))

        def forward(self, inputs, hidden=None):
            output, next_hidden = self.gru(inputs, hidden)
            return self.head(output), next_hidden

    flag_features = list(FEATURE_COLUMNS) + list(FEATURE_NAMES)
    rows = []
    for number, groups in enumerate(rounds, 1):
        print(f"Ronda {number}/{ROUNDS}", flush=True)
        round_dir = run_dir / f"round_{number:02d}"
        flag_dir = round_dir / "a_boundary_flags"
        base_wp = references[number - 1]["base_wp"]
        mean = references[number - 1]["mean"]
        scale = references[number - 1]["scale"]
        flag_wp = completed_wp(flag_dir, flag_features) if args.resume else None
        if flag_wp is None:
            torch.manual_seed(args.model_seed)
            model = GRUModel(len(flag_features))
            optimizer = torch.optim.Adam(model.parameters())
            losses = train_with_flags(model, optimizer, train, groups, mean, scale,
                                      args.epochs, args.chunk_size, torch)
            flag_wp, by_group = evaluate_with_flags(model, valid, validation, mean, scale,
                                                    args.chunk_size, torch)
            save_variant(flag_dir, model, flag_features, losses, flag_wp, by_group,
                         mean, scale, torch)
        row = {"round": number, "train_groups": len(groups), "valid_groups": len(validation),
               "base_t0": base_wp["t0"], "base_t1": base_wp["t1"],
               "base_mean": base_wp["weighted_pearson"],
               "flags_t0": flag_wp["t0"], "flags_t1": flag_wp["t1"],
               "flags_mean": flag_wp["weighted_pearson"],
               "delta_t0": flag_wp["t0"] - base_wp["t0"],
               "delta_t1": flag_wp["t1"] - base_wp["t1"],
               "delta_mean": flag_wp["weighted_pearson"] - base_wp["weighted_pearson"],
               "eligible_blocks": flag_wp["eligible_blocks"]}
        rows.append(row)
        with (round_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "variant", "input_features", "wp_t0", "wp_t1", "wp_mean",
                "delta_vs_base_t0", "delta_vs_base_t1", "delta_vs_base_mean",
            ))
            writer.writeheader()
            writer.writerow({"variant": "base_reused", "input_features": len(FEATURE_COLUMNS),
                             "wp_t0": base_wp["t0"], "wp_t1": base_wp["t1"],
                             "wp_mean": base_wp["weighted_pearson"]})
            writer.writerow({"variant": "a_boundary_flags", "input_features": len(flag_features),
                             "wp_t0": flag_wp["t0"], "wp_t1": flag_wp["t1"],
                             "wp_mean": flag_wp["weighted_pearson"],
                             "delta_vs_base_t0": row["delta_t0"],
                             "delta_vs_base_t1": row["delta_t1"],
                             "delta_vs_base_mean": row["delta_mean"]})
        write_summary(run_dir / "summary.csv", rows)
        print(f"Ronda {number}: delta WP medio={row['delta_mean']:+.6f}", flush=True)
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
