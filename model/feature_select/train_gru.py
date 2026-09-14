r"""Comparar GRU base contra top 5/10/20 de correlaciones temporales.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru.py

Ejemplo con más secuencias:

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru.py --train-groups 100 --valid-groups 20 --epochs 2

Solo se ejecuta al invocar este archivo. Usa valid.parquet como test local;
el test oficial no está disponible en el starterpack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "my_solution" / "models_first_try"))
from first_try_common import (  # noqa: E402
    TRAIN_PATH, VALID_PATH, FitProgress, group_wp, read_group, select_groups,
)
from features import (  # noqa: E402
    TemporalFeature, build_sequence, load_rankings, select_temporal_features,
)
from utils import BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_RANKING = ROOT / "my_solution" / "explore" / "feature_select" / "results" / "top_20_by_target.csv"
VARIANTS = (("base", 0), ("top_5", 5), ("top_10", 10), ("top_20", 20))
COMPARISON_FIELDS = (
    "variant", "top_k_per_target", "input_features", "added_features",
    "wp_t0", "wp_t1", "wp_mean", "delta_vs_base_t0", "delta_vs_base_t1",
    "delta_vs_base_mean", "eligible_blocks", "validation_groups",
)


def parser(train_groups_default: int = 20, valid_groups_default: int = 10) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    result.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    result.add_argument("--valid-path", type=Path, default=VALID_PATH)
    result.add_argument("--output-dir", type=Path, default=HERE / "results")
    result.add_argument("--train-groups", type=int, default=train_groups_default)
    result.add_argument("--valid-groups", type=int, default=valid_groups_default)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--chunk-size", type=int, default=128)
    return result


def normalize(sequence: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """NaN de historia insuficiente -> cero estandarizado para la GRU solamente."""
    normalized = (sequence - mean) / scale
    return np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def fit_normalizer(parquet: pq.ParquetFile, groups: list[int], specs) -> tuple[np.ndarray, np.ndarray]:
    width = len(FEATURE_COLUMNS) + len(specs)
    sums = np.zeros(width, dtype=np.float64)
    sums_squared = np.zeros(width, dtype=np.float64)
    n_rows = 0
    for position, group in enumerate(groups, start=1):
        _, need, x, _, _ = read_group(parquet, group)
        augmented = build_sequence(x, specs)
        selected = augmented[need]
        if not np.isfinite(selected).all():
            raise ValueError(f"Features no finitas en pasos requeridos del grupo {group}")
        sums += selected.sum(axis=0, dtype=np.float64)
        sums_squared += np.square(selected, dtype=np.float64).sum(axis=0)
        n_rows += len(selected)
        print(f"Normalización: {position}/{len(groups)} grupos", flush=True)
    mean = sums / n_rows
    variance = np.maximum(sums_squared / n_rows - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def variant_normalizer(
    full_specs, selected_specs, full_mean: np.ndarray, full_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = {spec.name: len(FEATURE_COLUMNS) + index for index, spec in enumerate(full_specs)}
    indices = list(range(len(FEATURE_COLUMNS))) + [positions[spec.name] for spec in selected_specs]
    return full_mean[indices], full_scale[indices]


def train_variant(
    model, optimizer, train: pq.ParquetFile, groups: list[int], specs,
    mean: np.ndarray, scale: np.ndarray, epochs: int, chunk_size: int, name: str,
    torch,
) -> list[float]:
    loss_fn = torch.nn.MSELoss()
    chunks_per_group = (SEQUENCE_LENGTH + chunk_size - 1) // chunk_size
    progress = FitProgress(name, epochs * len(groups) * chunks_per_group, every=chunks_per_group)
    completed = 0
    history = []
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        rows_total = 0
        for group_position, group in enumerate(groups, start=1):
            _, need, x, y, _ = read_group(train, group)
            features = normalize(build_sequence(x, specs), mean, scale)
            hidden = None  # Nunca continuar el estado recurrente a otra secuencia.
            for start in range(0, SEQUENCE_LENGTH, chunk_size):
                end = min(start + chunk_size, SEQUENCE_LENGTH)
                inputs = torch.from_numpy(features[start:end]).unsqueeze(0)
                targets = torch.from_numpy(y[start:end]).unsqueeze(0)
                optimizer.zero_grad()
                prediction, hidden = model(inputs, hidden)
                first_required = max(0, WARMUP - start)
                if first_required < end - start:
                    loss = loss_fn(prediction[:, first_required:], targets[:, first_required:])
                    loss.backward()
                    optimizer.step()
                    count = end - start - first_required
                    loss_total += float(loss.item()) * count
                    rows_total += count
                hidden = hidden.detach()  # BPTT truncado; no se pierde contexto causal.
                completed += 1
                progress.update(completed, f"época {epoch + 1}/{epochs}, grupo {group_position}/{len(groups)}")
        history.append(loss_total / rows_total)
        print(f"{name}: época {epoch + 1}/{epochs}, MSE={history[-1]:.6f}", flush=True)
    return history


def predict_sequence(model, features: np.ndarray, chunk_size: int, torch) -> np.ndarray:
    predictions = np.empty((len(features), len(TARGET_COLUMNS)), dtype=np.float32)
    hidden = None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), chunk_size):
            end = min(start + chunk_size, len(features))
            inputs = torch.from_numpy(features[start:end]).unsqueeze(0)
            output, hidden = model(inputs, hidden)
            predictions[start:end] = output.squeeze(0).numpy()
    return predictions


def evaluate_variant(
    model, valid: pq.ParquetFile, groups: list[int], specs,
    mean: np.ndarray, scale: np.ndarray, chunk_size: int, torch,
) -> tuple[dict, list[dict]]:
    accumulator = BlockAccumulator()
    by_group = []
    for position, group in enumerate(groups, start=1):
        seq_ix, need, x, y, scored = read_group(valid, group, validation=True)
        features = normalize(build_sequence(x, specs), mean, scale)
        prediction = predict_sequence(model, features, chunk_size, torch)
        if not np.isfinite(prediction[need]).all():
            raise ValueError(f"Predicciones no finitas en valid row group {group}")
        mask = need & scored
        accumulator.add(y, prediction, mask)
        by_group.append({
            "row_group": group, "seq_ix": seq_ix, "selected_rows": int(mask.sum()),
            **group_wp(y, prediction, mask),
        })
        print(f"Validación: {position}/{len(groups)} grupos", flush=True)
    return accumulator.result(), by_group


def comparison_row(name: str, top_k: int, added: int, wp: dict, base: dict | None,
                   validation_groups: int) -> dict:
    return {
        "variant": name, "top_k_per_target": top_k,
        "input_features": len(FEATURE_COLUMNS) + added, "added_features": added,
        "wp_t0": wp["t0"], "wp_t1": wp["t1"], "wp_mean": wp["weighted_pearson"],
        "delta_vs_base_t0": wp["t0"] - base["t0"] if base is not None else None,
        "delta_vs_base_t1": wp["t1"] - base["t1"] if base is not None else None,
        "delta_vs_base_mean": (
            wp["weighted_pearson"] - base["weighted_pearson"] if base is not None else None
        ),
        "eligible_blocks": wp["eligible_blocks"], "validation_groups": validation_groups,
    }


def select_variant_specs(rankings, top_k: int,
                         extras: tuple[TemporalFeature, ...] = ()) -> list[TemporalFeature]:
    specs = select_temporal_features(rankings, top_k)
    selected_names = {spec.name for spec in specs}
    ranked_t0 = set(rankings["t0"])
    for extra in extras:
        if extra not in ranked_t0 or extra.name in selected_names:
            raise ValueError(f"Feature adicional inválida o duplicada: {extra.name}")
        specs.append(extra)
        selected_names.add(extra.name)
    return specs


def run_with_groups(
    args: argparse.Namespace,
    train_groups: list[int],
    valid_groups: list[int],
    variants=VARIANTS,
    extra_features_by_variant: dict[str, tuple[TemporalFeature, ...]] | None = None,
    run_dir: Path | None = None,
    model_seed: int | None = None,
    resume: bool = False,
) -> list[dict]:
    """Entrena variantes usando exactamente los row groups recibidos."""
    if args.epochs < 1 or args.chunk_size < 1:
        raise SystemExit("--epochs y --chunk-size deben ser positivos")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("Falta PyTorch en el entorno: python -m pip install torch") from exc
    torch.set_num_threads(1)

    rankings = load_rankings(args.ranking)
    extras_by_variant = extra_features_by_variant or {}
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("Los nombres de variantes deben ser únicos")
    for name, top_k in variants:
        select_variant_specs(rankings, top_k, extras_by_variant.get(name, ()))
    full_specs = select_temporal_features(rankings, 20)
    train = pq.ParquetFile(args.train_path)
    valid = pq.ParquetFile(args.valid_path)
    for label, groups, total in (("train", train_groups, train.num_row_groups),
                                 ("validación", valid_groups, valid.num_row_groups)):
        if not groups or len(set(groups)) != len(groups) or any(not 0 <= group < total for group in groups):
            raise ValueError(f"Grupos de {label} vacíos, repetidos o fuera de rango")

    print(f"Train: {len(train_groups)} secuencias; valid: {len(valid_groups)} secuencias")
    full_mean, full_scale = fit_normalizer(train, train_groups, full_specs)
    if run_dir is None:
        run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=resume)
    if model_seed is None:
        model_seed = args.seed
    config = {
        "ranking": str(args.ranking.resolve()),
        "ranking_sha256": hashlib.sha256(args.ranking.read_bytes()).hexdigest(),
        "train_dataset": str(args.train_path.resolve()),
        "validation_dataset": str(args.valid_path.resolve()),
        "train_row_groups": train_groups, "validation_row_groups": valid_groups,
        "variants": [
            {"name": name, "top_k_per_target": top_k,
             "extra_features": [spec.name for spec in extras_by_variant.get(name, ())]}
            for name, top_k in variants
        ],
        "seed": args.seed, "model_seed": model_seed,
        "epochs": args.epochs, "chunk_size": args.chunk_size,
        "architecture": {"kind": "GRU", "layers": 1, "hidden_size": 32, "outputs": list(TARGET_COLUMNS)},
        "optimizer": "Adam defaults", "loss": "MSE on steps >= 99",
        "normalizer": "mean/std fitted on required train rows only; warm-up missing history -> standardized zero",
        "evaluation": "validation Weighted Pearson on need_prediction AND is_scored",
    }
    config_path = run_dir / "run_config.json"
    if resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("ranking_sha256", "train_row_groups", "validation_row_groups",
                    "seed", "model_seed", "epochs", "chunk_size", "variants"):
            if previous.get(key) != config[key]:
                raise ValueError(f"La configuración previa de {run_dir} cambió: {key}")
    else:
        if resume and any(run_dir.iterdir()):
            raise ValueError(f"Directorio parcial sin run_config.json: {run_dir}")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    class GRUModel(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.gru = nn.GRU(input_size, 32, num_layers=1, batch_first=True)
            self.head = nn.Linear(32, len(TARGET_COLUMNS))

        def forward(self, inputs, hidden=None):
            output, next_hidden = self.gru(inputs, hidden)
            return self.head(output), next_hidden

    comparisons = []
    base_wp = None
    for name, top_k in variants:
        specs = select_variant_specs(rankings, top_k, extras_by_variant.get(name, ()))
        feature_names = [*FEATURE_COLUMNS, *(spec.name for spec in specs)]
        mean, scale = variant_normalizer(full_specs, specs, full_mean, full_scale)
        variant_dir = run_dir / name
        metrics_path = variant_dir / "metrics.json"
        if resume and all((variant_dir / filename).exists() for filename in
                          ("metrics.json", "by_rowgroup.csv", "model.pt")):
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved["input_features"] != feature_names or saved["top_k_per_target"] != top_k:
                raise ValueError(f"Checkpoint incompatible en {variant_dir}")
            wp = saved["validation_wp_official_mask"]
            if name == "base":
                base_wp = wp
            comparisons.append(comparison_row(name, top_k, len(specs), wp,
                                              base_wp, len(valid_groups)))
            print(f"{name}: resultado completo, se omite entrenamiento", flush=True)
            continue
        variant_dir.mkdir(exist_ok=resume)
        torch.manual_seed(model_seed)
        model = GRUModel(len(feature_names))
        optimizer = torch.optim.Adam(model.parameters())
        print(f"\n{name}: {len(feature_names)} entradas ({len(specs)} agregadas)", flush=True)
        loss_history = train_variant(
            model, optimizer, train, train_groups, specs, mean, scale,
            args.epochs, args.chunk_size, name, torch,
        )
        wp, by_group = evaluate_variant(
            model, valid, valid_groups, specs, mean, scale, args.chunk_size, torch,
        )
        if name == "base":
            base_wp = wp
        comparison = comparison_row(name, top_k, len(specs), wp, base_wp, len(valid_groups))
        comparisons.append(comparison)
        metrics = {
            "variant": name, "top_k_per_target": top_k, "input_features": feature_names,
            "added_features": [spec.name for spec in specs], "train_mse_by_epoch": loss_history,
            "validation_wp_official_mask": wp, "comparison": comparison,
        }
        (variant_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        with (variant_dir / "by_rowgroup.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(by_group[0]))
            writer.writeheader()
            writer.writerows(by_group)
        torch.save({
            "state_dict": model.state_dict(), "architecture": config["architecture"],
            "input_features": feature_names, "temporal_features": [
                {"base_feature": spec.base_feature, "transform": spec.transform, "lag": spec.lag}
                for spec in specs
            ], "feature_mean": mean, "feature_scale": scale,
        }, variant_dir / "model.pt")
        print(f"{name}: WP t0={wp['t0']:.6f}, t1={wp['t1']:.6f}, media={wp['weighted_pearson']:.6f}")

    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"\nComparación: {(run_dir / 'comparison.csv').resolve()}")
    return comparisons


def main(variants=VARIANTS, train_groups_default: int = 20,
         valid_groups_default: int = 10,
         extra_features_by_variant: dict[str, tuple[TemporalFeature, ...]] | None = None) -> None:
    args = parser(train_groups_default, valid_groups_default).parse_args()
    train = pq.ParquetFile(args.train_path)
    valid = pq.ParquetFile(args.valid_path)
    try:
        train_groups = select_groups(train.num_row_groups, args.train_groups, args.seed)
        valid_groups = select_groups(valid.num_row_groups, args.valid_groups, args.seed + 1)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_with_groups(args, train_groups, valid_groups, variants,
                    extra_features_by_variant)


if __name__ == "__main__":
    main()
