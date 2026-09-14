r"""Cinco pares GRU base vs top 3 + i0_v21_delta_5 en grupos nuevos.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_five_group_pairs.py

Entrena exactamente 10 modelos: dos por ronda, con 100 secuencias de train
distintas entre rondas y los mismos 50 grupos nuevos de validación. Excluye
los grupos registrados por experimentos anteriores antes de seleccionar.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from features import ROOT, TemporalFeature
from train_gru import DEFAULT_RANKING, HERE, run_with_groups
from utils import SEQUENCE_LENGTH


ROUNDS = 5
TRAIN_GROUPS_PER_ROUND = 100
VALID_GROUPS = 50
VARIANTS = (("base", 0), ("top3_plus_v21_delta5", 3))
EXTRAS = {"top3_plus_v21_delta5": (TemporalFeature("i0_v21", "delta", 5),)}
SUMMARY_FIELDS = (
    "round", "train_groups", "valid_groups", "base_t0", "base_t1", "base_mean",
    "augmented_t0", "augmented_t1", "augmented_mean", "delta_t0", "delta_t1",
    "delta_mean", "eligible_blocks",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    result.add_argument("--train-path", type=Path, default=ROOT / "datasets" / "train.parquet")
    result.add_argument("--valid-path", type=Path, default=ROOT / "datasets" / "valid.parquet")
    result.add_argument("--output-dir", type=Path, default=HERE / "results" / "group_repeats")
    result.add_argument("--selection-seed", type=int, default=2027)
    result.add_argument("--model-seed", type=int, default=2026)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--chunk-size", type=int, default=128)
    result.add_argument("--resume", type=Path, default=None,
                        help="Retomar una corrida existente sin repetir modelos completos")
    return result


def _recorded_indices(values, total: int, source: Path, field: str) -> set[int]:
    if not isinstance(values, list):
        raise ValueError(f"{source}: {field} debe ser una lista")
    result = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < total:
            raise ValueError(f"{source}: índice inválido en {field}: {value}")
        result.add(value)
    return result


def previously_used_groups(
    train_total: int, valid_total: int
) -> tuple[set[int], set[int], list[str]]:
    """Lee todos los manifiestos de experimentos previos, sin modificar datos."""
    used_train: set[int] = set()
    used_valid: set[int] = set()
    sources = []
    for path in sorted((ROOT / "my_solution").rglob("run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "train_row_groups" not in config and "validation_row_groups" not in config:
            continue
        used_train.update(_recorded_indices(
            config.get("train_row_groups", []), train_total, path, "train_row_groups"
        ))
        used_valid.update(_recorded_indices(
            config.get("validation_row_groups", []), valid_total, path, "validation_row_groups"
        ))
        sources.append(str(path.relative_to(ROOT)))
    ranking_metadata = ROOT / "my_solution" / "explore" / "feature_select" / "results" / "selection_metadata.json"
    if ranking_metadata.exists():
        metadata = json.loads(ranking_metadata.read_text(encoding="utf-8"))
        used_train.update(_recorded_indices(
            metadata["row_groups_selected"], train_total, ranking_metadata, "row_groups_selected"
        ))
        sources.append(str(ranking_metadata.relative_to(ROOT)))
    return used_train, used_valid, sources


def choose_new_groups(
    train_total: int, valid_total: int, used_train: set[int], used_valid: set[int],
    selection_seed: int, rounds: int = ROUNDS,
    train_per_round: int = TRAIN_GROUPS_PER_ROUND,
    valid_count: int = VALID_GROUPS,
) -> tuple[list[list[int]], list[int]]:
    if rounds < 1 or train_per_round < 1 or valid_count < 1:
        raise ValueError("Las cantidades de grupos deben ser positivas")
    train_available = sorted(set(range(train_total)) - used_train)
    valid_available = sorted(set(range(valid_total)) - used_valid)
    train_needed = rounds * train_per_round
    if len(train_available) < train_needed or len(valid_available) < valid_count:
        raise ValueError("No quedan suficientes grupos nuevos de train o validación")
    train_chosen = np.random.default_rng(selection_seed).choice(
        train_available, size=train_needed, replace=False
    ).tolist()
    valid_groups = sorted(np.random.default_rng(selection_seed + 1).choice(
        valid_available, size=valid_count, replace=False
    ).tolist())
    train_rounds = [
        sorted(train_chosen[start:start + train_per_round])
        for start in range(0, train_needed, train_per_round)
    ]
    flat = [group for round_groups in train_rounds for group in round_groups]
    if (len(flat) != len(set(flat)) or used_train.intersection(flat)
            or len(valid_groups) != len(set(valid_groups))
            or used_valid.intersection(valid_groups)):
        raise AssertionError("La selección reutilizó grupos")
    return train_rounds, valid_groups


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.chunk_size < 1:
        raise SystemExit("--epochs y --chunk-size deben ser positivos")
    train_total = pq.ParquetFile(args.train_path).num_row_groups
    valid_total = pq.ParquetFile(args.valid_path).num_row_groups
    if args.resume is not None:
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        expected = {
            "train_dataset": str(args.train_path.resolve()),
            "validation_dataset": str(args.valid_path.resolve()),
            "ranking": str(args.ranking.resolve()),
            "selection_seed": args.selection_seed, "model_seed": args.model_seed,
            "epochs": args.epochs, "chunk_size": args.chunk_size,
            "train_groups_per_round": TRAIN_GROUPS_PER_ROUND,
            "validation_groups_per_round": VALID_GROUPS,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("El manifiesto no coincide con los argumentos de la corrida")
        train_rounds = [round_data["train_row_groups"] for round_data in manifest["rounds"]]
        valid_groups = manifest["validation_row_groups"]
        flat = [group for groups in train_rounds for group in groups]
        if (len(train_rounds) != ROUNDS or any(len(groups) != TRAIN_GROUPS_PER_ROUND for groups in train_rounds)
                or len(flat) != len(set(flat)) or any(not 0 <= group < train_total for group in flat)
                or len(valid_groups) != VALID_GROUPS or len(valid_groups) != len(set(valid_groups))
                or any(not 0 <= group < valid_total for group in valid_groups)
                or any(round_data["validation_row_groups"] != valid_groups for round_data in manifest["rounds"])):
            raise ValueError("El manifiesto contiene grupos incompletos o repetidos")
        print(f"Retomando corrida: {run_dir}", flush=True)
    else:
        used_train, used_valid, sources = previously_used_groups(train_total, valid_total)
        train_rounds, valid_groups = choose_new_groups(
            train_total, valid_total, used_train, used_valid, args.selection_seed
        )
        run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "train_dataset": str(args.train_path.resolve()),
            "validation_dataset": str(args.valid_path.resolve()),
            "ranking": str(args.ranking.resolve()),
            "selection_seed": args.selection_seed, "model_seed": args.model_seed,
            "epochs": args.epochs, "chunk_size": args.chunk_size,
            "sequence_length": SEQUENCE_LENGTH,
            "train_groups_per_round": TRAIN_GROUPS_PER_ROUND,
            "validation_groups_per_round": VALID_GROUPS,
            "train_row_groups": [group for groups in train_rounds for group in groups],
            "validation_row_groups": valid_groups,
            "rounds": [
                {"round": index, "train_row_groups": groups, "validation_row_groups": valid_groups}
                for index, groups in enumerate(train_rounds, start=1)
            ],
            "variants": ["base", "top3_plus_v21_delta5"],
            "previous_train_groups_excluded": len(used_train),
            "previous_validation_groups_excluded": len(used_valid),
            "previous_group_sources": sources,
        }
        # Se escribe antes de entrenar: aun ante una interrupción, estos grupos
        # quedarán excluidos de futuras selecciones.
        (run_dir / "run_config.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Grupos previos excluidos: train={len(used_train)}, valid={len(used_valid)}")
        print("Cinco rondas con 100 grupos nuevos cada una; validacion fija de 50 grupos nuevos.")
    print(f"Manifiesto: {(run_dir / 'run_config.json').resolve()}", flush=True)

    rows = []
    for round_number, train_groups in enumerate(train_rounds, start=1):
        print(f"\n=== Ronda {round_number}/{ROUNDS} ===", flush=True)
        training_args = argparse.Namespace(
            ranking=args.ranking, train_path=args.train_path, valid_path=args.valid_path,
            output_dir=args.output_dir, seed=args.model_seed,
            epochs=args.epochs, chunk_size=args.chunk_size,
        )
        comparisons = run_with_groups(
            training_args, train_groups, valid_groups,
            variants=VARIANTS, extra_features_by_variant=EXTRAS,
            run_dir=run_dir / f"round_{round_number:02d}", model_seed=args.model_seed,
            resume=args.resume is not None,
        )
        by_name = {row["variant"]: row for row in comparisons}
        base = by_name["base"]
        augmented = by_name["top3_plus_v21_delta5"]
        row = {
            "round": round_number, "train_groups": len(train_groups),
            "valid_groups": len(valid_groups),
            "base_t0": base["wp_t0"], "base_t1": base["wp_t1"], "base_mean": base["wp_mean"],
            "augmented_t0": augmented["wp_t0"], "augmented_t1": augmented["wp_t1"],
            "augmented_mean": augmented["wp_mean"],
            "delta_t0": augmented["wp_t0"] - base["wp_t0"],
            "delta_t1": augmented["wp_t1"] - base["wp_t1"],
            "delta_mean": augmented["wp_mean"] - base["wp_mean"],
            "eligible_blocks": augmented["eligible_blocks"],
        }
        rows.append(row)
        write_summary(run_dir / "summary.csv", rows)
        print(f"Ronda {round_number}: delta WP promedio={row['delta_mean']:+.6f}", flush=True)

    aggregate = {}
    for suffix in ("t0", "t1", "mean"):
        deltas = np.array([row[f"delta_{suffix}"] for row in rows], dtype=np.float64)
        aggregate[suffix] = {
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "min_delta": float(deltas.min()), "max_delta": float(deltas.max()),
            "rounds_improved": int(np.count_nonzero(deltas > 0)), "rounds": len(rows),
        }
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResumen: {(run_dir / 'summary.csv').resolve()}")
    print(f"Agregado: {(run_dir / 'aggregate.json').resolve()}")


if __name__ == "__main__":
    main()
