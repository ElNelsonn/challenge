r"""Correlaciones de variables originales, lags y deltas con t0/t1.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\explore\feature_select\temporal_correlations.py --row-groups 100

Para tomar los primeros 100 grupos en vez de una muestra aleatoria:

    .\.venv\Scripts\python.exe my_solution\explore\feature_select\temporal_correlations.py --row-groups 100 --first

Cada row group de train es una secuencia completa de 20.000 pasos. Se leen
los targets una vez y una variable de entrada por vez. Spearman se calcula
exactamente sobre las filas válidas de los grupos seleccionados, no como
promedio de correlaciones por bloque. El parquet original no se modifica.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS  # noqa: E402


LAGS = (1, 5, 10, 50)
FIELDS = ("feature", "base_feature", "transform", "lag", "target",
          "pearson", "spearman", "n_valid")
DEFAULT_DATASET = ROOT / "datasets" / "train.parquet"
DEFAULT_RESULTS = Path(__file__).with_name("results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--row-groups", type=int, default=100,
                        help="Secuencias completas a analizar; 100 equivalen a 2 millones de filas")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--first", action="store_true",
                        help="Tomar los primeros grupos en lugar de muestrear")
    return parser


def select_row_groups(total: int, requested: int, seed: int, first: bool) -> list[int]:
    if not 1 <= requested <= total:
        raise ValueError(f"--row-groups debe estar entre 1 y {total}")
    if first:
        return list(range(requested))
    return sorted(np.random.default_rng(seed).choice(total, requested, replace=False).tolist())


def lag_within_sequences(values: np.ndarray, lag: int, sequence_length: int) -> np.ndarray:
    """Deja NaN en las primeras `lag` filas de cada secuencia."""
    if lag < 1 or len(values) % sequence_length:
        raise ValueError("Lag inválido o longitud incompatible con secuencias completas")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for start in range(0, len(values), sequence_length):
        result[start + lag:start + sequence_length] = values[start:start + sequence_length - lag]
    return result


def pearson_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.clip(np.dot(xc, yc) / denominator, -1.0, 1.0))


def correlations(values: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(values) & np.isfinite(target)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        return float("nan"), float("nan"), n_valid
    x, y = values[valid], target[valid]
    pearson = pearson_from_pairs(x, y)
    if not np.isfinite(pearson):
        return pearson, float("nan"), n_valid
    spearman = pearson_from_pairs(rankdata(x), rankdata(y))
    return pearson, spearman, n_valid


def score(row: dict) -> float:
    finite = [abs(value) for value in (row["pearson"], row["spearman"])
              if np.isfinite(value)]
    return max(finite, default=float("-inf"))


def main() -> None:
    args = build_parser().parse_args()
    parquet = pq.ParquetFile(args.dataset)
    expected = {"seq_ix", "step_in_seq", *FEATURE_COLUMNS, *TARGET_COLUMNS}
    missing = expected - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"Faltan columnas esperadas en train: {sorted(missing)}")
    try:
        groups = select_row_groups(parquet.num_row_groups, args.row_groups, args.seed, args.first)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    n_rows = len(groups) * SEQUENCE_LENGTH
    targets = {name: np.empty(n_rows, dtype=np.float64) for name in TARGET_COLUMNS}
    seen_sequences = set()
    for position, group in enumerate(groups):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "step_in_seq", *TARGET_COLUMNS], use_threads=False
        )
        if table.num_rows != SEQUENCE_LENGTH:
            raise SystemExit(f"Row group {group}: se esperaban {SEQUENCE_LENGTH} filas")
        seq = table["seq_ix"].to_numpy(zero_copy_only=False)
        steps = table["step_in_seq"].to_numpy(zero_copy_only=False)
        if not np.all(seq == seq[0]):
            raise SystemExit(f"Row group {group}: mezcla secuencias")
        if not np.array_equal(steps, np.arange(SEQUENCE_LENGTH)):
            raise SystemExit(f"Row group {group}: step_in_seq no está en orden 0..{SEQUENCE_LENGTH - 1}")
        seq_id = int(seq[0])
        if seq_id in seen_sequences:
            raise SystemExit(f"Secuencia duplicada: {seq_id}")
        seen_sequences.add(seq_id)
        sl = slice(position * SEQUENCE_LENGTH, (position + 1) * SEQUENCE_LENGTH)
        for name in TARGET_COLUMNS:
            targets[name][sl] = table[name].to_numpy(zero_copy_only=False)
    print(f"Analizando {len(groups)} secuencias completas ({n_rows:,} filas)", flush=True)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.results_dir / "temporal_correlations.csv"
    rows: list[dict] = []
    with full_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for feature_number, base in enumerate(FEATURE_COLUMNS, start=1):
            x = np.empty(n_rows, dtype=np.float64)
            for position, group in enumerate(groups):
                sl = slice(position * SEQUENCE_LENGTH, (position + 1) * SEQUENCE_LENGTH)
                x[sl] = parquet.read_row_group(
                    group, columns=[base], use_threads=False
                )[base].to_numpy(zero_copy_only=False)

            def record(feature: str, transform: str, lag: int, values: np.ndarray) -> None:
                for target_name, target_values in targets.items():
                    pearson, spearman, n_valid = correlations(values, target_values)
                    row = {"feature": feature, "base_feature": base, "transform": transform,
                           "lag": lag, "target": target_name, "pearson": pearson,
                           "spearman": spearman, "n_valid": n_valid}
                    writer.writerow(row)
                    rows.append(row)

            record(base, "original", 0, x)
            for lag in LAGS:
                past = lag_within_sequences(x, lag, SEQUENCE_LENGTH)
                record(f"{base}_lag_{lag}", "lag", lag, past)
                record(f"{base}_delta_{lag}", "delta", lag, x - past)
            print(f"Variable {feature_number}/{len(FEATURE_COLUMNS)}: {base}", flush=True)

    top_path = args.results_dir / "top_20_by_target.csv"
    with top_path.open("w", newline="", encoding="utf-8") as top_stream:
        top_writer = csv.DictWriter(top_stream, fieldnames=("rank", *FIELDS))
        top_writer.writeheader()
        for target_name in TARGET_COLUMNS:
            ranked = sorted((row for row in rows
                             if row["target"] == target_name
                             and row["transform"] in ("lag", "delta")),
                            key=score, reverse=True)
            ranked_path = args.results_dir / f"temporal_correlations_{target_name}_ranked.csv"
            with ranked_path.open("w", newline="", encoding="utf-8") as ranked_stream:
                writer = csv.DictWriter(ranked_stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(ranked)
            print(f"\nTop 20 para {target_name}:")
            for rank, row in enumerate(ranked[:20], start=1):
                top_writer.writerow({"rank": rank, **row})
                print(f"{rank:2}. {row['feature']:<26} Pearson={row['pearson']:.5f} "
                      f"Spearman={row['spearman']:.5f} n_valid={row['n_valid']}")

    metadata = {"dataset": str(args.dataset.resolve()), "row_groups_selected": groups,
                "selection": "first" if args.first else "random_without_replacement",
                "seed": args.seed, "sequences": len(groups), "rows": n_rows,
                "inputs": list(FEATURE_COLUMNS), "targets": list(TARGET_COLUMNS),
                "lags": list(LAGS), "correlation_scope": "selected complete sequences"}
    (args.results_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCSV completo: {full_path}\nTop 20: {top_path}")


if __name__ == "__main__":
    main()
