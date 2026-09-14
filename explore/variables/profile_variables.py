r"""Generate a YData Profiling HTML report for 112 inputs and t0/t1.

From the project root (PowerShell):

    .\.venv\Scripts\python.exe my_solution\explore\variables\profile_variables.py --row-groups 10

    python my_solution\explore\variables\profile_variables.py --row-groups 300 --max-rows 2000000 --pearson

Only the selected row groups are read. By default, at most 20,000 rows total
are sampled in a balanced way across them; use --max-rows to change this.
The lightweight YData profile is used by default; --pearson adds a Pearson
correlation matrix without enabling all the expensive interaction analyses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS  # noqa: E402


DEFAULT_DATASET = ROOT / "datasets" / "train.parquet"
DEFAULT_RESULTS = Path(__file__).with_name("results")
PROFILE_COLUMNS = [*FEATURE_COLUMNS, *TARGET_COLUMNS]


def select_row_groups(total: int, requested: int, seed: int, first: bool) -> list[int]:
    if requested < 1 or requested > total:
        raise ValueError(f"--row-groups debe estar entre 1 y {total}")
    if first:
        return list(range(requested))
    return sorted(
        np.random.default_rng(seed).choice(total, size=requested, replace=False).tolist()
    )


def sample_counts(groups: int, max_rows: int) -> list[int]:
    """Allocate an equal number of rows per group, keeping at least one."""
    if max_rows < groups:
        raise ValueError("--max-rows debe ser al menos la cantidad de row groups")
    budget = min(max_rows, groups * SEQUENCE_LENGTH)
    base, remainder = divmod(budget, groups)
    return [base + int(position < remainder) for position in range(groups)]


def load_sample(
    parquet: pq.ParquetFile,
    selected_groups: list[int],
    counts: list[int],
    seed: int,
):
    import pandas as pd

    batches = []
    nonfinite = np.zeros(len(PROFILE_COLUMNS), dtype=np.int64)
    sampled_by_group = []

    for position, (row_group, requested_rows) in enumerate(zip(selected_groups, counts), start=1):
        table = parquet.read_row_group(row_group, columns=PROFILE_COLUMNS, use_threads=False)
        rows = table.num_rows
        if rows != SEQUENCE_LENGTH:
            raise ValueError(
                f"row group {row_group} tiene {rows} filas; se esperaban {SEQUENCE_LENGTH}"
            )
        rng = np.random.default_rng(seed + row_group + 1)
        indices = (
            np.arange(rows)
            if requested_rows >= rows
            else np.sort(rng.choice(rows, size=requested_rows, replace=False))
        )
        sampled = table.take(indices)
        # Keep the official 112-feature order, followed by t0 and t1. Convert non-finite
        # values to NaN so the report can treat them as missing data.
        matrix = np.column_stack([sampled[name].to_numpy() for name in PROFILE_COLUMNS])
        invalid = ~np.isfinite(matrix)
        nonfinite += invalid.sum(axis=0)
        if invalid.any():
            matrix = matrix.astype(np.float32, copy=True)
            matrix[invalid] = np.nan
        batches.append(pd.DataFrame(matrix, columns=PROFILE_COLUMNS))
        sampled_by_group.append({"row_group": row_group, "rows_sampled": len(indices)})
        if position % 25 == 0 or position == len(selected_groups):
            print(f"Leídos {position}/{len(selected_groups)} row groups", flush=True)

    frame = pd.concat(batches, ignore_index=True)
    return frame, sampled_by_group, dict(zip(PROFILE_COLUMNS, nonfinite.tolist()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crear un HTML de ydata-profiling para 112 variables y los targets t0/t1."
    )
    parser.add_argument(
        "--row-groups", type=int, required=True,
        help="Cantidad de row groups a leer (selección aleatoria reproducible).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=20_000,
        help="Máximo de filas muestreadas entre todos los grupos (default: 20000).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--first", action="store_true", help="Usar los primeros N grupos en vez de una muestra aleatoria."
    )
    parser.add_argument(
        "--pearson", action="store_true",
        help="Añadir correlaciones Pearson sin activar las interacciones costosas.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from ydata_profiling import ProfileReport
    except ModuleNotFoundError as exc:
        if exc.name == "ydata_profiling":
            raise SystemExit(
                "Falta ydata-profiling. Instálalo en el entorno activo con: "
                "python -m pip install ydata-profiling"
            ) from exc
        raise SystemExit(
            f"ydata-profiling está instalado, pero falta una dependencia: {exc.name}. "
            "Comprueba las dependencias del entorno."
        ) from exc

    parquet = pq.ParquetFile(args.dataset)
    missing = set(PROFILE_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise SystemExit(f"Faltan variables en el dataset: {sorted(missing)}")
    try:
        groups = select_row_groups(parquet.num_row_groups, args.row_groups, args.seed, args.first)
        counts = sample_counts(len(groups), args.max_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    frame, sampled_by_group, nonfinite = load_sample(parquet, groups, counts, args.seed)
    print(f"Generando perfil de {len(frame):,} filas x {len(frame.columns)} variables...", flush=True)
    report = ProfileReport(
        frame,
        title=f"Variables del connectome · {len(groups)} row groups · {len(frame):,} filas",
        minimal=True,
        correlations={"pearson": {"calculate": True}} if args.pearson else None,
        interactions=None,
        progress_bar=False,
    )

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = args.results_dir / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    html_path = output_dir / "variables_profile.html"
    report.to_file(html_path)
    metadata = {
        "dataset": str(args.dataset.resolve()),
        "profile_mode": "minimal_plus_pearson" if args.pearson else "minimal",
        "selection": "first" if args.first else "random_without_replacement",
        "seed": args.seed,
        "row_groups_requested": args.row_groups,
        "row_groups_available": parquet.num_row_groups,
        "row_groups_selected": groups,
        "max_rows_requested": args.max_rows,
        "rows_profiled": len(frame),
        "columns_profiled": list(frame.columns),
        "nonfinite_values_converted_to_nan_in_sample": nonfinite,
        "sampled_by_group": sampled_by_group,
        "note": "Statistics describe the sampled rows, not all rows in the selected row groups.",
    }
    (output_dir / "sample_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"HTML: {html_path.resolve()}")
    print(f"Muestra: {(output_dir / 'sample_metadata.json').resolve()}")


if __name__ == "__main__":
    main()
