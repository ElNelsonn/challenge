r"""Plot t0 and t1 for selected Parquet row groups.

Run from the project root (PowerShell):

    .\.venv\Scripts\python.exe -m pip install matplotlib
    .\.venv\Scripts\python.exe my_solution\explore\objectives\plot_targets.py "12-15, 20, 31-33"

Ranges are inclusive and row-group indices start at 0. For train.parquet,
valid indices are 0 through 10606. One PNG is saved per selected row group in
my_solution/explore/objectives/results/plots/. Only the required columns and
row groups are read; the full dataset is never loaded into memory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "train.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results") / "plots"
SELECTOR_PATTERN = re.compile(r"(\d+)(?:\s*-\s*(\d+))?\Z")


def parse_row_groups(specification: str, available: int) -> list[int]:
    """Expand comma-separated inclusive ranges, removing repeated indices."""
    if not specification.strip():
        raise ValueError("indica al menos un row group")

    chosen: list[int] = []
    seen: set[int] = set()
    for raw_item in specification.split(","):
        item = raw_item.strip()
        match = SELECTOR_PATTERN.fullmatch(item)
        if match is None:
            raise ValueError(
                f"selección inválida {item!r}; usa, por ejemplo, '12-15,20,31'"
            )

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if end < start:
            raise ValueError(f"rango descendente {item!r}: el final debe ser >= al inicio")
        if end >= available:
            raise ValueError(
                f"{item!r} queda fuera del dataset: índices válidos 0-{available - 1}"
            )

        for index in range(start, end + 1):
            if index not in seen:
                seen.add(index)
                chosen.append(index)
    return chosen


def read_targets(parquet: pq.ParquetFile, row_group: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    table = parquet.read_row_group(
        row_group,
        columns=["seq_ix", "step_in_seq", "t0", "t1"],
        use_threads=False,
    )
    sequence_ids = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    t0 = table["t0"].to_numpy()
    t1 = table["t1"].to_numpy()

    if table.num_rows == 0 or not np.all(sequence_ids == sequence_ids[0]):
        raise ValueError(f"row group {row_group}: no contiene una sola secuencia")
    if not np.array_equal(steps, np.arange(table.num_rows)):
        raise ValueError(f"row group {row_group}: step_in_seq no es consecutivo desde 0")
    if not np.isfinite(t0).all() or not np.isfinite(t1).all():
        raise ValueError(f"row group {row_group}: t0/t1 contiene valores no finitos")

    return int(sequence_ids[0]), steps, t0, t1


def draw_row_group(
    row_group: int,
    sequence_id: int,
    steps: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    dataset_name: str,
    output_path: Path,
) -> None:
    # Import after selecting the non-interactive backend so Windows can save
    # plots even when no desktop window is available.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle(
        f"Targets t0 y t1 · {dataset_name} · row group {row_group} · secuencia {sequence_id}",
        fontsize=14,
    )

    for axis, values, label, color in zip(
        axes, (t0, t1), ("t0", "t1"), ("#2563eb", "#e11d48")
    ):
        axis.plot(steps, values, color=color, linewidth=0.7, alpha=0.85)
        axis.axhline(0, color="#334155", linewidth=0.8)
        axis.axhline(2, color="#94a3b8", linewidth=0.7, linestyle="--")
        axis.axhline(-2, color="#94a3b8", linewidth=0.7, linestyle="--")
        axis.axvspan(0, 98, color="#f59e0b", alpha=0.12)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.margins(x=0)

    axes[-1].set_xlabel("Paso dentro de la secuencia (step_in_seq)")
    fig.text(
        0.5,
        0.015,
        "Zona amarilla: calentamiento (pasos 0–98). Líneas discontinuas: límites ±2 usados solo en el cálculo del puntaje.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    try:
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grafica t0 y t1 por row group; acepta rangos inclusivos e índices separados por comas."
    )
    parser.add_argument(
        "row_groups",
        help='Ejemplo: "12-15,20,31-33" (índices desde cero).',
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401 - checked before processing the dataset
    except ModuleNotFoundError as exc:
        parser.error(
            "falta matplotlib; instálalo en este entorno con: "
            "python -m pip install matplotlib"
        )

    parquet = pq.ParquetFile(args.dataset)
    required = {"seq_ix", "step_in_seq", "t0", "t1"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        parser.error(f"faltan columnas en el dataset: {', '.join(sorted(missing))}")
    try:
        selected = parse_row_groups(args.row_groups, parquet.num_row_groups)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for position, row_group in enumerate(selected, start=1):
        sequence_id, steps, t0, t1 = read_targets(parquet, row_group)
        output_path = args.output_dir / (
            f"{args.dataset.stem}_rowgroup_{row_group:05d}_seq_{sequence_id}.png"
        )
        draw_row_group(
            row_group, sequence_id, steps, t0, t1, args.dataset.name, output_path
        )
        print(f"[{position}/{len(selected)}] {output_path}", flush=True)


if __name__ == "__main__":
    main()
