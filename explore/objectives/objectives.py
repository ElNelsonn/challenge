r"""Explore the anonymized targets without loading the full dataset into RAM.

STARTUP COMMANDS (PowerShell, run from the project root)
========================================================

Create the Python 3.11 environment (only the first time):

    py -3.11 -m venv .venv

Install the dependencies (only the first time):

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Quick test with 10 sequences:

    .\.venv\Scripts\python.exe my_solution\explore\objectives\objectives.py --max-sequences 10

Analyze the complete training dataset:

    .\.venv\Scripts\python.exe my_solution\explore\objectives\objectives.py

The JSON report is saved in:

    my_solution\explore\objectives\results\objectives_report.json

Open the visual report in:

    my_solution\explore\objectives\report.html

Exact, streaming calculations:
  * count, mean, standard deviation, min and max
  * percentage close to zero and outside the metric interval [-2, 2]
  * Pearson correlation between t0 and t1
  * within-sequence autocorrelation at configurable lags

Approximate, stratified-sample calculations:
  * percentiles
  * IQR and z-score outlier rates

By default warm-up rows are excluded because no prediction is required for
them. Autocorrelation pairs never cross sequence boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets" / "train.parquet"
DEFAULT_REPORT = Path(__file__).with_name("results") / "objectives_report.json"
TARGETS = ("t0", "t1")


@dataclass
class RunningStats:
    """Numerically stable univariate moments, merged one array at a time."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_count = int(values.size)
        batch_mean = float(values.mean(dtype=np.float64))
        batch_m2 = float(np.square(values - batch_mean).sum(dtype=np.float64))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else math.nan


@dataclass
class RunningCorrelation:
    """Numerically stable bivariate moments for Pearson correlation."""

    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    covariance_sum: float = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size != y.size:
            raise ValueError("correlation arrays must have equal length")
        if x.size == 0:
            return

        batch_count = int(x.size)
        batch_mean_x = float(x.mean(dtype=np.float64))
        batch_mean_y = float(y.mean(dtype=np.float64))
        centered_x = x - batch_mean_x
        centered_y = y - batch_mean_y
        batch_m2_x = float(np.dot(centered_x, centered_x))
        batch_m2_y = float(np.dot(centered_y, centered_y))
        batch_covariance = float(np.dot(centered_x, centered_y))

        delta_x = batch_mean_x - self.mean_x
        delta_y = batch_mean_y - self.mean_y
        total = self.count + batch_count
        correction = self.count * batch_count / total

        self.m2_x += batch_m2_x + delta_x * delta_x * correction
        self.m2_y += batch_m2_y + delta_y * delta_y * correction
        self.covariance_sum += batch_covariance + delta_x * delta_y * correction
        self.mean_x += delta_x * batch_count / total
        self.mean_y += delta_y * batch_count / total
        self.count = total

    def value(self) -> float:
        denominator = math.sqrt(self.m2_x * self.m2_y)
        return self.covariance_sum / denominator if denominator > 0 else math.nan


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("provide non-negative comma-separated values")
    return values


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("provide positive comma-separated integers")
    return values


def finite(values: np.ndarray, name: str, group: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values in row group {group}")
    return values


def sampled_values(
    values: np.ndarray,
    eligible_indices: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if eligible_indices.size <= sample_count:
        return values[eligible_indices]
    chosen = rng.choice(eligible_indices, size=sample_count, replace=False)
    return values[chosen]


def percentile_report(sample: np.ndarray, percentiles: Iterable[float]) -> dict:
    percentiles = tuple(percentiles)
    quantiles = np.percentile(sample, percentiles)
    q1, q3 = np.percentile(sample, [25, 75])
    iqr = float(q3 - q1)
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    mean = float(sample.mean(dtype=np.float64))
    std = float(sample.std(dtype=np.float64))
    return {
        "percentiles_approx": {
            str(percentile): float(value)
            for percentile, value in zip(percentiles, quantiles)
        },
        "sample_iqr": iqr,
        "sample_iqr_outlier_limits": [lower, upper],
        "sample_iqr_outlier_pct": float(100 * np.mean((sample < lower) | (sample > upper))),
        "sample_abs_z_gt_3_pct": (
            float(100 * np.mean(np.abs(sample - mean) > 3 * std)) if std > 0 else 0.0
        ),
    }


def analyze(args: argparse.Namespace) -> dict:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'pyarrow'. Install the project requirements with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    parquet = pq.ParquetFile(args.dataset)
    available = set(parquet.schema_arrow.names)
    required = {"need_prediction", *TARGETS}
    if args.scored_only:
        required.add("is_scored")
    missing = required - available
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    total_groups = parquet.num_row_groups
    groups_to_read = total_groups if args.max_sequences is None else min(
        args.max_sequences, total_groups
    )
    if groups_to_read <= 0:
        raise ValueError("no sequences selected")

    stats = {target: RunningStats() for target in TARGETS}
    cross_target = RunningCorrelation()
    autocorrelation = {
        target: {lag: RunningCorrelation() for lag in args.lags} for target in TARGETS
    }
    near_zero = {
        target: {threshold: 0 for threshold in args.near_zero} for target in TARGETS
    }
    outside_clip = {target: 0 for target in TARGETS}

    # Every row group is one equally-sized sequence, so taking the same number
    # from each gives a reproducible stratified approximation of percentiles.
    samples_per_group = max(1, math.ceil(args.sample_size / groups_to_read))
    samples: dict[str, list[np.ndarray]] = {target: [] for target in TARGETS}
    rng = np.random.default_rng(args.seed)
    selected_rows = 0

    columns = ["need_prediction", *TARGETS]
    if args.scored_only:
        columns.append("is_scored")

    for group in range(groups_to_read):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        mask = table["need_prediction"].to_numpy().astype(bool)
        if args.include_warmup:
            mask[:] = True
        if args.scored_only:
            mask &= table["is_scored"].to_numpy().astype(bool)

        values = {
            target: finite(table[target].to_numpy(), target, group) for target in TARGETS
        }
        chosen = np.flatnonzero(mask)
        selected_rows += int(chosen.size)

        for target in TARGETS:
            selected = values[target][mask]
            stats[target].add(selected)
            outside_clip[target] += int(np.count_nonzero(np.abs(selected) > 2.0))
            for threshold in args.near_zero:
                near_zero[target][threshold] += int(
                    np.count_nonzero(np.abs(selected) <= threshold)
                )
            samples[target].append(
                sampled_values(values[target], chosen, samples_per_group, rng)
            )

        cross_target.add(values["t0"][mask], values["t1"][mask])

        # Pair t with t-lag only when both original rows are selected. This
        # preserves the meaning of the lag even with a discontinuous score mask.
        for lag in args.lags:
            if lag >= table.num_rows:
                continue
            pair_mask = mask[lag:] & mask[:-lag]
            for target in TARGETS:
                autocorrelation[target][lag].add(
                    values[target][:-lag][pair_mask],
                    values[target][lag:][pair_mask],
                )

        if args.progress_every and (group + 1) % args.progress_every == 0:
            print(f"Processed {group + 1:,}/{groups_to_read:,} sequences", flush=True)

    report = {
        "dataset": str(args.dataset.resolve()),
        "sequences_analyzed": groups_to_read,
        "sequences_available": total_groups,
        "selected_rows": selected_rows,
        "selection": {
            "include_warmup": args.include_warmup,
            "scored_only": args.scored_only,
        },
        "notes": {
            "exact": "moments, ranges, threshold rates and correlations",
            "approximate": "percentiles and outlier rates use a stratified sample",
            "correlation_type": "ordinary unweighted Pearson (not competition WP)",
        },
        "corr_t0_t1": cross_target.value(),
        "targets": {},
    }

    for target in TARGETS:
        sample = np.concatenate(samples[target])
        # A final cap handles rounding from samples_per_group.
        if sample.size > args.sample_size:
            sample = sample[rng.choice(sample.size, args.sample_size, replace=False)]
        target_stats = stats[target]
        report["targets"][target] = {
            "count": target_stats.count,
            "mean": target_stats.mean,
            "std_population": target_stats.std,
            "min": target_stats.minimum,
            "max": target_stats.maximum,
            "near_zero_pct": {
                str(threshold): 100 * count / target_stats.count
                for threshold, count in near_zero[target].items()
            },
            "outside_minus2_plus2_pct": 100 * outside_clip[target] / target_stats.count,
            "autocorrelation": {
                str(lag): autocorrelation[target][lag].value() for lag in args.lags
            },
            "autocorrelation_pair_counts": {
                str(lag): autocorrelation[target][lag].count for lag in args.lags
            },
            "sample_size": int(sample.size),
            **percentile_report(sample, args.percentiles),
        }

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze t0/t1 distributions and temporal correlation in streaming mode."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Analyze only the first N row groups; omit for the complete dataset.",
    )
    parser.add_argument("--lags", type=parse_int_list, default=(1, 5, 10))
    parser.add_argument(
        "--near-zero", type=parse_float_list, default=(0.01, 0.05, 0.1)
    )
    parser.add_argument(
        "--percentiles",
        type=parse_float_list,
        default=(0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100),
    )
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include steps 0-98; excluded by default.",
    )
    parser.add_argument(
        "--scored-only",
        action="store_true",
        help="Use only is_scored rows (valid.parquet only).",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(report_json, encoding="utf-8")

    # report.html loads this JavaScript file directly. Unlike fetch(), a local
    # script works when the HTML is opened with a double click (file://).
    browser_data = report_json.replace("<", "\\u003c").replace("\u2028", "\\u2028")
    browser_data_path = args.output.with_suffix(".js")
    browser_data_path.write_text(
        f"window.OBJECTIVES_REPORT = {browser_data};\n", encoding="utf-8"
    )

    print(report_json)
    print(f"\nJSON report saved to {args.output.resolve()}")
    print(f"Browser data saved to {browser_data_path.resolve()}")
    print(f"Open {(Path(__file__).parent / 'report.html').resolve()} to view it.")


if __name__ == "__main__":
    main()
