"""Shared data loading and challenge-aligned metrics for the FirstTry scripts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    BlockAccumulator,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    WARMUP,
    validate_sequence,
    weighted_pearson,
)

TRAIN_PATH = ROOT / "datasets" / "train.parquet"
VALID_PATH = ROOT / "datasets" / "valid.parquet"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-groups", type=int, default=3,
                        help="Grupos de train: 1 a 10607 en este dataset (default: 3).")
    parser.add_argument("--valid-groups", type=int, default=1,
                        help="Grupos de validación: 1 a 1873 en este dataset (default: 1).")
    parser.add_argument("--rows-per-group", type=int, default=1000,
                        help="Filas por grupo: 1 a 19901 efectivas; ajuste tabular o normalización secuencial (default: 1000).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Semilla para seleccionar grupos y filas (default: 2026).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Carpeta base de resultados (default: results dentro de la carpeta del modelo).")
    return parser


def select_groups(total: int, count: int, seed: int) -> list[int]:
    if not 1 <= count <= total:
        raise ValueError(f"La cantidad de row groups debe estar entre 1 y {total}")
    return sorted(np.random.default_rng(seed).choice(total, count, replace=False).tolist())


def read_group(parquet: pq.ParquetFile, index: int, validation: bool = False):
    columns = ["seq_ix", "step_in_seq", "need_prediction"]
    if validation:
        columns.append("is_scored")
    columns.extend(FEATURE_COLUMNS)
    columns.extend(TARGET_COLUMNS)
    table = parquet.read_row_group(index, columns=columns, use_threads=False)
    seq_ix, need = validate_sequence(table)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError(f"Targets no finitos en row group {index}")
    # Inputs are expected to be finite, but use a deterministic fallback if not.
    x[~np.isfinite(x)] = 0.0
    scored = table["is_scored"].to_numpy().astype(bool) if validation else None
    return seq_ix, need, x, y, scored


def sample_training_rows(
    parquet: pq.ParquetFile, groups: list[int], rows_per_group: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if rows_per_group < 1:
        raise ValueError("--rows-per-group debe ser positivo")
    xs, ys = [], []
    for position, group in enumerate(groups, start=1):
        _, need, x, y, _ = read_group(parquet, group)
        possible = np.flatnonzero(need)
        n = min(rows_per_group, len(possible))
        selected = np.sort(np.random.default_rng(seed + group).choice(possible, n, replace=False))
        xs.append(x[selected])
        ys.append(y[selected])
        print(f"Muestra de train: {position}/{len(groups)} grupos", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


@dataclass
class R2Accumulator:
    count: int = 0

    def __post_init__(self) -> None:
        self.sum_y = np.zeros(2, dtype=np.float64)
        self.sum_y2 = np.zeros(2, dtype=np.float64)
        self.sse = np.zeros(2, dtype=np.float64)
        self.sae = np.zeros(2, dtype=np.float64)

    def add(self, y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> None:
        actual = y[mask].astype(np.float64)
        fitted = pred[mask].astype(np.float64)
        self.count += len(actual)
        self.sum_y += actual.sum(axis=0)
        self.sum_y2 += np.square(actual).sum(axis=0)
        self.sse += np.square(actual - fitted).sum(axis=0)
        self.sae += np.abs(actual - fitted).sum(axis=0)

    def result(self) -> dict:
        if self.count == 0:
            return {"t0": None, "t1": None, "mean": None, "mae": None, "rmse": None, "rows": 0}
        sst = self.sum_y2 - np.square(self.sum_y) / self.count
        values = np.where(sst > 1e-12, 1.0 - self.sse / np.maximum(sst, 1e-12),
                          np.where(self.sse <= 1e-12, 1.0, 0.0))
        mae = self.sae / self.count
        rmse = np.sqrt(self.sse / self.count)
        return {
            "t0": float(values[0]), "t1": float(values[1]),
            "mean": float(values.mean()),
            "mae": {"t0": float(mae[0]), "t1": float(mae[1]), "mean": float(mae.mean())},
            "rmse": {"t0": float(rmse[0]), "t1": float(rmse[1]), "mean": float(rmse.mean())},
            "rows": self.count,
        }


def group_wp(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    selected = y[mask]
    eligible = len(selected) > 0 and all(
        np.any(selected[:, j] > 0) and np.any(selected[:, j] < 0)
        for j in range(2)
    )
    if not eligible:
        return {"eligible": False, "wp_t0": None, "wp_t1": None, "wp": None}
    t0 = weighted_pearson(y[mask, 0], pred[mask, 0])
    t1 = weighted_pearson(y[mask, 1], pred[mask, 1])
    return {"eligible": True, "wp_t0": t0, "wp_t1": t1, "wp": (t0 + t1) / 2}


def evaluate(
    predict: Callable[[np.ndarray], np.ndarray],
    train: pq.ParquetFile,
    valid: pq.ParquetFile,
    train_groups: list[int],
    valid_groups: list[int],
) -> tuple[dict, list[dict]]:
    r2 = {
        "train_required": R2Accumulator(),
        "validation_required": R2Accumulator(),
        "validation_scored": R2Accumulator(),
    }
    wp_train = BlockAccumulator()
    wp_valid = BlockAccumulator()
    by_group = []
    for split, parquet, groups in (
        ("train", train, train_groups),
        ("validation", valid, valid_groups),
    ):
        for position, group in enumerate(groups, start=1):
            seq_ix, need, x, y, scored = read_group(parquet, group, split == "validation")
            prediction = np.asarray(predict(x), dtype=np.float32)
            if prediction.shape != (SEQUENCE_LENGTH, 2) or not np.isfinite(prediction[need]).all():
                raise ValueError(f"Predicciones inválidas en {split} row group {group}")
            mask = need if split == "train" else (need & scored)
            if split == "train":
                r2["train_required"].add(y, prediction, need)
                wp_train.add(y, prediction, need)
            else:
                r2["validation_required"].add(y, prediction, need)
                r2["validation_scored"].add(y, prediction, mask)
                wp_valid.add(y, prediction, mask)
            by_group.append({
                "split": split, "row_group": group, "seq_ix": seq_ix,
                "selected_rows": int(mask.sum()), **group_wp(y, prediction, mask),
            })
            print(f"Evaluación {split}: {position}/{len(groups)} grupos", flush=True)
    return {
        "train_wp_proxy_all_required": wp_train.result(),
        "validation_wp_official_mask": wp_valid.result(),
        "r2": {name: accumulator.result() for name, accumulator in r2.items()},
        "note": "Test oficial oculto: validation es el test local. R² usa targets/predicciones sin clipping; WP aplica el clipping oficial. Train WP es proxy porque train no tiene is_scored.",
    }, by_group


def prepare_run(
    folder: Path, args: argparse.Namespace, train: pq.ParquetFile, valid: pq.ParquetFile
) -> tuple[list[int], list[int], Path]:
    train_groups = select_groups(train.num_row_groups, args.train_groups, args.seed)
    valid_groups = select_groups(valid.num_row_groups, args.valid_groups, args.seed + 1)
    output_root = args.output_dir or (folder / "results")
    output = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "train_dataset": str(TRAIN_PATH), "validation_dataset": str(VALID_PATH),
        "train_row_groups": train_groups, "validation_row_groups": valid_groups,
        "rows_per_group_for_fit": args.rows_per_group, "seed": args.seed,
        "scoring_domain": "validation: need_prediction AND is_scored; train proxy: need_prediction",
        "selection": "random without replacement, then sorted",
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return train_groups, valid_groups, output


def save_metrics(output: Path, model_name: str, metrics: dict, by_group: list[dict]) -> None:
    model_dir = output / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8"
    )
    with (model_dir / "by_rowgroup.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_group[0]))
        writer.writeheader()
        writer.writerows(by_group)
    wp = metrics["validation_wp_official_mask"]
    train_r2 = metrics["r2"]["train_required"]
    valid_r2 = metrics["r2"]["validation_required"]
    scored_r2 = metrics["r2"]["validation_scored"]
    print(
        f"{model_name}: WP valid={wp['weighted_pearson']:.6f} "
        f"(t0={wp['t0']:.6f}, t1={wp['t1']:.6f})\n"
        f"  R² train: t0={train_r2['t0']:.6f}, t1={train_r2['t1']:.6f}, media={train_r2['mean']:.6f}\n"
        f"  R² valid: t0={valid_r2['t0']:.6f}, t1={valid_r2['t1']:.6f}, media={valid_r2['mean']:.6f}\n"
        f"  R² valid scored: t0={scored_r2['t0']:.6f}, "
        f"t1={scored_r2['t1']:.6f}, media={scored_r2['mean']:.6f}"
    )


def _duration(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{rest:02d}"


class FitProgress:
    """Show completed units and a rough ETA without changing model parameters."""

    def __init__(self, name: str, total: int, every: int = 1):
        self.name = name
        self.total = total
        self.every = every
        self.started = time.monotonic()
        self.last = 0
        self.last_printed = self.started

    def update(self, done: int, detail: str = "") -> None:
        now = time.monotonic()
        if done <= self.last or (
            done % self.every and done != self.total and now - self.last_printed < 30
        ):
            return
        self.last = done
        self.last_printed = now
        elapsed = now - self.started
        remaining = elapsed * (self.total - done) / done
        suffix = f" | {detail}" if detail else ""
        print(
            f"{self.name}: {done}/{self.total} ({100 * done / self.total:.1f}%)"
            f" | transcurrido {_duration(elapsed)}"
            f" | restante estimado {_duration(remaining)}{suffix}",
            flush=True,
        )


class MLPOutputProgress:
    """Convert scikit-learn's native iteration messages into a count with ETA."""

    def __init__(self, progress: FitProgress, original):
        self.progress = progress
        self.original = original
        self.pending = ""

    def write(self, chunk: str) -> int:
        self.pending += chunk
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = re.match(r"Iteration (\d+), loss = (.+)", line)
            if match:
                self.progress.update(int(match.group(1)), f"loss={match.group(2)}")
            else:
                self.original.write(line + "\n")
                self.original.flush()
        return len(chunk)

    def flush(self) -> None:
        if self.pending:
            self.original.write(self.pending)
            self.pending = ""
        self.original.flush()


def fit_with_progress(model, name: str, x_fit: np.ndarray, y_fit: np.ndarray) -> None:
    """Use each estimator's native progress hook; never restart an MLP fit."""
    started = time.monotonic()
    if name == "MLP":
        mlp = model[-1]
        mlp.verbose = True  # Reporting only; max_iter and stopping stay unchanged.
        print(f"MLP: hasta {mlp.max_iter} iteraciones; puede converger antes.", flush=True)
        progress = FitProgress("MLP", mlp.max_iter, every=max(1, mlp.max_iter // 20))
        output = MLPOutputProgress(progress, sys.stdout)
        with redirect_stdout(output):
            model.fit(x_fit, y_fit)
        output.flush()
        print(f"MLP: terminó en {mlp.n_iter_} iteraciones; tiempo {_duration(time.monotonic() - started)}.", flush=True)
        return

    if name == "RandomForest":
        # Joblib reports completed trees; verbosity does not alter the forest.
        model.verbose = 1
        print(f"RandomForest: construyendo {model.n_estimators} árboles; Joblib informa los completados.", flush=True)
        try:
            model.fit(x_fit, y_fit)
        finally:
            model.verbose = 0  # Do not repeat progress logs during evaluation.
        print(f"RandomForest: {len(model.estimators_)}/{model.n_estimators} árboles; tiempo {_duration(time.monotonic() - started)}.", flush=True)
        return

    if name in ("LightGBM", "XGBoost"):
        from sklearn.base import clone

        rounds = model.estimator.n_estimators
        progress = FitProgress(name, 2 * rounds, every=max(1, rounds // 20))
        estimators = []
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            estimator = clone(model.estimator)
            offset = target_index * rounds
            if name == "LightGBM":
                def report(env, offset=offset, target_name=target_name):
                    iteration = env.iteration - env.begin_iteration + 1
                    progress.update(offset + iteration, f"{target_name}: ronda {iteration}/{rounds}")

                report.order = 10
                report.before_iteration = False
                estimator.fit(x_fit, y_fit[:, target_index], callbacks=[report])
            else:
                from xgboost.callback import TrainingCallback

                class ReportProgress(TrainingCallback):
                    def after_iteration(self, booster, epoch, evals_log):
                        iteration = epoch + 1
                        progress.update(offset + iteration, f"{target_name}: ronda {iteration}/{rounds}")
                        return False

                estimator.set_params(callbacks=[ReportProgress()])
                estimator.fit(x_fit, y_fit[:, target_index])
            estimators.append(estimator)
        model.estimators_ = estimators  # Same two independent fits as MultiOutputRegressor.fit().
        completed = sum(getattr(estimator, "n_iter_", rounds) for estimator in estimators)
        print(f"{name}: {completed}/{2 * rounds} rondas; tiempo {_duration(time.monotonic() - started)}.", flush=True)
        return

    print(f"{name}: ajuste único, sin contador de iteraciones disponible.", flush=True)
    model.fit(x_fit, y_fit)
    print(f"{name}: ajuste terminado; tiempo {_duration(time.monotonic() - started)}.", flush=True)


def run_tabular(names: list[str], folder: Path) -> None:
    parser = build_parser("FirstTry tabular: sin búsqueda de hiperparámetros ni K-fold.")
    args = parser.parse_args()
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        parser.error("Falta scikit-learn. Instala: python -m pip install scikit-learn")

    train = pq.ParquetFile(TRAIN_PATH)
    valid = pq.ParquetFile(VALID_PATH)
    train_groups, valid_groups, output = prepare_run(folder, args, train, valid)
    x_fit, y_fit = sample_training_rows(train, train_groups, args.rows_per_group, args.seed + 2)
    for name in names:
         
        if name == "Ridge":
            model = make_pipeline(
            StandardScaler(),
            Ridge()
        )

        elif name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                max_features="sqrt",
                random_state=args.seed,
                n_jobs=-1
            )

        elif name == "MLP":
            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=100,
                    random_state=args.seed
                )
            )

        elif name == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                parser.error(
                    "Falta LightGBM. Instala: python -m pip install lightgbm"
                )

            model = MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    n_jobs=-1,
                    verbosity=-1,
                    random_state=args.seed
                ),
                n_jobs=1
            )

        elif name == "XGBoost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                parser.error(
                    "Falta XGBoost. Instala: python -m pip install xgboost"
                )

            model = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=args.seed
                ),
                n_jobs=1
            )

        else:
            raise ValueError(f"Modelo desconocido: {name}")

        print(f"Ajustando {name} con {len(x_fit):,} filas...", flush=True)
        fit_with_progress(model, name, x_fit, y_fit)
        metrics, by_group = evaluate(model.predict, train, valid, train_groups, valid_groups)
        metrics["fit_rows"] = len(x_fit)
        metrics["model"] = name
        save_metrics(output, name, metrics, by_group)
    print(f"Resultados en {output.resolve()}")
