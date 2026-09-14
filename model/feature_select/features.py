"""Selección y construcción causal de entradas para la GRU.

Las filas sin historia conservan NaN aquí. Solo el normalizador del modelo las
convierte en cero estandarizado durante el warm-up; el CSV de correlaciones no
se altera.
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402


LAGS = (1, 5, 10, 50)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
REQUIRED_RANKING_COLUMNS = {"rank", "feature", "base_feature", "transform", "lag", "target"}


@dataclass(frozen=True)
class TemporalFeature:
    base_feature: str
    transform: str
    lag: int

    @property
    def name(self) -> str:
        return f"{self.base_feature}_{self.transform}_{self.lag}"


def load_rankings(path: Path) -> dict[str, list[TemporalFeature]]:
    ranked: dict[str, list[tuple[int, TemporalFeature]]] = {
        target: [] for target in TARGET_COLUMNS
    }
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_RANKING_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"Ranking sin columnas requeridas: {sorted(REQUIRED_RANKING_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                target = row["target"]
                rank = int(row["rank"])
                lag = int(row["lag"])
                spec = TemporalFeature(row["base_feature"], row["transform"], lag)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Ranking inválido en línea {line}") from exc
            if (target not in ranked or rank < 1 or spec.base_feature not in FEATURE_INDEX
                    or spec.transform not in ("lag", "delta") or lag not in LAGS
                    or row["feature"] != spec.name):
                raise ValueError(f"Feature o target inválido en línea {line}: {row}")
            ranked[target].append((rank, spec))

    output = {}
    for target, entries in ranked.items():
        entries.sort(key=lambda item: item[0])
        ranks = [rank for rank, _ in entries]
        names = [spec.name for _, spec in entries]
        if ranks != list(range(1, len(entries) + 1)) or len(set(names)) != len(names):
            raise ValueError(f"Ranks salteados o features duplicadas para {target}")
        if len(entries) < 20:
            raise ValueError(f"Se requieren al menos 20 features derivadas para {target}")
        output[target] = [spec for _, spec in entries]
    return output


def select_temporal_features(
    rankings: dict[str, list[TemporalFeature]], top_k: int
) -> list[TemporalFeature]:
    if top_k not in (0, 3, 5, 10, 20):
        raise ValueError("top_k debe ser 0, 3, 5, 10 o 20")
    selected: list[TemporalFeature] = []
    seen = set()
    for target in TARGET_COLUMNS:
        if len(rankings[target]) < top_k:
            raise ValueError(f"Ranking insuficiente para {target}")
        for spec in rankings[target][:top_k]:
            if spec.name not in seen:
                selected.append(spec)
                seen.add(spec.name)
    return selected


def build_sequence(raw: np.ndarray, specs: list[TemporalFeature]) -> np.ndarray:
    """Entrada original más derivadas; `raw` ya corresponde a una sola secuencia."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Se esperaban {len(FEATURE_COLUMNS)} columnas originales")
    if not np.isfinite(raw).all():
        raw = np.where(np.isfinite(raw), raw, 0.0)
    if not specs:
        return raw
    derived = np.full((len(raw), len(specs)), np.nan, dtype=np.float32)
    for column, spec in enumerate(specs):
        if spec.lag >= len(raw):
            continue
        base = raw[:, FEATURE_INDEX[spec.base_feature]]
        past = base[:-spec.lag]
        derived[spec.lag:, column] = (
            past if spec.transform == "lag" else base[spec.lag:] - past
        )
    return np.column_stack((raw, derived))


class CausalFeatureBuilder:
    """Versión fila a fila para validación de paridad y futuro test oculto."""

    def __init__(self, specs: list[TemporalFeature]):
        self.specs = list(specs)
        self.history: deque[np.ndarray] = deque(maxlen=max(LAGS))
        self.current_seq_ix: int | None = None
        self.previous_step: int | None = None

    def push(self, seq_ix: int, step_in_seq: int, state: np.ndarray) -> np.ndarray:
        seq_ix = int(seq_ix)
        step_in_seq = int(step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step_in_seq != 0:
                raise ValueError("Una nueva secuencia debe empezar en step_in_seq=0")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.history.clear()
        if self.previous_step is not None and step_in_seq != self.previous_step + 1:
            raise ValueError("step_in_seq no es consecutivo")
        raw = np.asarray(state, dtype=np.float32)
        if raw.shape != (len(FEATURE_COLUMNS),):
            raise ValueError(f"state debe contener {len(FEATURE_COLUMNS)} entradas")
        if not np.isfinite(raw).all():
            raw = np.where(np.isfinite(raw), raw, 0.0)
        derived = np.full(len(self.specs), np.nan, dtype=np.float32)
        for column, spec in enumerate(self.specs):
            if len(self.history) >= spec.lag:
                past = self.history[-spec.lag][FEATURE_INDEX[spec.base_feature]]
                current = raw[FEATURE_INDEX[spec.base_feature]]
                derived[column] = past if spec.transform == "lag" else current - past
        self.history.append(raw.copy())
        self.previous_step = step_in_seq
        return np.concatenate((raw, derived))
