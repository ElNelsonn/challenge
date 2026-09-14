"""Indicadores de saturacion de las columnas a0..a7, sin cambiar sus valores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


A_COLUMNS = tuple(f"a{i}" for i in range(8))
assert all(name in FEATURE_COLUMNS for name in A_COLUMNS)
A_INDICES = tuple(FEATURE_COLUMNS.index(name) for name in A_COLUMNS)
FEATURE_NAMES = tuple(name + suffix for name in A_COLUMNS
                      for suffix in ("_is_min", "_is_max"))
BOUNDARY = 5.199337482
# x fue convertido a float32 por read_group: comparar contra el mismo float32.
# 2e-7 es menos de medio ULP alrededor de 5.2; no etiqueta valores cercanos
# distintos de la representacion del limite observado.
ATOL = 2e-7


def boundary_flags(x: np.ndarray) -> np.ndarray:
    """Acepta una fila de 112 entradas o una secuencia de filas, en ese orden."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError("Se esperan las 112 entradas originales por fila")
    a_values = values[..., A_INDICES]
    flags = np.stack((np.isclose(a_values, np.float32(-BOUNDARY), rtol=0, atol=ATOL),
                      np.isclose(a_values, np.float32(BOUNDARY), rtol=0, atol=ATOL)),
                     axis=-1)
    return flags.reshape(*values.shape[:-1], len(FEATURE_NAMES)).astype(np.float32)


def model_inputs(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Originales normalizadas y flags binarios sin normalizar."""
    raw = np.asarray(x, dtype=np.float32)
    normalized = (raw - mean) / scale
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((normalized, boundary_flags(raw)), axis=-1).astype(np.float32)
