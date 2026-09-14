"""Features causales construidas solo con predicciones anteriores de t0/t1."""

from __future__ import annotations

from collections import deque

import numpy as np

from utils import TARGET_COLUMNS


LAGS = (1, 2, 5, 10)
TARGETS = TARGET_COLUMNS
FEATURE_NAMES = tuple(
    f"{target}_pred_{kind}_{lag}"
    for target in TARGETS
    for kind in ("lag", "delta")
    for lag in LAGS
)


class PredictionHistory:
    """Se instancia de nuevo al comenzar cada secuencia de 20.000 pasos."""

    def __init__(self, target_scale: np.ndarray):
        self.scale = np.asarray(target_scale, dtype=np.float32)
        if self.scale.shape != (2,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("target_scale debe tener dos valores positivos y finitos")
        self.history: deque[np.ndarray] = deque(maxlen=max(LAGS) + 1)

    def features(self) -> np.ndarray:
        """Valores disponibles justo ANTES de predecir el paso actual."""
        result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        position = 0
        for target_index in range(2):
            for lag in LAGS:
                if len(self.history) >= lag:
                    result[position] = self.history[-lag][target_index] / self.scale[target_index]
                position += 1
            for lag in LAGS:
                # delta_k[t] = pred[t-1] - pred[t-1-k], sin pred[t].
                if len(self.history) >= lag + 1:
                    result[position] = (
                        self.history[-1][target_index] - self.history[-1-lag][target_index]
                    ) / self.scale[target_index]
                position += 1
        return result

    def append(self, prediction: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (2,) or not np.isfinite(prediction).all():
            raise ValueError("La predicción debe tener t0 y t1 finitos")
        # La misma regla se usa durante entrenamiento y validación.
        self.history.append(np.clip(prediction, -2.0, 2.0).copy())


def feedback_from_predictions(predictions: np.ndarray, target_scale: np.ndarray) -> np.ndarray:
    """Referencia por secuencia para comprobar la paridad con el flujo online."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Se esperaba una predicción t0/t1 por paso")
    history = PredictionHistory(target_scale)
    output = np.empty((len(predictions), len(FEATURE_NAMES)), dtype=np.float32)
    for step, prediction in enumerate(predictions):
        output[step] = history.features()
        history.append(prediction)
    return output
