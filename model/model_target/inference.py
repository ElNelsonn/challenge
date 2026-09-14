"""Callback causal para evaluar un checkpoint target_feedback paso a paso.

Ejemplo de uso en otro Python (no ejecuta nada al importar):

    from my_solution.model.model_target.inference import TargetFeedbackPredictor
    predictor = TargetFeedbackPredictor("ruta/a/round_01/target_feedback/model.pt")
    # scorer.score(predictor)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP  # noqa: E402
from .feedback import FEATURE_NAMES, PredictionHistory  # noqa: E402


class TargetFeedbackPredictor:
    """Solo necesita `state`, `seq_ix` y `step_in_seq` del DataPoint."""

    def __init__(self, checkpoint: str | Path):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Falta PyTorch: python -m pip install torch") from exc

        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(len(FEATURE_COLUMNS) + len(FEATURE_NAMES), 32,
                                  num_layers=1, batch_first=True)
                self.head = nn.Linear(32, len(TARGET_COLUMNS))

            def forward(self, inputs, hidden=None):
                values, hidden = self.gru(inputs, hidden)
                return self.head(values), hidden

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["input_features"] != [*FEATURE_COLUMNS, *FEATURE_NAMES]:
            raise ValueError("El checkpoint no corresponde a target_feedback")
        self.model = GRUModel()
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.torch = torch
        self.x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
        self.x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
        self.target_scale = np.asarray(payload["target_scale"], dtype=np.float32)
        self.seq_ix = None
        self.next_step = 0
        self.hidden = None
        self.history = PredictionHistory(self.target_scale)

    def predict(self, point) -> np.ndarray | None:
        if point.seq_ix != self.seq_ix:
            if point.step_in_seq != 0:
                raise ValueError("Una secuencia nueva debe comenzar en step_in_seq=0")
            self.seq_ix = point.seq_ix
            self.next_step = 0
            self.hidden = None
            self.history = PredictionHistory(self.target_scale)
        if point.step_in_seq != self.next_step:
            raise ValueError("Pasos fuera de orden dentro de la secuencia")
        state = np.asarray(point.state, dtype=np.float32)
        if state.shape != (len(FEATURE_COLUMNS),):
            raise ValueError("El estado debe contener las 112 entradas originales")
        state = np.where(np.isfinite(state), state, 0.0)
        normalized = (state - self.x_mean) / self.x_scale
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        row = np.concatenate((normalized, self.history.features())).astype(np.float32)
        with self.torch.inference_mode():
            output, self.hidden = self.model(self.torch.from_numpy(row).view(1, 1, -1),
                                             self.hidden)
        prediction = output.view(-1).numpy()
        self.history.append(prediction)
        self.next_step += 1
        if point.step_in_seq < WARMUP:
            return None
        return prediction.copy()
