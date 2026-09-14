"""Callback fila a fila para checkpoints de la variante a_boundary_flags."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP  # noqa: E402
from .flags import ATOL, BOUNDARY, FEATURE_NAMES, model_inputs  # noqa: E402


class BoundaryFlagPredictor:
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
        if (payload["input_features"] != [*FEATURE_COLUMNS, *FEATURE_NAMES]
                or payload["boundary"] != BOUNDARY or payload["atol"] != ATOL):
            raise ValueError("Checkpoint incompatible con los flags de limites")
        self.model = GRUModel()
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.scale = np.asarray(payload["scale"], dtype=np.float32)
        self.torch = torch
        self.seq_ix = None
        self.next_step = 0
        self.hidden = None

    def predict(self, point) -> np.ndarray | None:
        if point.seq_ix != self.seq_ix:
            if point.step_in_seq != 0:
                raise ValueError("Cada secuencia debe comenzar en step_in_seq=0")
            self.seq_ix = point.seq_ix
            self.next_step = 0
            self.hidden = None
        if point.step_in_seq != self.next_step:
            raise ValueError("Paso fuera de orden dentro de la secuencia")
        state = np.asarray(point.state, dtype=np.float32)
        if state.shape != (len(FEATURE_COLUMNS),):
            raise ValueError("Se esperan las 112 entradas originales")
        state = np.where(np.isfinite(state), state, 0.0).astype(np.float32)
        row = model_inputs(state, self.mean, self.scale)
        with self.torch.inference_mode():
            output, self.hidden = self.model(self.torch.from_numpy(row).view(1, 1, -1),
                                             self.hidden)
        self.next_step += 1
        if point.step_in_seq < WARMUP:
            return None
        return output.view(-1).numpy().copy()
