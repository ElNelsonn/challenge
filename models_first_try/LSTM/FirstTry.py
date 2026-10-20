r"""Primer experimento solo con LSTM (misma infraestructura que GRU -LSTM -TCN).

Desde challenge/ (PowerShell):
    python "models_first_try\LSTM\FirstTry.py" --train-groups 20 --valid-groups 10 --epochs 1 --seed 2026

Opciones: ver "GRU -LSTM -TCN\FirstTry.py" (son las mismas).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_try_torch import run_torch_models  # noqa: E402


if __name__ == "__main__":
    run_torch_models(["LSTM"], Path(__file__).resolve().parent)
