r"""Primeros experimentos con GRU, LSTM y TCN; resultados separados.

Desde la raíz del proyecto (PowerShell):
    .\.venv\Scripts\python.exe "my_solution\models\GRU -LSTM -TCN\FirstTry.py"

Ejemplo con tamaños elegidos:
    .\.venv\Scripts\python.exe "my_solution\models\GRU -LSTM -TCN\FirstTry.py" --train-groups 2 --valid-groups 2 --rows-per-group 1000 --epochs 1 --seed 2026

Opciones disponibles:
    --train-groups N     Row groups completos para entrenar: 1 a 10607 (default: 3).
    --valid-groups N     Row groups completos de validación: 1 a 1873 (default: 1).
    --rows-per-group N   Filas SOLO para normalizar: 1 a 19901 efectivas (default: 1000).
    --epochs N           Épocas de entrenamiento (default: 1; mínimo: 1).
    --seed N             Semilla para elegir grupos e inicializar redes (default: 2026).
    --output-dir RUTA    Carpeta base de resultados (default: GRU -LSTM -TCN/results).
    -h, --help           Muestra la ayuda de la línea de comandos.

Estas redes entrenan con cada secuencia completa, no con --rows-per-group filas.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_try_torch import run_torch_models  # noqa: E402


if __name__ == "__main__":
    run_torch_models(["GRU", "LSTM", "TCN"], Path(__file__).resolve().parent)
