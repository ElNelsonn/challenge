r"""Primer experimento con un MLP simple.

Desde la raíz del proyecto (PowerShell):
    .\.venv\Scripts\python.exe "my_solution\models\MLP simple\FirstTry.py"

Ejemplo con tamaños elegidos:
    .\.venv\Scripts\python.exe "my_solution\models\MLP simple\FirstTry.py" --train-groups 10 --valid-groups 3 --rows-per-group 2000 --seed 2026

Opciones disponibles:
    --train-groups N     Row groups de train: 1 a 10607 (default: 3).
    --valid-groups N     Row groups de validación: 1 a 1873 (default: 1).
    --rows-per-group N   Filas por grupo para ajustar: 1 a 19901 efectivas (default: 1000).
    --seed N             Semilla para elegir grupos y filas (default: 2026).
    --output-dir RUTA    Carpeta base de resultados (default: MLP simple/results).
    -h, --help           Muestra la ayuda de la línea de comandos.

Las métricas usan todas las filas requeridas de los grupos seleccionados.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_try_common import run_tabular  # noqa: E402


if __name__ == "__main__":
    run_tabular(["MLP"], Path(__file__).resolve().parent)
