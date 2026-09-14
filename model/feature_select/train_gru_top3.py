r"""Comparar GRU base contra top 3 temporal de cada target.

Reproduce los ajustes de la corrida anterior: 100 secuencias de train,
50 de validación, 2 épocas, seed 2026 y fragmentos de 128 pasos. Entrena de
nuevo la base y top_3 sobre exactamente los mismos row groups de esa corrida.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_top3.py

No usa targets reales como entradas. Los seis lags/deltas elegidos se crean
con las 112 entradas originales y solo con pasos anteriores de cada secuencia.
"""

from train_gru import main


if __name__ == "__main__":
    main(
        variants=(("base", 0), ("top_3", 3)),
        train_groups_default=100,
        valid_groups_default=50,
    )
