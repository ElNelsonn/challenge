r"""GRU con las 112 entradas originales más solo i0_v21_delta_5.

Usa los mismos ajustes de las corridas anteriores: 100 secuencias de train,
50 de validación, 2 épocas, semilla 2026, GRU de 1 capa y 32 unidades.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_only_v21_delta5.py

Entrena un único modelo. No incluye ninguna de las features del top 3.
Para comparar, usa el WP de la base de la corrida anterior (0.4413856794).
"""

from features import TemporalFeature
from train_gru import main


if __name__ == "__main__":
    main(
        variants=(("only_v21_delta5", 0),),
        train_groups_default=100,
        valid_groups_default=50,
        extra_features_by_variant={
            "only_v21_delta5": (TemporalFeature("i0_v21", "delta", 5),),
        },
    )
