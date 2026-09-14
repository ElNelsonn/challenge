r"""Dos ablaciones: top 3 por target más una feature adicional de t0.

Modelo A: top 3 de t0 y t1 + i0_v21_delta_5.
Modelo B: top 3 de t0 y t1 + i0_p3_delta_50.

Mismos grupos y ajustes que el experimento top 3 anterior: 100 train,
50 valid, 2 épocas, semilla 2026, GRU de 1 capa y 32 unidades.

Desde la raíz del proyecto (PowerShell):

    .\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_top3_ablation.py

El script entrena solamente estos dos modelos; no vuelve a entrenar la base.
"""

from features import TemporalFeature
from train_gru import main


if __name__ == "__main__":
    main(
        variants=(
            ("top3_plus_v21_delta5", 3),
            ("top3_plus_p3_delta50", 3),
        ),
        train_groups_default=100,
        valid_groups_default=50,
        extra_features_by_variant={
            "top3_plus_v21_delta5": (TemporalFeature("i0_v21", "delta", 5),),
            "top3_plus_p3_delta50": (TemporalFeature("i0_p3", "delta", 50),),
        },
    )
