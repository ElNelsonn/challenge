# Entrenamiento oficial GPU/CPU (`train_official_gpu_v1.py`)

Un solo script para la PC (CPU) y para Kaggle (GPU). Detecta `cuda` solo.

## Qué cambia respecto a `models_first_try` y `model/feature_select`

| | FirstTry / feature_select | Este script |
|---|---|---|
| Secuencias por paso | 1 | B en paralelo (`--batch`, default 32 GPU / 8 CPU) |
| Loss | MSE cruda | Alineada a la métrica: targets clip ±2, peso \|y\| (`--loss wmse`), o −WeightedPearson (`--loss wp`) |
| Gradient clipping | no | sí (`--clip 1.0`) |
| Tamaño default | GRU 32×1 / LSTM 64×1 | GRU 128×2 (= baseline oficial) |
| Checkpoints / retomar | no | `checkpoint.pt` por época, `--resume RUN_DIR` |
| GPU | no | sí |
| Features extra | fijas por script | opcionales: `--top-k N` (lags/deltas de `feature_select/features.py`), `--sentinel-flags` |

La evaluación es la métrica oficial (`BlockAccumulator` de `utils.py`) sobre `need_prediction AND is_scored`.
Con `--valid-groups 10 --seed 2026` (default) son **los mismos 10 grupos de `EXPERIMENTS.md`**, donde el
baseline oficial da **0.568**. Reportar siempre la distancia a ese número.

## Uso local (desde `challenge/`)

```powershell
python model\official_gpu\train_official_gpu_v1.py --train-groups 500 --epochs 3
```

Opciones principales: `--cell gru|lstm`, `--hidden`, `--layers`, `--epochs`, `--lr`, `--lr-decay`,
`--clip`, `--loss wmse|wp`, `--top-k 0|3|5|10|20`, `--sentinel-flags`, `--train-groups N|all`,
`--valid-groups N`, `--seed`, `--run-name`, `--resume RUN_DIR`, `--eval-every N`.

## Uso en Kaggle (GPU)

Notebook con los 3 datasets adjuntos y `Accelerator = GPU`:

```bash
python train_official_gpu_v1.py \
  --train-parquet /kaggle/input/connectome-train-part1/train_part1.parquet \
                  /kaggle/input/connectome-train-part2/train_part2.parquet \
  --valid-parquet /kaggle/input/connectome-valid/valid.parquet \
  --train-groups all --epochs 5 --output-dir /kaggle/working/results --run-name full_gru128x2
```

Si la sesión se corta (12 h), en la siguiente: `--resume /kaggle/working/results/full_gru128x2`.

## Salida de cada corrida (`results/<run>/`)

- `run_config.json` — todos los parámetros y los ids de grupos usados.
- `metrics.json` — mejor WP, último resultado, historial por época.
- `best_model.pt` — pesos + config + normalizador + features (para inferencia).
- `best_by_rowgroup.csv` — WP por secuencia de valid del mejor modelo.
- `checkpoint.pt` — para `--resume`.
- `last_model.pt` — pesos finales.

## Nota sobre batches vs. FirstTry

Con batch B hay B veces menos actualizaciones por época que con una secuencia por vez. Con pocas
secuencias (≤100) y 1 época eso puede dar un número más bajo que el FirstTry equivalente; no es un
bug. Con todo el dataset (10.607 secuencias) son ~13.000 actualizaciones por época y la diferencia
desaparece. Si se quiere comparar con FirstTry en chico, usar `--batch 1 --chunk 128`.
