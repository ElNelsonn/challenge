# GRU con selección de features temporales

Desde la raíz del proyecto, con PyTorch instalado en `.venv`:

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru.py
```

Para probar **top 3 de cada target** contra la base con los ajustes de la corrida
anterior (100 secuencias train, 50 valid, 2 épocas, semilla 2026):

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_top3.py
```

Este segundo archivo reutiliza el mismo entrenamiento y genera otra carpeta de
resultados con `base` y `top_3`. Los seis lags/deltas provienen de las entradas
originales; no se construyen features con targets reales ni predicciones.

Para aislar las features 4 y 5 del ranking de `t0`, entrená **solo dos modelos**:
ambos usan el top 3 de cada target; el primero suma `i0_v21_delta_5` y el
segundo `i0_p3_delta_50`. Conserva los mismos 100 grupos de train, 50 de valid,
2 épocas y semilla que la corrida top 3:

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_top3_ablation.py
```

El CSV de esta corrida compara ambos Weighted Pearson. Como no vuelve a
entrenar la base, las columnas `delta_vs_base_*` quedan vacías; usá el
`comparison.csv` de la corrida top 3 anterior como referencia.

Para probar si `i0_v21_delta_5` aporta por sí sola, sin el top 3, entrená un
único modelo con las 112 entradas originales más esa variable:

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_only_v21_delta5.py
```

Usa los mismos 100 grupos de train, 50 de validación, 2 épocas y semilla 2026.
El CSV generado contiene el WP de este único modelo; comparalo con la base de
la corrida anterior (WP promedio 0.4413856794).

La ejecución predeterminada compara cuatro GRU de 1 capa y 32 unidades: `base` (112 entradas), `top_5`, `top_10` y `top_20`. Cada top K agrega la unión de K features temporales de `t0` y K de `t1` del archivo `my_solution/explore/feature_select/results/top_20_by_target.csv`; los duplicados se cuentan una sola vez. Todas predicen ambos targets. Usa las mismas 20 secuencias completas de train, 10 de `valid.parquet`, 2 épocas y semilla 2026. Para cambiarlo:

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru.py --train-groups 100 --valid-groups 20 --epochs 2 --seed 2026
```

Si el ranking está en otra ruta, agrega `--ranking RUTA.csv`. El programa exige al menos 20 lags/deltas válidos por target y falla antes de entrenar si el ranking no cumple el esquema.

Cada row group se valida como una secuencia completa con `step_in_seq` en orden. Lags y deltas se calculan con datos pasados de esa misma secuencia. La normalización se ajusta solo con pasos requeridos de train. En los primeros 50 pasos, la historia ausente de una feature temporal se representa como cero **después de normalizar**, únicamente para alimentar la GRU; el análisis de correlación conserva sus NaN. La pérdida empieza en el paso 99. El estado GRU se reinicia al comenzar cada secuencia y se mantiene entre fragmentos de 128 pasos.

Cada ejecución crea `results/run_FECHA_HORA/`. `run_config.json` registra los grupos y el hash del ranking; `comparison.csv` contiene Weighted Pearson de validación para `t0`, `t1`, promedio y diferencia respecto de `base`. Cada variante guarda `metrics.json` con sus entradas exactas, `by_rowgroup.csv` y `model.pt` con pesos, normalizador y definición de features. La métrica de validación usa `need_prediction AND is_scored`, igual que el evaluador local. El test oficial está oculto: `valid.parquet` es el test local. `CausalFeatureBuilder` en `features.py` permite construir las mismas entradas fila a fila en una futura inferencia.

Para correr las pruebas sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s my_solution\model\feature_select -p "test_*.py" -v
```

Este experimento no modifica los parquets ni genera un ZIP de envío.

## Cinco pares en grupos nuevos

Para comparar `base` contra `top 3 de ambos targets + i0_v21_delta_5` en cinco
conjuntos distintos de train (10 entrenamientos en total):

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_five_group_pairs.py
```

Cada par comparte sus 100 secuencias de train. Las cinco rondas usan 500
secuencias distintas, ninguna de las registradas en experimentos anteriores.
También comparten 50 secuencias nuevas de validación. El script revisa los
`run_config.json` de `my_solution` y el muestreo del ranking antes de elegirlas.
Usa semilla 2027 solo para elegir grupos y 2026 para inicializar las diez GRU.
Mantiene 2 épocas, fragmentos de 128 pasos, una capa y 32 unidades.

Al iniciar, escribe `results/group_repeats/run_FECHA/run_config.json` con los
grupos exactos. Cada `round_XX` contiene los dos modelos y su comparación;
`summary.csv` se actualiza después de cada ronda y `aggregate.json` resume
las cinco diferencias de WP por target y promedio. Si se interrumpe, los
grupos ya reservados quedan en el manifiesto. Para retomar la misma corrida sin
reentrenar modelos completos, ejecutá:

```powershell
.\.venv\Scripts\python.exe my_solution\model\feature_select\train_gru_five_group_pairs.py --resume RUTA_DE_LA_CORRIDA
```
