# FirstTry: modelos sin ajuste de hiperparámetros

Cada carpeta tiene un `FirstTry.py`. La carpeta de LightGBM/XGBoost ejecuta
ambos modelos por separado; la de GRU/LSTM/TCN ejecuta tres. No hay K-fold,
búsqueda de parámetros ni selección según validación.

Todos usan la misma selección reproducible de row groups con la misma semilla.
Por defecto usan 3 grupos de entrenamiento y 1 de validación. En modelos
tabulares, se ajusta con 1.000 filas elegidas de cada grupo de entrenamiento;
se evalúa en **todas** las filas que requieren predicción de esos grupos. Las
redes secuenciales se entrenan en las 20.000 filas de cada grupo, con una época
por defecto; `--rows-per-group` solo controla la muestra para normalizar.
Los defaults son una prueba rápida; un solo grupo de validación no da una
estimación estable del rendimiento general.

Durante el ajuste se informa el avance: MLP por iteraciones, Random Forest
por árboles completados, LightGBM/XGBoost por rondas de `t0` y `t1`, y las
redes secuenciales por fragmentos/grupos. El tiempo restante es una
estimación y puede variar; MLP puede detenerse antes del máximo de iteraciones.
Ridge tiene un único ajuste y solo muestra comienzo y duración final.

## Instalar lo necesario

Las bibliotecas de modelos no están en el `requirements.txt` original del
challenge. Instálalas en tu entorno virtual sin modificar ese archivo:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn lightgbm xgboost torch
```

## Comandos (desde la raíz del proyecto)

```powershell
.\.venv\Scripts\python.exe "my_solution\models\Ridge\FirstTry.py"
.\.venv\Scripts\python.exe "my_solution\models\RandomForest\FirstTry.py"
.\.venv\Scripts\python.exe "my_solution\models\MLP simple\FirstTry.py"
.\.venv\Scripts\python.exe "my_solution\models\LightGBM -XGBoost\FirstTry.py"
.\.venv\Scripts\python.exe "my_solution\models\GRU -LSTM -TCN\FirstTry.py"
```

Para cambiar el tamaño inicial del experimento, agrega por ejemplo:

```powershell
--train-groups 10 --valid-groups 3 --rows-per-group 2000 --seed 2026
```

En GRU/LSTM/TCN también se puede usar `--epochs 2`; el valor inicial es 1.
Los modelos son baselines locales, no ZIP de envío listos para subir. Antes de
enviarlos habría que medir velocidad/tamaño y crear un `solution.py` con la
interfaz fila a fila que exige la competencia.

## Resultados

Cada ejecución crea `results/run_FECHA_HORA/` dentro de su carpeta de modelos.
Cada modelo escribe `metrics.json` y `by_rowgroup.csv`; las redes también
guardan `model.pt`. `run_config.json` registra los índices de row group usados.

- `validation_wp_official_mask`: Weighted Pearson de la competencia, calculado
  por secuencia sobre `need_prediction AND is_scored`, con el recorte oficial
  `[-2,2]`. Incluye puntajes `t0`, `t1` y agregado.
- `train_wp_proxy_all_required`: misma fórmula sobre todas las filas requeridas
  de train. **No es el puntaje oficial**, porque train no tiene `is_scored`.
- `r2.train_required`: R² de train por target y promedio, sobre todas las filas
  que requieren predicción de los grupos elegidos, no solo la muestra de ajuste.
- `r2.validation_required`: R² local de validación sobre todas las filas
  requeridas.
- `r2.validation_scored`: R² local sobre las filas que puntúa el challenge.

Cada bloque de `r2` incluye además MAE y RMSE por target y su promedio.

El **test oficial está oculto**; no es posible calcular su R² localmente.
`valid.parquet` es el equivalente disponible a un test local. R² se calcula
con targets y predicciones sin recorte, mientras que Weighted Pearson aplica
el recorte especificado por el challenge.
