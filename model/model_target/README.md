# GRU con feedback de targets predichos

Compara cinco veces la GRU base (112 entradas) con una GRU que agrega 16 entradas:
para `t0` y `t1`, lags de 1, 2, 5 y 10 pasos y deltas de 1, 2, 5 y 10.
Cada ronda entrena los dos modelos desde cero sobre las mismas 100 secuencias;
las cinco rondas usan grupos de train disjuntos y comparten 50 grupos nuevos de
`valid.parquet`. La selección excluye grupos registrados en experimentos previos.
Ambas GRU tienen una capa de 32 unidades, dos salidas, 2 épocas, chunks de 128
pasos y la métrica Weighted Pearson oficial en validación.

El test del challenge no entrega `t0` ni `t1`. Por eso **los targets reales nunca
se usan para formar las entradas**: solo sirven para calcular pérdida, métrica y
escala de entrenamiento. Antes de predecir el paso `t`, el modelo conoce sus
propias predicciones anteriores. `lag_k[t] = pred[t-k]` y
`delta_k[t] = pred[t-1] - pred[t-1-k]`. Este delta se retrasa un paso porque
`pred[t]` aún no existe al formar la entrada del paso `t`. Al inicio de cada
secuencia, los valores sin historia se codifican como cero. El historial se
reinicia entre secuencias. Se aplica la misma regla en train, validación y en
el callback fila a fila de [inference.py](inference.py).

Desde la raíz del proyecto, ejecutar manualmente en PowerShell:

```powershell
.\.venv\Scripts\python.exe my_solution\model\model_target\train_gru_target_feedback.py
```

Prueba de la causalidad y de la paridad de generación:

```powershell
.\.venv\Scripts\python.exe -m unittest my_solution.model.model_target.test_feedback
```

Si se corta una corrida, `--resume RUTA_DE_LA_CORRIDA` continúa con los mismos
grupos y omite los modelos completos. `results/run_FECHA/run_config.json` guarda
la selección antes de entrenar. Cada `round_XX` contiene métricas, resultados
por row group y checkpoints. `summary.csv` muestra WP de `t0`, `t1`, promedio y
sus diferencias frente a la base. `aggregate.json` cuenta las rondas que
favorecieron al feedback. La validación es un test local; el test oficial está
oculto.
