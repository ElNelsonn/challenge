# GRU con flags de extremos en `a0`–`a7`

En el [perfil de train](../../explore/variables/results/runs/run_20260913_122511_562462/variables_profile.html)
de 3 millones de filas, el mínimo `−5.199337482` se repite 381.710 veces en
`a5` (12,7 %), 385.324 en `a6` (12,8 %) y 1.081.487 en `a7` (36,0 %).
En `a0`–`a4` aparece entre 1 y 65 veces. El extremo positivo es infrecuente:
en esa muestra aparece cuatro veces en `a5`, no aparece en `a6`, y dos veces
en `a7`. El perfil describe una muestra y no prueba por sí solo que sean
valores de clipping. Por eso el experimento mantiene los valores originales y
solo agrega `aN_is_min` y `aN_is_max` para cada `a0`–`a7`.

Se reutilizan las cinco GRU base de
[`feature_select/results/group_repeats/run_20260913_205201_146618`](../feature_select/results/group_repeats/run_20260913_205201_146618):
sus métricas, normalizadores, 100 grupos de train por ronda y 50 grupos
comunes de validación. **Este comando entrena solo cinco modelos nuevos**, uno
con flags por ronda. Son GRU de una capa, 32 unidades, salidas `t0`/`t1` y
112 entradas originales más 16 flags. Mantienen las 2 épocas, chunks de 128,
semilla de modelo 2026 y Weighted Pearson de la referencia. Los 500 grupos de
train de esas rondas son disjuntos. Los flags se calculan sobre valores sin
normalizar, con `np.isclose(..., rtol=0, atol=2e-7)` contra el límite convertido
a float32, y permanecen binarios. Las 112 originales usan exactamente el
normalizador guardado en el checkpoint base de su ronda. Se aplica lo mismo en
la inferencia fila a fila de
[inference.py](inference.py).

Desde la raíz del proyecto en PowerShell, ejecutar manualmente:

```powershell
.\.venv\Scripts\python.exe my_solution\model\a_boundary_flags\train_gru_five_rounds.py
```

Pruebas de generación de flags:

```powershell
.\.venv\Scripts\python.exe -m unittest my_solution.model.a_boundary_flags.test_flags
```

Si se interrumpe, usar `--resume RUTA_DEL_RUN` con los mismos parámetros. Se
puede elegir otra corrida base completa con `--reference-run RUTA`. El
`run_config.json` registra la referencia y sus grupos. Cada `round_XX` guarda
el checkpoint nuevo, métricas y `comparison.csv`. `summary.csv` compara
WP de `t0`, `t1` y promedio por ronda; `aggregate.json` registra la diferencia
promedio y cuántas rondas ganan los flags. El test oficial está oculto;
`valid.parquet` funciona como test local.
