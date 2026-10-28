# Scripts de medición

**Auditoría 2026-10-26:** las conclusiones históricas de la tabla requieren las salvedades de
[`docs/AUDITORIA_2026-10-26.md`](../docs/AUDITORIA_2026-10-26.md). En particular, el ranking de precios
anonimizados no demuestra profundidad física, la cuaterna con b256 no es el último ZIP y algunos
resúmenes de predicciones llamados causales usan filas puntuadas de casi toda la secuencia.

Scripts nuevos, con rutas relativas al proyecto y `--output` explícito:

- `auditoria_20261026.py`: cache completo contra parquet, baseline fijo opcional, scores,
  mezclas con clipping correcto, intervalos por secuencia y calibración cruzada exploratoria.
- `auditoria_precios_20261026.py`: precios anonimizados, hipótesis de redundancia y comprobación
  del horizonte real de los resúmenes antiguos. Ejecutar después del anterior.

**Instrumentos (2026-10-26).** Usan `instrumentos_comun.py` (cache en `analisis/_cache`). Correr primero
`python analisis/instrumentos.py valid` y `... train`.

- `instrumentos.py`: el instrumento de cada secuencia por el cuanto del target. **Son 4**: 0.1254 / 0.2413 /
  0.4685 / 0.6569. Train 12/12/33/43 %, valid 21/19/23/37 %.
- `instrumentos_deteccion.py`: reconocerlo sin target, por valores exactos de `a7`, `i0_p10`, `i0_p14`,
  `i0_dp1`. 100 % desde la fila 1.
- `por_instrumento.py`: puntaje por instrumento (0.799 / 0.771 / 0.729 / **0.435**; el 3 pesa 47 %) y
  recalibración por instrumento (+0.0000).
- `cabeza_volcado.py <best_model.pt> <tag>` + `cabeza_por_instrumento.py <tag>`: capa de salida por
  instrumento sobre la LSTM congelada. Peor que una común (0.6794 contra 0.6815).
- `pool_modelos.py` + `seleccion_por_instrumento.py`: ensamble distinto por instrumento. Sobreajusta.
- `filas_puntuadas_regla.py`: cómo caen las filas con `is_scored` alrededor de los cambios del target, por
  instrumento. Vienen en bloques; en el 3 evitan las zonas con cambios (0,4 veces la densidad de base).
- `filas_puntuadas_vs_resto.py`: la LR en filas puntuadas contra no puntuadas (**0.678 contra 0.482**) y la
  pendiente target/predicción en cada grupo (1,01–1,06 contra 0,79 en el instrumento 3).
- `cabeza_filas_puntuadas.py` (necesita `hid_valid_LR.npz` de `cabeza_volcado.py`): capa de salida reajustada
  con las filas puntuadas de valid, mitad y mitad. −0.0010 / +0.0001: nada.
- `residuo_microestructura.py <instrumento> <secuencias>`: 14 rasgos de microestructura contra el residuo de la
  LR. Todos ≤ 0,05; sumarlos empeora.
- `inst3_estados.py`: predicción media del ensamble por estado del target en el instrumento 3 y oráculos. La red
  distingue A de B pero no anticipa movimientos; saberlos llevaría el instrumento a 0.955.
- `inst3_anticipar_movimientos.py <train> <valid>`: AUC de "se mueve o no" dentro de cada estado. LightGBM 0.645 /
  0.617 contra 0.555 / 0.534 de la LR.
- `inst3_estructural_lgb.py <train> <valid> <rondas>`: predictor óptimo desde las probabilidades de los 5 estados.
  0.380 solo; mezclado no gana de forma estable.
- `inst3_apilado_lgb.py <valid> <rondas>` (necesita `hid_train_LR.npz` y `hid_valid_LR.npz`): lo mismo sobre el
  estado interno de la LR. 0.438 contra 0.458 de la LR sola.
- `ruido_leaderboard.py` (necesita `member_scored.npz`, las predicciones de cada miembro en las filas puntuadas):
  cuánto se mueve la diferencia entre dos entregas según cuántas secuencias tenga el test. Para S10 − S7: desvío
  0,0022 con 100 secuencias, 0,0015 con 200, 0,0010 con 400.

Los análisis que están detrás de los hallazgos de `EXPERIMENTS.md`. Corren **en la laptop, sin GPU**, sobre
predicciones de valid ya guardadas: no entrenan nada, solo miden.

**Todos necesitan dos cosas que no están en el repo** (son archivos grandes, excluidos por `.gitignore`):

1. `model/submission/results/valid_scored_cache.npz` — las 3.528.273 filas puntuadas de valid con sus targets.
2. Los `valid_preds_*.npz` de cada modelo, que se bajan de Kaggle con `kaggle kernels output <slug> -p <dir>`.

Las rutas están escritas arriba de cada script, en constantes (`CH`, `R`, `SC`). **Hay que ajustarlas**: `SC`
apunta a un directorio temporal de la sesión en la que se corrieron.

| script | qué contesta | hallazgo |
|---|---|---|
| `test_libro2.py` | ¿las columnas del libro están en orden de profundidad? (12 secuencias, 2 instrumentos) | **El orden es fijo y permutado: mejor bid = `i0_p10` (99,9 %), mejor ask = `i0_p14` (99,8 %).** 12/12 secuencias coinciden en precios |
| `test_libro.py` | lo mismo en detalle sobre una secuencia | el 'desequilibrio del tope' de `cross_features()` correlaciona +0,09 con el real |
| `ensamble_todo2.py` | ¿hay alguna combinación de los 38 modelos del proyecto que le gane a nuestra cuaterna? | **No.** Con validación partida: −0.0001 sin presupuesto, −0.0004 en 55 min. Y el presupuesto se mide en minutos: nos sobran ~20 de los 60 |
| `ensamble_baratos.py` | ¿conviene reemplazar una red por XGBoost o por la ESN? | las 8 opciones negativas (−0.0009 a −0.0029), medido en las 200 secuencias donde los tres existen |
| `ensamble_todo.py` | la primera versión de la búsqueda | **quedó como ejemplo de sesgo de selección**: daba +0.0009 sin partir los datos. Usar `ensamble_todo2.py` |
| `s23_mide.py` | ¿la normalización adaptativa aporta al ensamble? | **r = 0.9832, la correlación más baja que logramos, y +0.0001.** Las 70 cuaternas de los 8 modelos |
| `escala_paso4.py` | ¿se puede alcanzar el +0.047 de septiembre? | **No: es un oráculo conjunto y sus dos piezas dan R² = +0.003 cada una.** Correr este primero |
| `escala_paso3.py` | ¿de dónde sale el +0.047, y se predice la confianza por secuencia? | no es de escala (igualarla da −0.0566); **la confianza `r_s` sí se predice, R² = +0.57**, pero aplicarla empeora |
| `escala_paso2.py` | la escala del target por secuencia, aplicada a las predicciones | igualar la escala exacta **empeora** (−0.0566); confirma el +0.047 del oráculo factor+corrimiento |
| `escala_predecible.py` | piloto rápido: ¿tiene señal la escala del target? | R² 0.20–0.26 con la secuencia completa, 0.08–0.14 causal |
| `s24_cuaterna.py` | ¿alguna cuaterna de los 7 modelos le gana a la nuestra? ¿se cumple `s > S·r`? | **La regla `s > S·r` no sirve: los 3 nuevos la pasan con margen y la ganancia real es +0.0000** |
| `s24_final.py` | ¿qué palanca de diversidad baja la correlación entre modelos? | datos y maestros no la mueven; la arquitectura sí (0.977 contra 0.985) |
| `s24_analisis.py` | la primera versión, con dos modelos de tres | quedó para referencia; `s24_final.py` la reemplaza |
| `destilar_mato_ensamble.py` | ¿destilar de los mismos maestros nos achicó la ganancia del ensamble? | destilados 0.9854 → +0.0021 · sin destilar 0.9635 → +0.0051 |
| `esn.py` | ¿una Echo State Network aporta algo distinto? | correlación 0.8756, la más baja que logramos, y ganancia −0.0019 |
| `test_escala.py` | ¿cuánto cambia la escala de las features entre secuencias? | 97 de 112 cambian más de 2x (mediana 3,87x) → motiva el sprint 23 |
| `test_residuo.py` | ¿sirve un modelo que corrija el error del ensamble? | correlación −0.14 con el residuo, +0.0003 |
| `test_calentamiento.py` | ¿las 99 filas de warm-up son un problema? | no |
| `test_i1.py` | ¿se puede explotar mejor el instrumento correlacionado `i1`? | no |
