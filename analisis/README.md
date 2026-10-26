# Scripts de medición

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
