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
| `s24_cuaterna.py` | ¿alguna cuaterna de los 7 modelos le gana a la nuestra? ¿se cumple `s > S·r`? | **La regla `s > S·r` no sirve: los 3 nuevos la pasan con margen y la ganancia real es +0.0000** |
| `s24_final.py` | ¿qué palanca de diversidad baja la correlación entre modelos? | datos y maestros no la mueven; la arquitectura sí (0.977 contra 0.985) |
| `s24_analisis.py` | la primera versión, con dos modelos de tres | quedó para referencia; `s24_final.py` la reemplaza |
| `destilar_mato_ensamble.py` | ¿destilar de los mismos maestros nos achicó la ganancia del ensamble? | destilados 0.9854 → +0.0021 · sin destilar 0.9635 → +0.0051 |
| `esn.py` | ¿una Echo State Network aporta algo distinto? | correlación 0.8756, la más baja que logramos, y ganancia −0.0019 |
| `test_escala.py` | ¿cuánto cambia la escala de las features entre secuencias? | 97 de 112 cambian más de 2x (mediana 3,87x) → motiva el sprint 23 |
| `test_residuo.py` | ¿sirve un modelo que corrija el error del ensamble? | correlación −0.14 con el residuo, +0.0003 |
| `test_calentamiento.py` | ¿las 99 filas de warm-up son un problema? | no |
| `test_i1.py` | ¿se puede explotar mejor el instrumento correlacionado `i1`? | no |
