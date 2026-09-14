# Correlación temporal de train

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe my_solution\explore\feature_select\temporal_correlations.py --row-groups 100
```

Los 100 row groups seleccionados son 100 secuencias completas de 20.000 pasos: aproximadamente 2 millones de filas. La selección es reproducible con `--seed 2026`; `--first` usa los primeros grupos. Los CSV se guardan en `results/` junto con los grupos seleccionados y un resumen top 20 por target. El CSV completo incluye variables originales, lags y deltas; los CSV ordenados y el top 20 incluyen **solo lags y deltas**.

Para comprobar las pruebas sin ejecutar el análisis:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s my_solution\explore\feature_select -p "test_*.py" -v
```

Train completo contiene 10.607 secuencias y 212,14 millones de filas. Pearson y Spearman globales exactos sobre todas esas filas requieren mucha más memoria y tiempo; el valor predeterminado toma una muestra de secuencias completas.
