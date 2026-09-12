# Perfil de las 112 variables y los targets

Desde la carpeta principal del proyecto:

```powershell
.\.venv\Scripts\python.exe my_solution\explore\variables\profile_variables.py --row-groups 10
```

El comando elige 10 row groups de `train.parquet` al azar, de forma
reproducible con semilla 2026. Perfila las **112 variables de entrada** y los
targets **`t0` y `t1`**. No incluye `seq_ix` ni `step_in_seq`.

Por defecto se muestrean como máximo 20.000 filas entre todos los grupos,
repartidas casi por igual. El reporte describe **esa muestra**, no todas las
filas de los grupos elegidos. Puedes cambiar el límite y la semilla:

```powershell
.\.venv\Scripts\python.exe my_solution\explore\variables\profile_variables.py --row-groups 50 --max-rows 30000 --seed 123
```

Para usar los primeros N grupos, agrega `--first`. Para añadir la matriz de
correlación Pearson, agrega `--pearson`:

```powershell
.\.venv\Scripts\python.exe my_solution\explore\variables\profile_variables.py --row-groups 3 --max-rows 3000 --first --pearson
```

Cada ejecución crea una carpeta nueva en `results/runs/` con
`variables_profile.html` y `sample_metadata.json`. El JSON registra exactamente
qué row groups y cuántas filas de cada uno se usaron. Si `--max-rows` es menor
que la cantidad de grupos, el comando falla para evitar dejar grupos sin
representación.

El modo `--pearson` mantiene la configuración ligera de YData y evita las
interacciones completas, que pueden consumir mucho tiempo y memoria con 112
variables.

`ydata-profiling` ya está instalado en el entorno virtual actual. Si cambias
de entorno y falta, instálalo allí con:

```powershell
python -m pip install ydata-profiling
```

Este script no modifica `requirements.txt` ni los archivos de datos.
