# Exploración de los datos

Recorrido por el dataset del challenge y presentación para explicarlo.

- `eda_connectome.ipynb`: el notebook, ya ejecutado y con sus salidas.
- `Connectome_los_datos.pptx`: la presentación, con notas del orador en cada lámina.
- `figuras/` y `resultados.json`: los gráficos y números que genera el notebook y usa la presentación.
- `cache/`: resultados intermedios para no recalcular. No va al repositorio.

Para correrlo hacen falta `pandas`, `numpy`, `pyarrow` y `matplotlib`, y los parquets en `../../../datasets/` (el notebook busca la carpeta `datasets` hacia arriba). Hay que abrirlo desde esta carpeta.

La primera corrida tarda unos 15 minutos: unos 10 son del modelo lineal sobre todo valid. Las siguientes usan `cache/` y tardan unos 4. Para recalcular todo, borrá `cache/`.

**Ojo con la métrica:** `METRIC.md` y `docs/data_overview.md` del starter pack describen la versión vieja, por secuencia. La vigente es la global, publicada en <https://wundernn.io/connectome/docs/data_overview>.
