# Prompt de transferencia — Alpha Connectome ML Competition

> **Auditoría del 2026-10-26:** leer primero [AUDITORIA_2026-10-26.md](AUDITORIA_2026-10-26.md).
> Corrige referencias de este documento: baseline global completo = **0,617052**, no 0,589595;
> el ZIP local asociado a la última entrega contiene **mask SWA + LR SWA + LA best + long best,
> cuatro LSTM 128×2**, no una de 256. La proyección de 41m52s no corresponde a la cuaterna con b256.
> El supuesto orden físico del libro no está demostrado. Los resúmenes de primeras 2.000
> predicciones de escala_paso3/4 usan filas puntuadas, llegando hasta el final de la secuencia;
> no son causales tempranos. La validación partida conserva selección previa sobre TODO valid.
> Se conservan abajo los resultados históricos; las correcciones e incertidumbres están en la auditoría.

> Documento de entrada para un asistente que retoma este proyecto sin historia previa. Describe la tarea, los
> datos, la métrica, la infraestructura, el código escrito, las técnicas implementadas y la ubicación exacta de
> cada archivo. Los resultados se dan como mediciones, para que quien lea decida qué hacer con ellas.

---

## 1. La competencia

- **Nombre:** Alpha Connectome ML Competition, de **Wunder Fund** — `wundernn.io/connectome`.
- **Arranque del trabajo:** 2026-09-11.
- **Cierre:** **2026-11-15 23:59 UTC**. Anuncio de ganadores: 2026-12-01.
- **Premios:** 13.600 USDT repartidos entre los 8 primeros.
- **Verificación para premiar:** código de entrenamiento reproducible + informe de 2 a 8 páginas + una
  videollamada. La documentación de la competencia está en `wnn_connectome_starterpack/docs/`:
  `prizes.md`, `rules.md`, `timeline.md`, `submission_guide.md`, `faq.md`, `data_overview.md`,
  `quick_start.md`, `get_help.md`.

**Raíz del proyecto** (todas las rutas de este documento son relativas a ella salvo que se indique otra cosa):

```
C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack\
```

El trabajo propio vive en el subdirectorio `challenge/`, que es **un repo git independiente**:

- remoto: `https://github.com/ElNelsonn/challenge`, rama `main`
- `challenge/.gitignore` excluye `*.npz`, `*.pt`, `*.parquet`, `__pycache__/`, `submissions/*/`, lo que deja el
  repo en ~50 MB en vez de 6,7 GB. **Los pesos y las predicciones NO están en el repo**: se regeneran o se
  bajan de Kaggle.

---

## 2. La tarea, formalmente

Predecir **online y de forma causal** (fila por fila, sin ver el futuro) dos targets `t0` y `t1` del
instrumento `i0`, a partir de 112 features `float32` que describen el libro de órdenes de dos instrumentos.

- Los datos vienen en **secuencias de 20.000 filas**. Una secuencia = un `row group` del parquet.
- **Filas 0 a 98: calentamiento** (warm-up). No se puntúan y sirven para que el modelo cargue estado.
- **Filas 99 a 19.999: requieren dos predicciones finitas.** Una predicción `NaN` o infinita invalida la
  entrega.
- El estado recurrente **se reinicia en cada secuencia** (`seq_ix`). Las `seq_ix` vienen **shuffleadas**: el
  valor de `seq_ix` no es el índice del row group, y hay que construir el mapeo si se lo necesita.
- La columna `is_scored` marca qué filas entran en la métrica.

### Estructura de las 112 features

Por cada instrumento (`i0`, `i1`), 52 columnas:

| nombre | cantidad | significado |
|---|---|---|
| `iK_p0..iK_p10` | 11 | precios del lado **bid** |
| `iK_p11..iK_p21` | 11 | precios del lado **ask** |
| `iK_v0..iK_v10` | 11 | volúmenes del lado bid |
| `iK_v11..iK_v21` | 11 | volúmenes del lado ask |
| `iK_dp0..iK_dp3` | 4 | deltas de precio |
| `iK_dv0..iK_dv3` | 4 | deltas de volumen |

Más 8 columnas auxiliares `a0..a7`. Total: 52 × 2 + 8 = **112**.

El parquet tiene **118 columnas** = 112 features + 2 targets (`t0`, `t1`) + 4 de control (`seq_ix`,
`step_in_seq`, `need_prediction`, `is_scored`). No hay columnas escondidas.

La lista canónica y en orden está en `wnn_connectome_starterpack/utils.py`, constante `FEATURE_COLUMNS`.

### Propiedades medidas de las features

- **Vienen normalizadas columna por columna**, en un rango aproximado de `[-2,4, +2,4]`.
- **Valor centinela `-5.1993`** en las columnas `i1_dv*` y en `a5`, `a6`, `a7`. El código lo detecta con
  `np.isclose(x, -5.1993, atol=2e-4)`.
- La escala de cada feature **varía entre secuencias**: 97 de las 112 cambian de escala más de 2x entre
  secuencias (mediana 3,87x, las peores ~10x). Medido en `challenge/analisis/test_escala.py`.

### El orden de profundidad del libro está PERMUTADO (medido 2026-10-26)

Las columnas de precio **no están en orden de profundidad por su índice**, pero sí tienen un **orden fijo y
consistente**: verificado sobre 12 secuencias de `valid` y sobre los dos instrumentos, las 12 dan exactamente
el mismo orden, y `i1` da el mismo que `i0` (desvío máximo del ranking entre secuencias: 0,09 para bids, 0,01
para asks).

**Bids, del mejor (precio más alto) al más profundo:**

```
i0_p10, i0_p5, i0_p2, i0_p1, i0_p7, i0_p4, i0_p6, i0_p9, i0_p8, i0_p0, i0_p3
```

**Asks, del mejor (precio más bajo) al más profundo:**

```
i0_p14, i0_p21, i0_p11, i0_p15, i0_p17, i0_p12, i0_p13, i0_p18, i0_p20, i0_p16, i0_p19
```

- **mejor bid = `i0_p10`** en el 99,9 % de las filas
- **mejor ask = `i0_p14`** en el 99,8 % de las filas
- lo mismo aplica a `i1` con los índices equivalentes

**Consecuencia teórica.** Como cada columna se normalizó por separado, el orden de los valores normalizados no
coincide con el orden de los precios reales. Una capa lineal **sí** puede reconstruir *diferencias* de precio
(aprende las escalas; el offset constante lo absorbe el bias), pero **no** puede reconstruir `min`, `max` ni
`argmax` sobre los niveles, que es lo que hace falta para el mejor nivel, el spread verdadero, el microprecio y
la concentración de volumen.

**La función `cross_features()` en `challenge/model/official_gpu/train_official_gpu_v1.py` (líneas ~66-97) usa
el índice 0 como mejor nivel**, es decir `i0_p0` (que en el orden real es el décimo de once bids) y `i0_p11`
(tercer mejor ask). Su "desequilibrio del tope" correlaciona **+0,09** en promedio con el desequilibrio real
del tope del libro, y es **negativo en el 25 %** de las secuencias.

**Sin verificar todavía:** si `iK_v{j}` corresponde al mismo nivel de profundidad que `iK_p{j}`. Los volúmenes
**no** tienen orden consistente entre secuencias (1 a 3 de 12), lo cual es esperable porque los volúmenes
cambian, pero la correspondencia índice-precio ↔ índice-volumen hay que comprobarla antes de construir
features de volumen por profundidad.

Scripts: `challenge/analisis/test_libro.py` (una secuencia, en detalle) y
`challenge/analisis/test_libro2.py` (12 secuencias y los dos instrumentos). *Si no están en `analisis/`, están
en el scratchpad de la sesión; copiarlos.*

### Propiedades medidas de los targets

- **Cuantizados**: múltiplos de ~0,1254, y la escala del cuanto varía por secuencia.
- `corr(t0, t1) = −0,72` global.
- **Nunca ambos positivos en la misma fila.** Es una restricción estructural dura de los datos.
- Outliers hasta ±125. El 3,1 % cae fuera de ±2. El 21,9 % es aproximadamente 0.
- Desvío ponderado por secuencia: mediana 1,181, con p90/p10 ≈ 1,62x.

---

## 3. La métrica

**Weighted Pearson GLOBAL.** Importante: los organizadores **cambiaron la métrica el 2026-09-16**. La versión
vigente es:

1. Tomar **todas** las filas puntuadas de todo el dataset juntas (no por secuencia).
2. Recortar los targets a ±2: `y = clip(y, -2, 2)`.
3. Peso de cada fila: `w = |y|`.
4. Calcular la correlación de Pearson ponderada por `w`, **por target**:

```
WP_j = Σw(y-ȳ_w)(q-q̄_w) / sqrt( Σw(y-ȳ_w)² · Σw(q-q̄_w)² )
```

5. La métrica final es el **promedio de `WP_t0` y `WP_t1`**.

Las predicciones también se recortan a ±2 antes de puntuar en toda nuestra tooling.

**Consecuencias teóricas de esta métrica:**
- La **escala de las predicciones entre secuencias importa**, porque todo se promedia junto.
- Las filas con `|y|` grande **dominan** el resultado.
- La métrica es invariante a una transformación afín **global**, pero **no** a transformaciones afines
  **por secuencia**.

### Implementaciones en el código

| dónde | qué |
|---|---|
| `wnn_connectome_starterpack/utils.py` → `weighted_pearson` | la correlación ponderada base |
| `wnn_connectome_starterpack/utils.py` → `BlockAccumulator` | **calcula la métrica VIEJA** (por secuencia). No usar como métrica oficial |
| `wnn_connectome_starterpack/utils.py` → `ScorerStepByStep` | scorer local fila por fila |
| `challenge/model/official_gpu/train_official_gpu_v1.py` → `GlobalWP` | la métrica nueva, global |
| `challenge/model/lgbm_deltas/train_lgbm_deltas_v1.py` → `GlobalWP` | la misma, para los árboles |
| `wnn_connectome_starterpack/METRIC.md` | la definición oficial de los organizadores |

**Referencia de comparación usada en todo el proyecto:** los **mismos 10 grupos fijos de valid**, que se
obtienen con `--valid-groups 10 --seed 2026`. El baseline oficial da **0,6498** ahí con la métrica nueva
(0,568 con la vieja), y **0,589595** sobre todo valid.

---

## 4. Los datos en disco

```
wnn_connectome_starterpack/datasets/train.parquet        29,2 GB   10.607 secuencias
wnn_connectome_starterpack/datasets/valid.parquet         5,6 GB    1.873 secuencias
wnn_connectome_starterpack/datasets/valid_mask.parquet   92   MB
```

- Compresión ZSTD. **Un row group = una secuencia de 20.000 filas.**
- Lectura por secuencia con `pyarrow`:
  `pq.ParquetFile(ruta).read_row_group(i, columns=[...], use_threads=False)`
- Velocidad de lectura local medida: ~0,007 s por secuencia con 3 columnas, ~0,04-0,05 s con las 112.
- Las filas puntuadas de `valid` son **3.528.273** de 1.873 secuencias.
- El test tiene aproximadamente **37,46 M filas**.

**Cache de evaluación** (no está en git, hay que regenerarlo):
`challenge/model/submission/results/valid_scored_cache.npz`, con las claves `seq`, `row`, `y` — las 3.528.273
filas puntuadas de valid con sus targets. **Todos los scripts de análisis lo usan.**

---

## 5. La restricción de inferencia

Es la restricción que gobierna el diseño del modelo final:

- **1 vCPU**, sin GPU.
- **60 minutos** para las ~37,46 M filas del test → **~96 µs por fila**.
- El zip de la entrega debe pesar **≤ 20 MB**.
- La entrega es `solution.py` + `weights.npz`.

**Costos medidos** (calibrados con una submission real que tardó **41 min 52 s** con 3 × LSTM 128x2 +
1 × LSTM 256x2):

| arquitectura | costo |
|---|---|
| LSTM 128x2 | ~6,0 min |
| GRU 128x2 | ~4,5 min |
| LSTM 192x2 | ~12,4 min |
| LSTM 256x2 | ~21,2 min |
| TCN (la que probamos) | ~1,3 × una LSTM 128x2 |

El costo escala con las operaciones del kernel: matvec de entrada `112 × G × H` más recurrencia `H × G × H`
por capa, con `G = 4H` para LSTM y `G = 3H` para GRU. **El presupuesto se mide en minutos, no en cantidad de
modelos.** El ensamble actual consume ~40 de los 60 minutos.

**Comparación de kernels medida:**
- baseline con ONNX Runtime: **82,7 µs/fila**
- la misma GRU reimplementada en numba: **27,7 µs/fila** (diferencia máxima 2e-5 contra PyTorch).
  La versión vigente del kernel está en `challenge/model/submission/solution.py`; las mediciones de
  velocidad, en `challenge/model/submission/kernel_bench/`.

---

## 6. El código

### 6.1 El entrenador principal

**`challenge/model/official_gpu/train_official_gpu_v1.py`**

El mismo script corre en CPU (laptop) y en GPU (Kaggle); detecta `cuda` solo. Es el archivo central del
proyecto. Qué hace:

- Entrena **B secuencias en paralelo** por batch, con **BPTT truncado** en chunks.
- Loss alineada a la métrica (targets a ±2, peso `|y|`), o `-WeightedPearson` directa con `--loss wp`.
- Gradient clipping, learning rate configurable con decay y schedule, checkpoints por época, `--resume`.
- Evalúa con la métrica global en grupos fijos de valid.
- **SWA** (Stochastic Weight Averaging) desde la época `--swa-start`.
- **Destilación**: `--distill-teachers` toma uno o más checkpoints como maestros y entrena contra sus salidas;
  `--distill-cache` precalcula los soft targets una vez en fp16 en vez de correr los maestros en cada chunk.
- Guarda `checkpoint.pt` (config completa), `best_model.pt` (config reducida + `pipeline`), `metrics.json`, y
  con `--save-valid-preds` también `valid_preds_best.npz` y `valid_preds_swa.npz`.

**Clases y funciones clave:**

| nombre | línea aprox. | qué hace |
|---|---|---|
| `ParquetGroups` | 173 | lee secuencias; `read(i)` devuelve `(seq_ix, need, x, y, scored)`, con `x` = features |
| `FeaturePipeline` | 205 | construye las features y guarda su normalizador |
| `FeaturePipeline.raw_to_features` | 236 | aplica centinelas, cross, EMA y lags |
| `FeaturePipeline.transform` | 270 | normaliza (global o adaptativa) |
| `FeaturePipeline.state` | 294 | serializa: `top_k`, `sentinel_flags`, `cross`, `ema`, `ema_plus`, `adapt_span`, `specs`, `mean`, `std`, `names` |
| `cross_features` | 66 | las 20 features cruzadas (ver §2 sobre la permutación de profundidad) |
| `GlobalWP` | — | la métrica global |
| `select_groups` | ~199 | elige los mismos grupos que `EXPERIMENTS.md` |

**Detalle importante para exportar a inferencia:** `best_model.pt` guarda una **config reducida** con solo
`cell`, `hidden`, `layers`, `arch`, `tcn_*`, `slow_*`. Los parámetros del pipeline —entre ellos `adapt_span`—
**no están en esa config**: viven en `best_model.pt["pipeline"]`. `checkpoint.pt` sí trae la config completa
(se construye como `{**vars(args), ...}`, así que todo flag de argparse aparece ahí y en `metrics.json`).

**Flags disponibles** (lista completa, extraída del argparse):

```
--adapt-span --arch --aux-targets --aux-weight --batch --cell --chunk --clip
--cross-features --distill-cache --distill-teachers --distill-weight --dropout
--ema-features --ema-plus --epochs --eval-batch --eval-every --exclude-train-groups
--gwp-weight --hidden --init-from --layers --loss --lr --lr-decay --lr-min
--lr-schedule --mask-coef --mask-power --model-seed --norm-rows --output-dir
--ranking --resume --resume-cache --resume-epochs --resume-swa-start --run-name
--save-epochs --save-valid-preds --seed --sentinel-flags --seqwp-decay
--slow-every --slow-hidden --swa-start --target-cd --tcn-channels --tcn-kernel
--tcn-layers --threads --top-k --train-groups --train-half --train-parquet
--valid-groups --valid-parquet --weight-decay --weight-power --wmse-anchor --workers
```

Ejemplo de uso local:
```bash
python model/official_gpu/train_official_gpu_v1.py --train-groups 500 --epochs 3
```

Ejemplo de uso en Kaggle:
```bash
python train_official_gpu_v1.py \
  --train-parquet /kaggle/input/.../train_part1.parquet /kaggle/input/.../train_part2.parquet \
  --valid-parquet /kaggle/input/.../valid.parquet \
  --train-groups all --valid-groups all --epochs 10 --swa-start 7 \
  --cell lstm --hidden 128 --layers 2 --lr 5e-4 --save-valid-preds \
  --output-dir /kaggle/working/results --run-name mi_corrida
```

### 6.2 Los árboles

**`challenge/model/lgbm_deltas/train_lgbm_deltas_v1.py`**

LightGBM y XGBoost sobre features tabulares con lags y deltas. Tiene un adaptador `ModeloXGB` para usar
XGBoost con la misma interfaz, seleccionable con `--lib xgboost`.

Flags:
```
--bagging-fraction --ema-features --ens-preds --feature-fraction --gru-model
--importance --lags --lambda-l2 --learning-rate --lib --max-depth
--min-data-in-leaf --n-estimators --num-leaves --output-dir --rows-per-group
--run-name --seed --threads --top-features --train-groups --train-parquet
--valid-groups --valid-parquet
```

**Detalle de memoria que importa:** `xgb.DMatrix` guarda 8 bytes por número incluso con datos densos, lo que
para esta matriz son ~16 GB y provoca OOM en una máquina de 33,7 GB. La solución usada es
**`xgb.QuantileDMatrix`**, que guarda 1 byte por número.

Las predicciones de árboles se guardan como `valid_predictions.npz` con **una clave por `seq_ix`**, cada una
de forma `(20000, 2)` — formato distinto al de las redes.

### 6.3 La inferencia / entrega

| archivo | qué hace |
|---|---|
| `challenge/model/submission/solution.py` | el kernel de inferencia en **numba**, con pesos **int8**. Soporta GRU y LSTM, varias redes en ensamble, normalizador propio por red (`MEAN`, `STD`, `CLIP`) |
| `challenge/model/submission/build_submission.py` | arma el zip de entrega a partir de checkpoints |
| `challenge/model/submission/eval_ensemble.py` | evalúa combinaciones de modelos |
| `challenge/model/submission/eval_full_valid.py` | evalúa sobre todo valid |
| `challenge/model/submission/combine_local.py` | tabla de ensambles local |
| `challenge/model/submission/lgbm_numba.py` | inferencia de árboles en numba |
| `challenge/model/submission/kernel_bench/` | mediciones de velocidad del kernel |
| `challenge/model/submission/results/` | cache de evaluación, incluido `valid_scored_cache.npz` |

**Cómo normaliza `solution.py`** (importante si se cambia el pipeline de features): la normalización se aplica
**en float32**, por red, en el bucle principal:

```python
v = (x[j] - MEAN[m, j]) / STD[m, j]
v = clip(v, -CLIP[m], CLIP[m])
z[j] = v
```

La cuantización int8 es de los **pesos** (`W1T`, `R1T` con sus escalas `S_W1`, `S_R1`), no de las entradas.
`MEAN` y `STD` son arrays fijos de forma `(N_MODELS, 112)`. **La normalización adaptativa por secuencia no está
implementada en este kernel**: habría que agregar estado EMA por feature y verificar paridad con PyTorch.

### 6.4 Otros módulos

| ruta | qué hay |
|---|---|
| `challenge/model/feature_select/features.py` | `load_rankings`, `select_temporal_features`: lags y deltas rankeados, usados por `--top-k` |
| `challenge/model/official_gpu/mask_coef.json` | coeficientes del "modelo de la máscara" |
| `challenge/model/a_boundary_flags/` | exploración de las columnas `a*` y sus bordes |
| `challenge/model/model_target/` | exploración de la estructura de los targets |
| `challenge/models_first_try/` | los primeros modelos del proyecto (`first_try_torch.py` y compañía) |
| `challenge/explore/` | notebooks y scripts de exploración de datos |
| `wnn_connectome_starterpack/baseline/` | `baseline.onnx` (GRU 128x2, solo inferencia), su `solution.py` y `baseline_submission.zip` |

---

## 7. Infraestructura de entrenamiento: Kaggle

Todo el entrenamiento con GPU se hace en **Kaggle**, con GPU T4 gratuita.

- **Cuenta activa:** `joaqfio`. Hay una segunda cuenta, `joaquinfiorina`, que es la dueña de los datasets.
- **Datasets subidos** (compartidos entre las cuentas):
  - `joaquinfiorina/connectome-train-part1`
  - `joaquinfiorina/connectome-train-part2`
  - `joaquinfiorina/connectome-valid`
  - `joaquinfiorina/connectome-models` (los checkpoints, para usarlos como maestros)
- **Cuota:** 30 h de GPU por semana, que se reponen los sábados. Máximo **2 sesiones de GPU simultáneas**.
- **Límite por corrida: 12 horas.** Si se pasa, Kaggle la cancela (`CANCEL_ACKNOWLEDGED`) y se pierde todo lo
  no guardado.

### Cómo se suben los notebooks

```bash
kaggle kernels push   -p <directorio>              # subir / relanzar
kaggle kernels status joaqfio/<slug>               # estado
kaggle kernels output joaqfio/<slug> -p <destino>  # bajar resultados y log
kaggle quota                                       # cuota restante
```

Cada directorio de subida contiene el `.ipynb` y un `kernel-metadata.json` así:

```json
{
 "id": "joaqfio/connectome-s23b-adapt8000",
 "title": "connectome_s23b_adapt8000",
 "code_file": "connectome_sprint23_adapt8000.ipynb",
 "language": "python", "kernel_type": "notebook", "is_private": true,
 "enable_gpu": true, "enable_tpu": false, "enable_internet": true, "keywords": [],
 "dataset_sources": ["joaquinfiorina/connectome-models",
                     "joaquinfiorina/connectome-train-part1",
                     "joaquinfiorina/connectome-train-part2",
                     "joaquinfiorina/connectome-valid"],
 "kernel_sources": [], "competition_sources": [], "model_sources": [],
 "machine_shape": "NvidiaTeslaT4"
}
```

### Los generadores de notebooks

Los notebooks no se escriben a mano: los genera un script por sprint, en
`challenge/notebooks_kaggle/make_sprint{17,18,19,20,21,22,23,24}.py`. Cada uno escribe uno o más `.ipynb` con
la misma estructura: celda que escribe `train_official_gpu_v1.py` con `%%writefile`, celda que localiza los
inputs, **prueba de humo**, y celda de entrenamiento real.

### Trampas operativas conocidas (hechos, para no tropezar de nuevo)

1. **Rutas de los datasets.** Kaggle monta el mismo dataset en dos rutas. Los generadores traen una función
   `buscar(nombre, espera=600)` que busca el archivo bajo `/kaggle/input/**`, prefiere las rutas que contienen
   `/datasets/` y espera hasta 600 s a que el dataset se monte.

2. **Velocidad de lectura variable.** La misma operación de precalcular soft targets tardó **37 minutos** en
   una corrida y **695 minutos** en otra con el **mismo código**: depende de la máquina que toque y de si el
   dataset quedó cacheado en su disco local. Los generadores desde el sprint 24 incluyen una **guarda de
   velocidad** que mide 15 secuencias al arrancar y aborta si tarda más de 1 s cada una:

   ```python
   import pyarrow.parquet as pq
   pf = pq.ParquetFile(TRAIN.split()[0])
   t0 = time.time()
   for g in range(15):
       pf.read_row_group(g, use_threads=False)
   seg = (time.time() - t0) / 15
   assert seg < 1.0, f'LECTURA LENTA: {seg:.1f} s por secuencia'
   ```

   **Limitación conocida de esta guarda:** mide el arranque. En la corrida de 695 minutos midió 0,05 s por
   secuencia y la dejó pasar, porque el frenazo apareció más tarde. Un control de presupuesto **dentro** del
   bucle de soft targets cubriría ese caso.

3. **Pushear a un kernel que quedó en estado ERROR da `409 Conflict`.** La solución usada es subir con un slug
   nuevo (`connectome-s24b-*`, `connectome-s23b-*`, etc.).

4. **Una cuenta nueva necesita verificación por teléfono** para que `enable_gpu: true` realmente use GPU. Sin
   eso la corrida arranca en CPU y el contador de cuota no se mueve.

5. **Las pruebas de humo tienen que verificarse contra archivos locales reales antes de subir.** Dos bugs
   históricos: usar la clave `columns` en vez de `names` en el estado del pipeline, y buscar `target_cd` en la
   config reducida de `best_model.pt` cuando vive en la de `checkpoint.pt`.

---

## 8. Técnicas implementadas, con su teoría

### 8.1 Destilación (knowledge distillation)

Un "alumno" se entrena contra las salidas continuas de uno o varios "maestros" ya entrenados, en vez de (o
además de) contra los targets reales. La señal del maestro es más suave y densa que el target, que acá está
cuantizado y tiene un 21,9 % de ceros.

Flags: `--distill-teachers <checkpoints...>`, `--distill-weight`, `--distill-cache <ruta.npz>`.
El cache precalcula las salidas de los maestros una vez en fp16 (ocupa ~849 MB para train completo) y evita
correr los maestros en cada chunk. **Es el paso más caro de una corrida** (ver trampa 2).

### 8.2 SWA (Stochastic Weight Averaging)

Promedia los pesos de las últimas épocas en vez de quedarse con el mejor checkpoint. Flag `--swa-start N`.
Produce un `valid_preds_swa.npz` aparte del `valid_preds_best.npz`.

### 8.3 BPTT truncado

Se propaga el gradiente sólo dentro de chunks de `--chunk` filas, llevando el estado oculto hacia adelante sin
gradiente entre chunks. Probados chunks de 256, 512, 1024, 2000 y 2048.

### 8.4 Loss alineada a la métrica

Tres opciones por `--loss`:
- `wmse`: error cuadrático con targets recortados a ±2 y peso `|y|`.
- `wp`: `-WeightedPearson` directo.
- `seqwp`: variante por secuencia con decaimiento (`--seqwp-decay`).

Además `--weight-power` y `--wmse-anchor` modulan el peso, y `--gwp-weight` mezcla la WP global en la loss.

### 8.5 Features derivadas

| flag | qué agrega |
|---|---|
| `--sentinel-flags` | columnas binarias que marcan dónde estaba el centinela `-5.1993`, y pone esas entradas en 0 |
| `--cross-features` | 20 features no lineales de la misma fila: desequilibrios, desvíos entre niveles, spread, mid, concentración, productos. **Construidas con el índice 0 como mejor nivel: ver §2** |
| `--ema-features` | medias móviles exponenciales de 40 features seleccionadas |
| `--ema-plus` | la versión extendida, 73 features. La lista extra es `EMA_EXTRA = ["i0_p2","i0_p9","i0_p15","i0_p20","i0_v4","i0_v6","i0_v9","i0_v12","i1_v4","i1_v9","i1_v20"]` |
| `--top-k N` | lags y deltas rankeados de `model/feature_select/features.py` |

### 8.6 Normalización adaptativa por secuencia

Flag `--adapt-span N`. En vez de normalizar con media y desvío globales, usa EMA causales **dentro de cada
secuencia**, inicializadas en los valores globales:

```python
def _normalizar_adaptativo(self, f):
    x = np.nan_to_num(f.astype(np.float64), nan=0.0)
    mu  = _ema_desde(x,     self.adapt_span, self.mean)
    seg = _ema_desde(x * x, self.adapt_span, self.mean**2 + self.std**2)
    var = np.maximum(seg - mu * mu, 1e-8)
    sd  = np.clip(np.sqrt(var), self.std * 0.1, self.std * 10.0)
    return (x - mu) / sd
```

Efecto medido en la laptop: la variación de escala que ve la red entre secuencias baja de **4,07x a 1,09x**.
Probado con spans 2000 y 8000. **No está implementado en `solution.py`** (ver §6.3).

### 8.7 Reparametrización de los targets

Flag `--target-cd`: entrena sobre `(t0+t1)/2` y `(t0−t1)/2` en vez de `t0` y `t1`, aprovechando que
`corr(t0,t1) = −0,72`. Al guardar, la transformación **se pliega en la última capa** para que el modelo
exportado tenga las dos salidas reales:

```python
if cfg.get("target_cd"):
    M = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=sd["head.weight"].device)
    sd["head.weight"] = M @ sd["head.weight"][:2]
    sd["head.bias"]   = M @ sd["head.bias"][:2]
```

También existen `--aux-targets` y `--aux-weight` para targets auxiliares.

### 8.8 Arquitecturas implementadas

| `--arch` / `--cell` | qué es |
|---|---|
| `--cell gru` / `--cell lstm` | recurrente estándar, `--hidden` × `--layers` |
| `--arch tcn` + `--tcn-channels/--tcn-layers/--tcn-kernel` | red convolucional causal dilatada |
| `--slow-hidden H --slow-every K` | una **segunda red lenta** que corre una vez cada K filas sobre el promedio de los estados de la rápida y devuelve un contexto a la salida. Las filas de una ventana usan el contexto de la ventana **anterior**, así que es causal. Costo medido en inferencia: **+0,5 %** (1.152 MACs/fila sobre 254.208) |
| Echo State Network | reservorio de 400 unidades con pesos recurrentes **aleatorios y fijos**, radio espectral 0,95, fuga 0,3; sólo se entrena la salida, con regresión lineal ridge. Implementación en `challenge/analisis/esn.py` |

### 8.9 Bagging y particiones de datos

- `--train-half` y `--exclude-train-groups N1 N2 ...` permiten entrenar sobre subconjuntos de secuencias.
  La exclusión está aseverada:

  ```python
  if args.exclude_train_groups:
      fuera = set(args.exclude_train_groups)
      previos = set(train_ids)
      train_ids = [i for i in train_ids if i not in fuera]
      sacados = len(previos) - len(train_ids)
      assert sacados == len(fuera & previos), "la exclusión no sacó lo que debía"
  ```

- El sprint 24 usó bagging del 70 % de las secuencias (7.424 de 10.607) con semilla propia por modelo.

### 8.10 Cuantización int8 para la entrega

Los pesos se guardan en int8 con una escala por matriz, y el kernel numba hace los productos en enteros. Es lo
que permite meter 4 redes en el presupuesto de 60 minutos y en el zip de 20 MB.

---

## 9. Los modelos entrenados y dónde están sus predicciones

Hay **58 archivos de predicciones de valid, de 38 corridas distintas**. Formato de las redes:
`valid_preds_best.npz` y `valid_preds_swa.npz`, con las claves `seq_ix` de forma `(1873,)` y `preds` de forma
`(1873, 20000, 2)` en `float16`.

### El ensamble que está entregado

Cuatro modelos, promedio simple:

| alias | ruta de las predicciones |
|---|---|
| `LR` | `challenge/model/official_gpu/results/sprint6/results/student_lr5e-4/valid_preds_best.npz` |
| `LA` | `challenge/model/official_gpu/results/sprint5/results/student_LA/valid_preds_best.npz` |
| `long` | `challenge/model/official_gpu/results/sprint11_ema/results/student_long/valid_preds_best.npz` |
| `b256` | `challenge/model/official_gpu/results/sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz` |

Es 3 × LSTM 128x2 + 1 × LSTM 256x2, todos destilados de los mismos 4 maestros. Tarda **41 min 52 s** en
inferencia.

### Resto de las corridas, bajo `challenge/model/official_gpu/results/`

```
_challenge_dl                     cpu_check_gru128x2_100s_2e     full_gru128x2_10ep
gru128_drop_seed2028              gru192_seed2027
s13_notebook092bd5e903            s13_notebook3d1591b8fb
s14_notebook3fdc883f92            s14_notebook6ca5ecbe4a         s14_notebook9b850709f8
s15_notebookbb3c35ea9f            s15_notebookfd82629297         (student_mask)
s16_notebook663a43c73a            s16_notebook7c4c66a5fa
sprint10_notebookd0e4cb9378       sprint10_notebookd45097018d    (student_seqwp)
sprint11_ema                      sprint11_long
sprint13_a  sprint13_b            sprint13_notebook092bd5e903    sprint13_notebook360eab04e2
sprint2  sprint3
sprint4_distill  sprint4_lstm  sprint4_x  sprint4_y               (los SIN destilar: l1,l2,l3,x1,x2,x3,y1,y2,y3)
sprint5  sprint5b  sprint6
sprint8_notebooka8398748db        sprint8_notebookdeb2463071
sprint9_connectome-sprint9-cross  sprint9_notebook89856cd57e
```

Nombres de corrida que aparecen dentro de esos directorios: `student_X`, `student_Y`, `student_Z`,
`student_LA`, `student_lr5e-4`, `student_long`, `student_ema`, `student_mask`, `student_cross`,
`student_seqwp`, `student_seqwp_dist`, `student_big192`, `student_big256`, `student_half1`, `student_half2`,
`student_chunk2000`, `l1`, `l2`, `l3`, `x1`, `x2`, `x3`, `y1`, `y2`, `y3`, `distill_128`, `gwp_128`,
`cosine_128`, `tcn40`, `tcn_big`, `tcn_full`, `tcn_lr15e3`, `tcn_lr5e4`.

### Corridas que están sólo en Kaggle

Estas no están en el repo; se bajan con `kaggle kernels output`:

| slug | contenido |
|---|---|
| `joaqfio/connectome-s23b-adapt8000` | normalización adaptativa, span 8000 |
| `joaqfio/connectome-s24b-div-a` | bagging 70 %, 2 maestros, LSTM lr 5e-4 |
| `joaqfio/connectome-s24b-div-c` | bagging 70 %, 2 maestros, LSTM lr 6e-4 |
| `joaqfio/connectome-s24b-div-d` | bagging 70 %, 2 maestros, **GRU** lr 6e-4 |
| `joaqfio/connectome-s24c-div-b` | quedó cancelado |
| `joaqfio/connectome-s23b-adapt2000` | quedó cancelado en el límite de 12 h |

### Árboles y ESN

| modelo | ruta | cobertura |
|---|---|---|
| LightGBM | `challenge/model/lgbm_deltas/results/lgbm_deltas_all/valid_predictions.npz` | 10 secuencias |
| XGBoost | (scratchpad) `k18/results/xgb_v1/valid_predictions.npz` | ~200 secuencias |
| ESN | (scratchpad) `esn_preds.npz`, claves `preds`, `seq`, `row` | ~200 secuencias |

**Nota de formato:** los árboles usan una clave por `seq_ix`; la ESN usa arrays planos `seq`/`row`/`preds`.
Las redes usan `seq_ix`/`preds`. Hay que convertir según el caso.

### Las entregas armadas

`challenge/submissions/` tiene los directorios y zips de cada submission construida:
`ens2_B_Cswa`, `ens3_A_B_Cswa`, `ens3_A_B_Cswa_f32`, `ens3_A_B_Cswa_int8`, `ens3_B_Cswa_Dswa`,
`ens3_B_Cswa_Dswa_int8`, `ens3_Dswa_Eswa_Gswa_int8`, `ens3_long_LR_LA_int8`, `ens3_y2_Eswa_X_int8`,
`ens4_Eswa_l1swa_LRswa_LA8_int8`. (Excluidos de git por `.gitignore`.)

---

## 10. Calibración y leaderboard

**Relación medida entre valid y el leaderboard, estable a lo largo de 7 submissions:**

```
LB ≈ valid − 0,0323
```

**Posiciones del leaderboard al 2026-10-21:**

| puesto | puntaje |
|---|---|
| 1 | 0,67380 |
| 50 | 0,66517 |
| 100 | 0,66256 |
| 200 | 0,65934 |
| 300 | 0,65685 |
| 400 | 0,65491 |
| nuestra entrega | **0,65400** (~410) |

Traducción a valid con la brecha de 0,0323, y lo que haría falta desde nuestro 0,6864:

| para el puesto | el LB pide | valid necesario | diferencia |
|---|---|---|---|
| ~400 | 0,65491 | 0,6872 | +0,0008 |
| ~300 | 0,65685 | 0,6892 | +0,0028 |
| ~200 | 0,65934 | 0,6916 | +0,0052 |
| ~100 | 0,66256 | 0,6949 | +0,0085 |
| ~50 | 0,66517 | 0,6975 | +0,0111 |

**Esos umbrales son del 2026-10-21 y suben con el tiempo**, porque los demás equipos siguen mejorando.

**Submissions usadas: 8.** El límite es 5 por día. Los equipos del top 100 llevan entre 20 y 193.

---

## 11. Los análisis hechos, y qué midió cada uno

Todos corren en la laptop sin GPU, sobre predicciones de valid ya guardadas. Están en
`challenge/analisis/`, con su `README.md`. **Necesitan dos cosas que no están en git:**
`challenge/model/submission/results/valid_scored_cache.npz` y los `valid_preds_*.npz` de cada modelo. Las
rutas están en constantes arriba de cada script (`CH`, `R`, `SC`) y **hay que ajustar `SC`**, que apunta a un
directorio temporal de sesión.

| script | qué mide | resultado medido |
|---|---|---|
| `test_libro.py` | si las columnas del libro están en orden de profundidad (una secuencia) | no lo están por índice; el mejor bid está en `i0_p10`, el mejor ask en `i0_p14` |
| `test_libro2.py` | lo mismo sobre 12 secuencias y los dos instrumentos | el orden es fijo: 12/12 coinciden para precios; los volúmenes no tienen orden consistente |
| `test_escala.py` | cuánto varía la escala de las features entre secuencias | 97 de 112 varían >2x, mediana 3,87x |
| `escala_predecible.py` | si la escala del target por secuencia tiene señal en las features | R² 0,20–0,26 con el resumen de toda la secuencia; 0,08–0,14 con las primeras 2.000 filas |
| `escala_paso2.py` | aplicar la escala del target a las predicciones | oráculo de escala exacta (σ_y/σ_q): −0,0566. Oráculo de factor + corrimiento: +0,0470 |
| `escala_paso3.py` | de dónde sale el +0,047, y si la confianza por secuencia se predice | solo factor −0,0314; solo corrimiento +0,0046; los dos +0,0470. **La confianza `r_s` se predice con R² = 0,51 (solo libro) y 0,57 (agregando el resumen de nuestras predicciones)**, fuera de muestra |
| `escala_paso4.py` | predecir factor y corrimiento por separado | R² = +0,003 cada uno. Factor oráculo + corrimiento predicho: −0,0316. Al revés: +0,0035 |
| `destilar_mato_ensamble.py` | correlación y ganancia del ensamble, destilados vs sin destilar | destilados r=0,9854 → promediar vale +0,0021; sin destilar r=0,9635 → +0,0051 |
| `s24_final.py` | qué palanca de diversidad baja la correlación entre modelos | datos y maestros distintos: 0,9849 vs 0,9860. Cambiar LSTM→GRU: 0,9769 |
| `s24_cuaterna.py` | si se cumple la regla `s > S·r` y si alguna cuaterna gana | los 3 modelos nuevos pasan la regla con margen +0,0027 a +0,0035; la mejor cuaterna de 7 da 0,6861 vs 0,6860 |
| `s23_mide.py` | si la normalización adaptativa aporta al ensamble | `adapt8000` solo 0,6801; **r con el ensamble 0,9832**, la más baja de las redes entrenadas; mejor cuaterna de 70: +0,0001 |
| `ensamble_todo.py` | primera versión de la búsqueda sobre los 38 modelos | +0,0009 **sin partir los datos** (sesgo de selección: usar `ensamble_todo2.py`) |
| `ensamble_todo2.py` | la búsqueda con costo real por arquitectura y validación partida por secuencia | mejor sin presupuesto −0,0001; mejor en 55 min −0,0004, contra el ensamble actual |
| `ensamble_baratos.py` | reemplazar una red del ensamble por XGBoost o por la ESN | las 8 opciones entre −0,0009 y −0,0029, medido en las 200 secuencias comunes |
| `esn.py` | la Echo State Network | 0,6099 sola, con 600 secuencias de train; r con el ensamble **0,8756**, la más baja de todos los modelos construidos |
| `test_residuo.py` | un modelo que corrija el error del ensamble | correlación −0,14 con el residuo; +0,0003 |
| `test_calentamiento.py` | si las 99 filas de calentamiento son un problema | no lo son |
| `test_i1.py` | explotar el instrumento correlacionado `i1` | sin ganancia medida |
| `s24_analisis.py` | versión previa de `s24_final.py`, con 2 modelos de 3 | queda como referencia |

### Correlaciones medidas entre modelos y el ensamble

| modelo | puntaje solo | r con el ensamble |
|---|---|---|
| `b256` | 0,6844 | 0,9955 |
| `long` | 0,6838 | 0,9948 |
| `LR` | 0,6834 | 0,9939 |
| `LA` | 0,6823 | 0,9939 |
| `div_a` (bagging 70 %) | 0,6829 | 0,9902 |
| `div_c` | 0,6819 | 0,9900 |
| `div_d` (GRU) | 0,6777 | 0,9835 |
| `adapt8000` (normalización adaptativa) | 0,6801 | 0,9832 |
| XGBoost | 0,6405 | 0,9277 |
| Echo State Network | 0,6099 | 0,8755 |

Los modelos **sin destilar** del sprint 4 correlacionan **0,9635** entre sí, contra **0,9860** de los cuatro
destilados.

### Sobre la regla `s > S·r`

Durante gran parte del proyecto se usó como criterio de decisión la desigualdad `s > S·r` (con `s` = puntaje
del modelo nuevo, `S` = puntaje del ensamble, `r` = correlación del nuevo con el ensamble), que sale de suponer
que los errores de los modelos son intercambiables. **Dos casos medidos en los que la regla da positivo y la
ganancia real no lo es:**

| caso | margen sobre `S·r` | ganancia real medida |
|---|---|---|
| Echo State Network | +0,0184 | −0,0019 |
| `div_a` / `div_c` / `div_d` | +0,0027 a +0,0035 | +0,0000 |

Conviene medir promediando y evaluando, no aplicando la regla.

---

## 12. Documentación del proyecto

| archivo | qué contiene |
|---|---|
| `challenge/EXPERIMENTS.md` | **el registro central**: tabla de 45 experimentos numerados con fecha, cambio, datos, puntaje, tiempo, máquina y notas, más secciones largas por hallazgo |
| `challenge/docs/DIARIO.md` | diario de trabajo día por día desde 2026-09-11 |
| `challenge/docs/HANDOFF.md` | estado del proyecto para alguien que lo retoma |
| `challenge/docs/PLAN.md` | el plan |
| `challenge/docs/informes/` | informes |
| `challenge/analisis/README.md` | índice de los scripts de medición y qué contesta cada uno |
| `challenge/model/official_gpu/README.md` | notas del entrenador |
| `wnn_connectome_starterpack/METRIC.md` | la métrica oficial |
| `wnn_connectome_starterpack/README.md` | el readme del starter pack |
| `wnn_connectome_starterpack/docs/*.md` | reglas, premios, cronograma, guía de entrega, FAQ, overview de datos |

---

## 13. Mapa de rutas, completo

```
wnn_connectome_starterpack/
├── utils.py                     FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP,
│                                weighted_pearson, BlockAccumulator (métrica VIEJA),
│                                ScorerStepByStep, validate_sequence, DataPoint
├── METRIC.md  README.md  requirements.txt
├── datasets/
│   ├── train.parquet            29,2 GB · 10.607 secuencias
│   ├── valid.parquet             5,6 GB ·  1.873 secuencias
│   └── valid_mask.parquet       92 MB
├── baseline/
│   ├── baseline.onnx            GRU 128x2, solo inferencia · 0,589595 en valid completo
│   ├── solution.py              el solution del baseline
│   └── baseline_submission.zip
├── docs/                        prizes, rules, timeline, submission_guide, faq,
│                                data_overview, quick_start, get_help
└── challenge/                   ← REPO GIT (github.com/ElNelsonn/challenge)
    ├── EXPERIMENTS.md
    ├── .gitignore               excluye *.npz, *.pt, *.parquet, __pycache__/, submissions/*/
    ├── docs/
    │   ├── DIARIO.md  HANDOFF.md  PLAN.md  PROMPT_HANDOFF.md (este archivo)
    │   └── informes/
    ├── analisis/                18 scripts de medición + README.md
    ├── model/
    │   ├── official_gpu/
    │   │   ├── train_official_gpu_v1.py      EL ENTRENADOR
    │   │   ├── mask_coef.json  README.md  cpu_check.log
    │   │   └── results/                      38 corridas con metrics.json,
    │   │                                     checkpoint.pt, best_model.pt,
    │   │                                     valid_preds_{best,swa}.npz
    │   ├── lgbm_deltas/
    │   │   ├── train_lgbm_deltas_v1.py       LightGBM + XGBoost
    │   │   └── results/lgbm_deltas_all/valid_predictions.npz
    │   ├── submission/
    │   │   ├── solution.py                   kernel numba int8 (LA ENTREGA)
    │   │   ├── build_submission.py           arma el zip
    │   │   ├── eval_ensemble.py  eval_full_valid.py  combine_local.py
    │   │   ├── lgbm_numba.py                 inferencia de árboles
    │   │   ├── kernel_bench/                 mediciones de velocidad
    │   │   └── results/valid_scored_cache.npz   seq, row, y · 3.528.273 filas
    │   ├── feature_select/features.py        load_rankings, select_temporal_features
    │   ├── a_boundary_flags/                 exploración de las columnas a*
    │   └── model_target/                     exploración de los targets
    ├── notebooks_kaggle/
    │   ├── make_sprint{17,18,19,20,21,22,23,24}.py    generadores
    │   └── connectome_sprint*.ipynb                   ~40 notebooks
    ├── models_first_try/                     primeros modelos
    ├── explore/                              exploración de datos
    └── submissions/                          10 entregas armadas (fuera de git)
```

---

## 14. Para arrancar

1. **Clonar el repo** en `challenge/`: `git clone https://github.com/ElNelsonn/challenge`.
2. **Conseguir los datos**: `datasets/*.parquet` no están en git. Bajarlos del starter pack de la competencia.
3. **Regenerar el cache de evaluación** `challenge/model/submission/results/valid_scored_cache.npz` (claves
   `seq`, `row`, `y` de las filas con `is_scored`) para poder correr cualquier script de `analisis/`.
4. **Bajar las predicciones de los modelos** desde Kaggle con `kaggle kernels output`, o reentrenar.
5. **Verificar la métrica** reproduciendo el baseline: debe dar **0,589595** sobre valid completo y **0,6498**
   en los 10 grupos fijos (`--valid-groups 10 --seed 2026`).
6. **Antes de gastar GPU**, correr la prueba de humo del notebook contra archivos locales. Los dos bugs
   históricos fueron claves mal escritas en el estado del pipeline, y los dos se detectaban ejecutando el
   código de verificación en la laptop.

**Estado operativo al 2026-10-26:** nada corriendo en Kaggle. Cuota de GPU: 10,3 h de 30, se reponen el
2026-10-31. Última entrega: 0,65400 en el leaderboard, puesto ~410, con el ensamble de 4 descrito en §9.
