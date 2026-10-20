# HANDOFF — para retomar el trabajo en una sesión nueva de Claude

Archivo pensado para que una sesión nueva arranque con todo el contexto. Joaquín lo abre y escribe algo como:
`Leé challenge/docs/HANDOFF.md, challenge/docs/DIARIO.md (última entrada) y challenge/EXPERIMENTS.md (tabla de submissions) y seguimos desde ahí.`

Última actualización: **2026-09-27 (domingo).**

## 1. Qué es esto

Competencia **Alpha Connectome (WunderNN)**: predecir `t0`, `t1` fila por fila a partir de 112 features de un libro de órdenes. Métrica = correlación ponderada global (WP). Deadline **2026-11-15**. Objetivo de Joaquín: el mejor modelo posible, llegar al top 20 (≈ 0.657 en el leaderboard; top 1 ≈ 0.662), y **documentar todo** para un informe final.

Equipo: Joaquín (estudiante UM, principiante en ML, español rioplatense) y Santiago (compañero, hizo el primer modelo que dio 0.6426).

## 2. Reglas de trabajo con Joaquín (no negociables)

- **Nunca ejecutar git** sobre `challenge/` (él hace commits/push). Tampoco tocar el repo remoto.
- **Explicar en lenguaje simple**, sin jerga ni expresiones raras. Si hay que usar un término técnico, decir en una frase qué significa.
- **Documentar todo** a medida que pasa: `EXPERIMENTS.md` (tabla de experimentos y submissions) y `docs/DIARIO.md` (diario por día, incluye "ideas descartadas y por qué").
- **Ser honesto con las proyecciones**, no prometer de más. Cada +0.01 en valid ≈ +0.01 en el leaderboard, y el LB ≈ valid − 0.030.
- Flujo: yo propongo y preparo (notebooks, scripts, zips); **él sube los notebooks a Kaggle** (Import → Save & Run All) y **sube los zips a wundernn.io**. Yo bajo los outputs con `kaggle kernels output <slug> -p <carpeta>` y subo modelos al dataset con `kaggle datasets version`.
- Sin gastar plata: solo Kaggle gratis (30 h GPU/semana, se renueva el **jueves a la noche**, 2 sesiones GPU a la vez, 12 h por sesión) + laptop.
- Toda pieza nueva de GPU se prueba primero con una **"prueba de humo"** (32 secuencias, 1 época, en la GPU real) al inicio del notebook, para no perder una noche por un error tonto (pasó dos veces).

## 3. Dónde está todo

Raíz del proyecto: `C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\`
- `wnn_connectome_starterpack\wnn_connectome_starterpack\` = starter pack (`utils.py`, `datasets/{train,valid}.parquet`).
- `...\challenge\` = **repo del equipo, donde se trabaja**:
  - `EXPERIMENTS.md` → tabla de todos los experimentos (001–025) y submissions (S1–S7), calibración valid→LB.
  - `docs/DIARIO.md` → diario día por día + ideas descartadas + pendientes.
  - `docs/HANDOFF.md` → este archivo.
  - `model/official_gpu/train_official_gpu_v1.py` → **el entrenador** (GRU/LSTM, destilación con soft targets en cache, SWA, resume). Outputs bajados de Kaggle en `model/official_gpu/results/<sprint>/`.
  - `model/submission/solution.py` → **kernel de inferencia v3** (numba, GRU+LSTM, int8). `build_submission.py` arma el zip. `eval_ensemble.py` evalúa modelos y todas sus combinaciones en valid completo (1.873 secuencias) y estima si entran en los 60 min. `kernel_bench/time_anchored.py` mide el tiempo anclado a un submit real.
  - `submissions/` → zips armados (`ens4_y2_Eswa_X_Zswa_int8.zip` = S7, el actual).
  - `notebooks_kaggle/make_sprint{3,4,5,6}.py` → generadores de los notebooks de Kaggle (`python notebooks_kaggle/make_sprint6.py` desde `challenge/`).
- `kaggle_upload\models\` → copia local del dataset de Kaggle `joaquinfiorina/connectome-models` (v6). Para subir una versión nueva: `cd` a esa carpeta y `kaggle datasets version -p . -m "v7: ..."`.
- Memoria propia de Claude (se carga sola si la sesión se abre en `C:\Users\Joaquin\Desktop`): `C:\Users\Joaquin\.claude\projects\C--Users-Joaquin-Desktop\memory\orderbook-wp-competition.md`.

## 4. Estado al 2026-09-29

**Leaderboard público (leído el 28/9):** 1.º 0.6677 · 20.º 0.6616 · 50.º 0.6589 · 100.º 0.6566 · **nosotros 0.6542,
puesto 168** (con 8 submissions; los de arriba llevan 20–70). No bajamos por empeorar: en una semana todo el campo subió
~0.005 y nosotros estuvimos quietos. Serie propia: 0.6426 → 0.6471 → 0.6495 → 0.6524 → **0.6542** → 0.6541.
Calibración estable: **leaderboard ≈ valid − 0.031**, y los deltas se trasladan 1:1 *salvo* que el modelo se haya
elegido mirando muchas opciones en valid (eso infla ~0.001: caso S8). **Regla: no se sube nada que no gane ≥ 0.004 en valid.**

**Qué funciona (valid completo, 1.873 secuencias):**
- Receta base: LSTM 128x2, batch 32, trozos de 512, lr 5e-4 (decae 0.9), dropout 0.1, wd 1e-5, 8 épocas, SWA 5–8,
  destilación del mejor ensamble (loss WMSE con peso |y| contra el target y contra el maestro).
- GRU normal ≈ 0.676 · LSTM normal ≈ 0.680 · destilada ≈ 0.683.
- **Mejor individual: alumna LSTM 256x2 = 0.6843.** Mejor ensamble: **0.6855** (Eswa + l1swa + LRswa + LA8) = la S8.
- Inferencia MEDIDA (anclada a S7 = 35 min reales): 1×LSTM192x2 = 20,5 min · 1×LSTM256x2 = 31,1 · 2×LSTM192x2 = 39 ·
  4×LSTM128 ≈ 40. El kernel v3 soporta esos tamaños.
- **Las tablas de ensamble se arman en la laptop sin GPU:** `python model/submission/combine_local.py --models
  NOMBRE=<valid_preds_*.npz> ... --max-combo 4 --hidden E=128gru,l3=160`.

**Qué NO funciona (con evidencia, no repetir):** features a mano — 4 intentos (lags/deltas ×3 y combinaciones no
lineales dentro de la fila, exp 027) · LightGBM (no entra en tiempo) · schedule coseno · modelos más grandes
(192x2 = 0.6834 y 256x2 = 0.6843 contra 0.6834 de la de 128: exp 026) · trozos de 2.000 pasos · calibrar la escala por
secuencia (el oráculo vale +0.047 pero es inestimable: R² 0.05–0.15) · ajustar pesos del ensamble (+0.0000 fuera de
muestra) · elegir la mejor de muchas combinaciones por su valid.

**El diagnóstico:** los errores de todos nuestros modelos correlacionan **0.96–0.98** — son el mismo modelo con distinta
cara, por eso el ensamble satura. El 68 % del peso de la métrica está en las filas con |y| grande.

## 5. Qué está corriendo

**Sprint 10 (2026-09-29) — la loss.** Hallazgo por investigación: el competidor **dozken** (puesto ~51, 0.6588) tiene
[repo público](https://github.com/dozken/wunder) con arquitectura *peor* que la nuestra (GRU 192x2 + proyección 128) pero
entrena con **"sequence-level weighted Pearson on soft-masked rows"**, advirtiendo que el Pearson por chunk no se parece
a la métrica. La métrica es una **correlación** (invariante a escala) y nosotros entrenábamos con error cuadrático:
gastábamos capacidad en la magnitud, que no se puntúa. Ya estaba probado y descartado (exp 018), pero justo con el error
que ese repo señala: peso 0.05 sobre trozos de 512 filas.

`--loss seqwp` (nuevo): acumula los estadísticos de la correlación **a lo largo de toda la secuencia** (olvido
exponencial `--seqwp-decay`), más `--wmse-anchor` para que la escala de la salida no quede libre.
Notebooks: `connectome_sprint10_seqwp.ipynb` (sin destilar) y `connectome_sprint10_seqwp_dist.ipynb`
(**comparación exacta contra la alumna LR = 0.6834**). Criterio: **≥ 0.687 = la loss era la diferencia.**

### Hoja de ruta (2026-09-29, ordenada por valor / costo)

1. **Memoria larga (EMA)** — el entrenamiento corta el gradiente cada 512 pasos, así que el modelo **no puede aprender**
   a usar información de hace miles de filas aunque la secuencia tenga 20.000. Promedios móviles de las features con
   horizontes largos (100 / 1.000 / 10.000) se la dan servida. Una cuenta por fila (`nuevo = 0.99·viejo + 0.01·actual`),
   o sea nada en el kernel. **Es distinto de los 4 intentos de features que fallaron**: esos eran de memoria corta, que
   la red ya construye sola. Costo: 2,5 h.
2. **Capa recurrente lenta (multi-escala)** — una segunda red chica que se actualiza cada 32 filas. Ataca el mismo
   agujero por arquitectura, cuesta 1/32 del tiempo de inferencia, y sus errores **no** estarían correlacionados con los
   de la familia actual (hoy 0.96–0.98) → el ensamble volvería a sumar. Costo: 1 día de trabajo + 2,5 h de GPU.
3. **Máscara de filas puntuadas** — pesar el entrenamiento por la probabilidad de que la fila cuente (AUC 0.763 ya
   medido; `train` no trae `is_scored`, hay que generar la máscara con un modelo entrenado en valid). Costo: 2,5 h.
4. **Paquete de mejoras chicas en una corrida** — proyección lineal 112→128 antes de la red (el competidor la usa),
   tareas auxiliares `(t0+t1)/2` y `(t0−t1)/2`, batch más grande. Costo: 2,5 h.
5. **Entrenamientos largos** — todos nuestros modelos se plantan en la época 7 de 8; nunca probamos 20 épocas con paso
   chico y más regularización. Costo: 6 h.
6. **Otra familia de modelo** (convolucional dilatado / atención) — lo único con potencial de +0.01 de una, porque hoy
   todos nuestros modelos son el mismo con distinta cara. Hay que escribirlo en el kernel numba. Costo: varios días.

**Estrategia (no son modelos):**
- **Usar las submissions:** 5 por día disponibles, llevamos 8 en total; los de arriba llevan 20–70. El leaderboard es la
  única medición no contaminada que tenemos.
- **Regla de selección:** nada se elige sin confirmarlo en la **mitad de valid que no se usó para elegir**. Gratis, y
  evita exactamente el error que nos costó la S8.
- **Noviembre (idea de Joaquín):** 10 folds sobre train+valid para el modelo final — ahí sí conviene, porque permite
  usar todos los datos y aun así tener una medición honesta. Antes de eso, entrenar con valid **no** es estrategia.
