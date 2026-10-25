# HANDOFF — para retomar el trabajo en una sesión nueva de Claude

Archivo pensado para que una sesión nueva arranque con todo el contexto. Joaquín lo abre y escribe algo como:
`Leé challenge/docs/HANDOFF.md, challenge/docs/DIARIO.md (última entrada) y challenge/EXPERIMENTS.md (tabla de submissions) y seguimos desde ahí.`

Última actualización: **2026-10-21.**

## 1. Qué es esto

Competencia **Alpha Connectome (WunderNN)**: predecir `t0`, `t1` fila por fila a partir de 112 features de un libro de órdenes. Métrica = correlación ponderada global (WP). Deadline **2026-11-15**. Objetivo de Joaquín: el mejor modelo posible, llegar al top 20 (≈ 0.657 en el leaderboard; top 1 ≈ 0.662), y **documentar todo** para un informe final.

Equipo: Joaquín (estudiante UM, principiante en ML, español rioplatense) y Santiago (compañero, hizo el primer modelo que dio 0.6426).

## 2. Reglas de trabajo con Joaquín (no negociables)

- **Git:** la regla original era no tocarlo. **El 2026-10-20 Joaquín pidió explícitamente que lo haga yo** ("me pusheas todo vos") para que un compañero pudiera clonar el estado. Desde entonces commiteo y pusheo yo, con `.gitignore` puesto (sin el .gitignore el repo pesa 6,7 GB y tiene 48 archivos sobre el límite de 100 MB de GitHub; con él, 50 MB).
- **Explicar en lenguaje simple**, sin jerga ni expresiones raras. Si hay que usar un término técnico, decir en una frase qué significa.
- **Documentar todo** a medida que pasa: `EXPERIMENTS.md` (tabla de experimentos y submissions) y `docs/DIARIO.md` (diario por día, incluye "ideas descartadas y por qué").
- **Ser honesto con las proyecciones**, no prometer de más. Cada +0.01 en valid ≈ +0.01 en el leaderboard, y el LB ≈ valid − 0.030.
- **Kaggle:** desde el 2026-10-16 **subo y corro yo los notebooks** con `kaggle kernels push -p <carpeta>` (carpeta con el `.ipynb` y un `kernel-metadata.json`). Él sigue subiendo los zips a wundernn.io. Los outputs se bajan con `kaggle kernels output <slug> -p <carpeta>`.
- Sin gastar plata: Kaggle gratis + laptop. **30 h de GPU por semana y 20 de TPU (sin usar), 2 sesiones de GPU a la vez, 12 h máximo por sesión.** `kaggle quota` dice cuánto queda y cuándo se repone (se repone los **sábados**, no los jueves).
- **Hay DOS cuentas de Kaggle** (desde el 2026-10-21): la principal `joaquinfiorina` (dueña de los datasets) y `joaqfio`, con los 4 datasets compartidos como colaborador. Son 60 h de GPU por semana entre las dos. Para cambiar de cuenta: la sesión de la principal está respaldada en `~/.kaggle/credentials_principal.json` y la segunda usa un token en `~/.kaggle/access_token` (el token no va al repo). `kaggle quota` confirma en cuál estás.
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

## 4. Estado al 2026-10-21

**Leaderboard público (leído el 21/10, completo):** 1.º 0.67380 · 50.º 0.66517 · 100.º 0.66256 · 200.º 0.65934 ·
300.º 0.65685 · 400.º 0.65491 · **nosotros 0.65400, puesto ~410** con 8 submissions.

**El campo es densísimo: de #400 a #100 hay 0.0077.** Con la brecha medida valid → leaderboard de **0.0323**:

| mejora en valid | puesto |
|---|---|
| +0.001 | ~396 |
| +0.002 | ~352 |
| +0.003 | ~294 |
| +0.005 | ~214 |
| +0.008 | ~118 |

**Corrección de criterio importante:** durante semanas descarté mejoras de +0.002 llamándolas "ruido". Son ruido
*para medirlas en valid*, pero valen **60 puestos**. Y los equipos del top 100 llevan entre 20 y 193
submissions; nosotros 8, teniendo 5 por día.

**El leaderboard dejó de moverse el 19/09.** Serie: S7 valid 0.6843 → LB 0.6542 · S8 0.6855 → 0.6541 ·
S9 0.6857 → 0.6533 · S10 0.6863 → 0.6540. Valid subió +0.0020 y el leaderboard −0.0002: todo lo ganado en valid
desde septiembre es sesgo de selección.

**Qué funciona (valid completo, 1.873 secuencias):** receta base LSTM 128x2, batch 32, trozos de 512, lr 5e-4
(decae 0.9), dropout 0.1, wd 1e-5, 10 épocas, SWA 7–10, destilada de 4 maestros. GRU normal ≈ 0.676 · LSTM
normal ≈ 0.680 · destilada ≈ 0.683. Mejor individual **0.6843** (LSTM 256x2), mejor ensamble **0.6864**.
Inferencia medida: 4×LSTM128x2 = 41 min de los 60 (estimado 41, real 41m52s: el modelo de costo acierta).

**LA REGLA QUE EXPLICA TODO EL ESTANCAMIENTO.** Un modelo nuevo aporta al ensamble si y solo si
`s > S·r`, que se puede escribir **`s/r > S`** (s = su puntaje, S = el del ensamble, r = su correlación con él).
Medido sobre las mismas 200 secuencias, con S = 0.6755:

| modelo | s | r | s/r | aporte honesto |
|---|---|---|---|---|
| otra LSTM | 0.6729 | 0.9951 | 0.6762 | +0.0000 |
| LightGBM | 0.6390 | 0.9271 | 0.6892 | −0.0014 |
| XGBoost | 0.6405 | 0.9279 | 0.6903 | −0.0015 |
| Echo State Network | 0.6099 | 0.8756 | 0.6966 | −0.0019 |

**`s/r` es casi constante (0.676–0.697) para cuatro familias radicalmente distintas, y el ensamble está en
0.6755, justo en el medio.** Cada modelo que podemos construir cae sobre la línea de empate: los más distintos
son peores en la misma proporción.

**Qué NO funciona (con evidencia, NO repetir):** arquitectura convolucional (4 corridas) · árboles LightGBM y
XGBoost (correlacionan 0.993 entre sí) · Echo State Network · corrector del error del ensamble (r = −0.14 y
+0.0003: el error es impredecible) · features a mano (lags/deltas, combinaciones dentro de la fila, memoria
larga ×2, banderas de centinela) · reparametrizar el target a suma/diferencia (−0.0031) · train+valid
(+17,7 % de datos = +0.0000) · horizonte de aprendizaje 1024 · modelos más grandes · schedule coseno · ajustar
pesos del ensamble · un quinto modelo en el ensamble (−0.0001) · elegir la mejor de muchas combinaciones por
su valid.

**EL DIAGNÓSTICO VIGENTE (2026-10-21).** Nuestro ensamble de 4 gana apenas **+0.0021** sobre el mejor modelo
solo; en una competencia normal promediar 4 modelos vale +0.005 a +0.015. Medido:

| familia | correlación entre ellos | gana promediar |
|---|---|---|
| los 4 destilados de los MISMOS maestros | 0.9854 | +0.0021 |
| 4 sin destilar (sprint 4) | 0.9635 | **+0.0051** |

**Nunca variamos nada entre los modelos del ensamble salvo la semilla:** mismos datos, mismos maestros, misma
receta. Eso es lo que atacó el sprint 24 — **y falló. El diagnóstico de arriba está refutado; ver abajo.**

## 5. El sprint 24 y por qué cierra la rama del ensamble (2026-10-22)

**Sprint 24 — ensamble diverso de verdad** (cuenta `joaqfio`, 4 corridas de ~1,7 h). Movió las tres palancas de
diversidad a la vez, las tres sin usar hasta ese día:

| | datos | maestros | receta |
|---|---|---|---|
| div_a | 70 % (semilla 2301) | E_gru + l1 | LSTM 128x2, lr 5e-4 |
| div_b | 70 % (semilla 2302) | LR + LA | LSTM 128x2, lr 4e-4, dropout 0.15 |
| div_c | 70 % (semilla 2303) | E_gru + LA | LSTM 128x2, lr 6e-4 |
| div_d | 70 % (semilla 2304) | l1 + LR | **GRU** 128x2, lr 6e-4 |

Cada uno ve 7.424 de las 10.607 secuencias; comparten el 70 % de a pares y entre los cuatro cubren el 99 %.
Generador: `notebooks_kaggle/make_sprint24.py`.

Corrieron **tres de los cuatro**; `div_b` quedó cancelado porque se agotó la cuota.

**RESULTADO (medido sobre las 3.528.273 filas puntuadas, `analisis/s24_final.py` y `analisis/s24_cuaterna.py`).**
Qué palanca bajó la correlación y qué palanca no:

| par | qué cambia entre ellos | correlación |
|---|---|---|
| `div_a` – `div_c` | datos, maestros, lr — **las dos LSTM** | **0.9849** |
| `div_a` – `div_d` | lo mismo **+ LSTM contra GRU** | **0.9769** |
| `div_c` – `div_d` | lo mismo **+ LSTM contra GRU** | **0.9768** |
| los 4 viejos entre sí | solo la semilla | 0.9860 |

**Cambiar datos y maestros no mueve la correlación. Cambiar la arquitectura sí.** Eso refuta la hipótesis con la
que se lanzó el sprint: los 4 maestros son ellos mismos casi idénticos (~0.98 entre sí), así que darle a cada
alumno un par distinto es darle **el mismo objetivo**.

**Y lo grave: los tres pasan la regla `s > S·r` con margen y la ganancia es cero.**

| modelo | `s` | `r` con el ensamble | `S·r` | margen | la regla dice |
|---|---|---|---|---|---|
| `div_a` | 0.6829 | 0.9902 | 0.6793 | +0.0035 | PASA |
| `div_c` | 0.6819 | 0.9900 | 0.6792 | +0.0027 | PASA |
| `div_d` | 0.6777 | 0.9835 | 0.6747 | +0.0030 | PASA |

Probadas **las 35 cuaternas** de los 7 modelos (cuatro es lo que entra en los 60 min de inferencia):

| ensamble | valid | leaderboard |
|---|---|---|
| mejor cuaterna: `div_a`+LR+long+b256 | 0.6861 | 0.6538 |
| **el nuestro: LR+LA+long+b256** | **0.6860** | **0.6537** |
| **ganancia real** | **+0.0000** | |

Con pesos no uniformes buscados a sorteo: **+0.0000**.

**⚠️ LA REGLA `s > S·r` NO SIRVE COMO CRITERIO DE DECISIÓN.** Es la segunda vez que falla en la misma dirección:
la Echo State Network (exp 041) pasaba por +0.0184 y su ganancia real fue −0.0019. Sale de suponer que los
errores de los modelos son intercambiables, y acá no lo son. **El único criterio válido es promediar y medir.**
Las decisiones tomadas con la regla en sprints anteriores hay que releerlas con eso en mente.

**Con esto se cierra la rama del ensamble**, la última abierta después de la de arquitecturas (la Echo State
Network, exp 041). El mapa completo de lo intentado: cambiar la **arquitectura** baja la correlación hasta 0.88
y da cero; cambiar **datos, maestros y receta** no la baja y da cero.

**Hallazgo lateral que vale por sí solo:** `div_a` sacó 0.6829 con el **70 % de los datos y 2 maestros** contra
0.6834 del equivalente con el 100 % y 4. **El volumen de datos y la cantidad de maestros casi no importan**, lo
mismo que ya había dicho la prueba de train+valid (+17,7 % de datos = 0.0000). Lo que falta no es más de lo
mismo: es **información que todavía no le damos a la red.**

**EL ÚLTIMO FRENTE, YA CERRADO (sprint 23, 2026-10-25).** La **normalización adaptativa por secuencia**
(`--adapt-span`) era lo único que quedaba y lo único que cambia *qué ve* la red en vez de agregar otro modelo
del mismo tipo: 97 de las 112 features cambian de escala más de 2x entre secuencias y las normalizábamos con un
valor global.

| | puntaje solo | r con el ensamble |
|---|---|---|
| `adapt8000` (SWA) | 0.6801 | **0.9832** |
| LR / LA / long / b256 | 0.6823 – 0.6843 | 0.9939 – 0.9955 |
| `div_a` / `div_c` / `div_d` | 0.6777 – 0.6829 | 0.9835 – 0.9902 |

**La hipótesis se confirmó: es la correlación más baja de cualquier red que entrenamos.** Y la ganancia fue
**+0.0001** — probadas las 70 cuaternas de los 8 modelos (cuatro es lo que entra en los 60 min), la mejor da
0.6861 contra nuestro 0.6860 y la nuestra queda 4ª de 70.

**El resultado más limpio del proyecto sale de juntar los dos últimos sprints:** bajamos la correlación de
0.9955 a 0.9832 usando la palanca más agresiva que existe —cambiar la representación de entrada— y ganamos
+0.0001. **La correlación de ~0.98 no es un defecto de nuestra receta: es cuánta información distinta hay para
sacar de estos datos con una red recurrente de este tamaño.**

La corrida hermana `adapt2000` murió: gastó **695 minutos solo precalculando los soft targets** y la mató el
límite de 12 h, cuando `adapt8000` con el mismo código tardó **37 minutos** en ese paso. Suerte de máquina otra
vez; la guarda de velocidad mide el arranque (0,05 s/secuencia, la dejó pasar) y el frenazo apareció durante la
corrida. Para cubrirlo haría falta un control de presupuesto dentro del bucle de soft targets.

**Trabajo técnico pendiente si se retoma:** la normalización adaptativa **no está implementada en el kernel de
inferencia** (`model/submission/solution.py` sólo tiene `MEAN`/`STD` fijos). Es viable —el kernel normaliza en
float, la cuantización int8 es de los pesos— y cuesta ~1 min de los 19 que sobran del presupuesto, pero habría
que escribirla y verificar paridad con PyTorch. El `adapt_span` queda guardado en `best_model.pt` dentro de
`pipeline`, **no** en su config reducida: ahí lo tiene que leer el exportador.

**Trampa de Kaggle que costó 24 h de cuota, y la corrección del diagnóstico:** dos corridas de 12 h murieron
sin entrenar porque la lectura del parquet iba a 4,5 s por secuencia en vez de 0,06 s. Primero lo atribuí a que
Kaggle monta el dataset en dos rutas y una es lenta, y los generadores pasaron a preferir la que contiene
`/datasets/`. **Ese diagnóstico era incorrecto:** `div_a` y `div_b` tomaron los dos la ruta "rápida" y uno igual
corrió a 6,6 s por secuencia. La causa real es **qué máquina te toca** (si el dataset quedó cacheado en su disco
local o no), y no se puede elegir. **El arreglo que sí sirve es la guarda de velocidad** que ya traen los
generadores desde el sprint 24: lee 15 secuencias al arrancar y aborta en 1 minuto si tarda más de 1 s cada una,
en vez de quemar 12 h de cuota.

      pf = pq.ParquetFile(TRAIN.split()[0])
      t0 = time.time()
      for g in range(15): pf.read_row_group(g, use_threads=False)
      seg = (time.time() - t0) / 15
      assert seg < 1.0, f'LECTURA LENTA: {seg:.1f} s por secuencia...'

**Cuota de GPU: 0.00 h en las dos cuentas, se repone el 2026-10-24.**
