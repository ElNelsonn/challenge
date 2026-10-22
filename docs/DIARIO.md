# Diario del proyecto — Alpha Connectome (WunderNN)

Registro cronológico de qué se hizo, qué se decidió y por qué. Los números exactos de cada experimento
están en `EXPERIMENTS.md`; los artefactos (configs, curvas, pesos, logs) en `model/*/results/`.

---

## 2026-09-11 (jue) — Arranque

- Lectura completa de la documentación del challenge (data overview, submission guide, reglas, FAQ, premios) y del starter pack.
- **Problema**: predecir `t0`, `t1` (movimiento futuro del instrumento `i0`) fila a fila, a partir de 112 features de orderbook L11 de dos instrumentos + 8 extras. 10.607 secuencias de train (212M filas), 1.873 de valid.
- **Restricción clave detectada**: inferencia en 1 vCPU, 60 min para todo el test → ~96 µs por fila si el test es del tamaño de valid.
- Decisiones de infraestructura: GPU en **Kaggle** (gratis, 30 h/semana); PCs Windows del usuario como CPU; AWS descartado; sin gastar dinero.
- Hallazgos iniciales en 6 secuencias: features ya normalizadas (~rank-gaussian, rango ±2.4), sentinel `-5.1993`, targets cuantizados en múltiplos de ~0.125, `corr(t0,t1)` fuertemente negativa.

## 2026-09-12 (vie) — Baseline, numba, primeros entrenamientos

- Baseline oficial (GRU 128x2 en ONNX) medido: **82.7 µs/fila → 51.6 min proyectados**. Al borde del límite.
- Reimplementación de la misma GRU en **numba**: 27.7 µs/fila, predicciones idénticas (diff 2e-5). Abre 3× de margen de inferencia. (`connectome/src/gru_numba.py`)
- Datos subidos a Kaggle como 3 datasets (train partido en 2 por el límite de 20 GB).
- Repo del equipo `challenge/` (compañero Santiago): framework `models_first_try/` (Ridge, RF, MLP, LGBM/XGB, GRU/LSTM/TCN) y exploración de targets/variables.
- Primeros experimentos con LSTM 64×1 en CPU: MSE cruda 0.344 → **loss alineada a la métrica** (clip ±2, peso |y|) 0.404 → 500 seqs 0.458 → 3 épocas 0.500 (métrica vieja, 10 grupos). Baseline en esos grupos: 0.568.
- Notebook de EDA de targets: `t0` y `t1` nunca son ambos positivos; `t0+t1` no es −spread simple. Hipótesis: "uno compra, otro vende" a horizontes distintos.

## 2026-09-13 (sáb) — Trabajo del equipo (sin mí)

- Santiago corrió 3 experimentos con GRU 32×1, 100 seqs, 2 épocas, MSE cruda, 5 rondas pareadas: lags/deltas seleccionados por correlación (−0.008), flags de centinela (−0.011), feedback de predicciones propias (+0.001). **Ninguna feature ayudó** a esa escala.

## 2026-09-15 (lun) — Script GPU y primer modelo que supera al baseline

- Calibración: baseline evaluado en los mismos grupos de valid de cada experimento del equipo (0.560 / 0.637 / 0.552): la GRU-32 del equipo estaba ~0.10 abajo.
- **`model/official_gpu/train_official_gpu_v1.py`**: entrenamiento por batches (GPU/CPU), loss alineada, clipping, checkpoints/resume, GRU 128x2 por defecto. Validado en CPU (100 seqs, 2 ép. → 0.490).
- Primer notebook en Kaggle (armado a mano por el usuario, celdas `%%writefile`). Primera corrida completa: **GRU 128x2, 10.607 seqs, 10 épocas, 64 min → 0.5838 (ép. 4) vs baseline 0.568** en los 10 grupos (métrica vieja). Primer modelo que supera al baseline.

## 2026-09-16 (mar) — Cambio de métrica, LightGBM, submission pipeline

- **Los organizadores cambiaron la métrica** a Weighted Pearson **global** (todas las filas juntas, no promedio por secuencia). Implementada como `GlobalWP`; el scorer del starter pack quedó obsoleto. Baseline con la nueva en los 10 grupos: 0.6498; nuestra GRU 0.6474 (la ventaja de la vieja desaparece en 10 grupos).
- LightGBM sobre 112 crudas + deltas (lags 1/5/20/100), 3.2M filas, en Kaggle CPU: 0.6253 solo; mezcla 0.3·LGBM + 0.7·GRU = 0.6528. Descartado después: los árboles cuestan ~100 µs/fila en numba; con 200 árboles la ganancia desaparece.
- **Pipeline de submission**: `solution.py` con kernel numba unificado (normalización + GRU + opcional LightGBM), `build_submission.py` (exporta pesos, verifica paridad con PyTorch, métrica y µs/fila), `eval_full_valid.py`. Bug encontrado y corregido: `fastmath=True` completo rompía el manejo de NaN en los árboles (usar flags sin `nnan`/`ninf`).
- **Evaluación en TODO valid (1.873)**: baseline 0.6171; GRU exp009 0.6748 en 700 seqs → +0.053. Los 10 grupos engañaban.
- Notebook "Challenge" de Santiago (4 packs de features: base/temporal/events/complete): en el piloto ninguna supera a base; full/base 0.6722. Los outputs se perdieron por correr en sesión interactiva (lección: siempre Save & Run All).
- Leaderboard público: baseline 0.5935; top 1 0.6622; puesto 100 ≈ 0.642.

## 2026-09-17 (mié) — Primer LB real, ensambles, sprint

- Santiago subió su `full/base` → **LB 0.6426, puesto 123**. Calibración: delta vs baseline en valid (+0.056) se traslada ~0.9× al LB (+0.049). **Valid es confiable.**
- Análisis de su corrida completa (`trainposta`): `events` (2.880 entradas) 0.6716 vs `base` 0.6731 en 10 épocas; loss de train de `events` estancada; no entra en tiempo (170 µs/fila). Decisión: no reentrenarlo con más épocas.
- Informe externo (revisión de todo el trabajo) aportó: (1) el modelo de Santiago se entrenó con entradas recortadas a ±20 y nuestro exportador no lo aplicaba → corregido (`input_clip` por modelo); (2) `loss_wp` era por secuencia, no global; (3) **±5.199337 = Φ⁻¹(1e-7) = clip de `QuantileTransformer(normal)`** → el "centinela" es saturación del cuantil mínimo, y las medias por columna revelan el nivel de profundidad; (4) filas scoreadas tienen |target| mayor que el promedio; (5) fine-tuning corto como control barato.
- `solution.py` generalizado a **ensamble de N GRUs** (arrays apilados, hidden mixto con relleno). `eval_ensemble.py` para evaluar combinaciones en GPU.
- **S3 `ens2_exp009_santiago.zip`** (promedio de 2 GRUs) → **LB 0.6471, puesto 117** (+0.0045; valid predijo +0.004). Tiempo reportado por wundernn: 32 min → servidor ≈ 1.15× más lento que la laptop.
- **Sprint 1 en Kaggle** (~4 h): GRU 192x2 (0.6735, SWA 0.6765) y GRU 128x2 con dropout 0.1 + wd 1e-5 (0.6743, SWA 0.6762). Dropout y SWA suman ~+0.003 cada uno. 192 no rinde más que 128 solo.
- Tabla de ensambles en las 1.873: 3 modelos 0.6817, 2 modelos 0.6796–0.6805. **Medido: solo ensambles de 2 entran en 60 min** (3×128 = 56 min en laptop → 64 en servidor).
- **S4 `ens2_B_Cswa.zip`** (Santiago + dropout-SWA, 0.6796) → **LB 0.6495** (+0.0024 vs S3; estimado 0.6486). 33 min en servidor. Valid sigue prediciendo bien la dirección; la magnitud varía ±0.001.
- Conclusión del día: el tiempo de inferencia pasó a ser la restricción dominante. El "promediar más modelos" tocó techo.
- **Sprint 2** preparado (script v3): destilación del ensamble de 3 en una GRU 128, schedule coseno, loss global WP + pesos |y|^1.5. Lanzado esta noche (~4 h).

## 2026-09-18 (jue) — Kernel v2: entran 3 modelos

- S4 confirmó LB 0.6495. Sprint 2 (destilación / coseno / gwp) corriendo toda la noche.
- Mientras entrena Kaggle: **optimización del kernel de inferencia en la laptop** (sin GPU). Benchmarks round-robin fijando un núcleo,
  porque 4 procesos huérfanos de la herramienta estaban girando al 100 % y las mediciones saltaban ±40 %.
- Diagnóstico: v1 ya usaba AVX2; lo que costaba era (a) reducciones horizontales cortas, (b) 760 KB de float32 por modelo (L2 = 512 KB), (c) exp/tanh escalares de libm.
- Solución (`solution.py` v2): matrices traspuestas en bloques de 4 k, **pesos int8 con escala por fila**, tanh racional de Eigen. 3×128: **85 → 54 µs/fila** (1,58×).
  int8 no cambia la métrica (0.6831 en 120 seqs para float32/int16/int8). `build_submission.py --bits 8|16|32`.
- Consecuencia: **el ensamble de 3 (0.6817 en valid) entra en ~39 min de servidor**, y uno de 4×128 en ~51. La palanca "más modelos" vuelve a estar abierta.
- Armado `ens3_A_B_Cswa_int8.zip` como prueba; no se sube hasta ver la tabla del sprint 2 y elegir la mejor terna/cuaterna.
- Medición limpia (01:00, núcleo fijo, callback oficial, anclada a S4 = 33 min reales): **3×128 v2 int8 ≈ 30 min de servidor; B+Cswa+Dswa (con la 192) ≈ 39 min**; 4×128 ≈ 40; 5×128 ≈ 50.
  Lección de medición: la laptop varía ±60 % según carga/temperatura → medir siempre round-robin con núcleo fijo y **anclar a un tiempo real del servidor** (S4), no a µs absolutos.

- **Resultados del sprint 2** (bajados con `kaggle kernels output` a `model/official_gpu/results/sprint2/`): la **destilación funciona** (E: 0.6790, SWA **0.6805**, mejor modelo individual, +0.004 sobre Cswa; curva por encima de C en todas las épocas). Coseno (0.6747, SWA 0.6717: empeora desde ép. 7) y loss global WP + |y|^1.5 (0.6761 SWA) **no aportan**. Pares: Dswa+Eswa 0.6818.
- Decisión con las 12 h de GPU que quedan antes del reinicio: **fabricar alumnos**. Trainer **v4**: `--distill-cache` precalcula los soft targets de los maestros una sola vez (fp16, ~850 MB, en `/kaggle/tmp`) y los reutiliza en todas las épocas y corridas → cada alumno cuesta ~1,8 h en vez de 2,8 (probado en CPU: loss y métrica idénticas al modo online). Dataset `connectome-models` v3 con E/F/G. Notebooks generados por `notebooks_kaggle/make_sprint3.py`: `connectome_eval_combos.ipynb` (10 min: combinaciones de 1–4 de las 8 GRUs → submission de hoy) y `connectome_sprint3_alumnos.ipynb` (~9 h: elige maestros = mejor combinación de ≤4, entrena 5 alumnos semillas 2032–2036, tabla final).
- S5 `ens3_B_Cswa_Dswa_int8.zip` subido (primera con kernel v2).
- Tabla de combinaciones (`connectome_eval_combos`, 162 combos, 1.873 seqs; `model/submission/results/eval_combos/`): mejor de todas Dswa+Eswa+F+Gswa **0.6829**; mejor que entra en tiempo Dswa+Eswa+Gswa **0.6825** (zip armado: `ens3_Dswa_Eswa_Gswa_int8.zip`, ~40 min). Un 4.º modelo suma +0.0004: **los modelos actuales tocaron techo en ~0.683** (todos parientes: mismos datos, misma receta).

## 2026-09-19 (vie) — El sprint 3 falló; fix y relanzamiento

- **Los 5 alumnos del sprint 3 fallaron al minuto** con `CUDA out of memory` (pedía 7,6 GB): el precálculo de soft targets del trainer v4 pasaba las 20.000 filas de cada secuencia de una sola vez por la GRU de cada maestro (batch 32). En la laptop la prueba fue en CPU con 2 secuencias y no lo detectó. Costó ~1 h de GPU (no 9: el notebook seguía con el siguiente alumno).
- **Fix**: el precálculo recorre la secuencia por trozos de 2.000 pasos arrastrando el estado (como el entrenamiento). Re-probado en CPU con 3 maestros incluida la 192: loss y métrica idénticas al modo online.
- **Lección**: toda pieza nueva que corre en GPU necesita una prueba de humo en GPU con el batch real antes de dejar horas. El notebook ahora arranca con una celda de humo (precálculo + 1 época sobre 32 secuencias) que corta la ejecución si falla.
- **LB: S5 (B+Cswa+Dswa) → top 70; S6 (Dswa+Eswa+Gswa, 0.6825 valid) → 0.6524, puesto 65.** El kernel int8 corre bien en el servidor y la calibración valid→LB sigue 1:1 en los deltas (+0.0029 en valid → +0.0029 en LB).
- Santiago mandó un notebook (`proxima_tanda.ipynb`, generado con Codex): fine-tuning de Eswa por 3 épocas (lr 1e-4) destilando de D+E+G, en tres variantes de loss (control / auxiliar asinh 0.1 / Pearson global 0.05 con peso |y|). ~2,5 h. Es una ablación de loss, cuidadosa y barata; no compite con el sprint 4 (diversidad), se corre además, cuando haya un hueco de GPU.
- **Sprint 4 lanzado** (~14:00): `base_x` (notebook `4491be9e20`) y `base_y` (notebook `54eec4046b`) corriendo **a la vez** en dos sesiones GPU T4 (Kaggle permite dos). Esperado: x listo ~18:30, y ~19:30. Después: subir los 12 .pt al dataset v4 → `distill` (~6 h) → tabla → submission.
- Hallazgo operativo: Kaggle deja dos sesiones GPU simultáneas, así que los sprints se pueden partir en dos notebooks y correr en paralelo (misma cuota, mitad de reloj).
- Pregunta de Joaquín: ¿entrenar con 1 − WP como loss? Registro: como término agregado (exp 018) no sumó; como loss única nunca se probó. Agregado `--loss gwp` (1 − WP global del batch, pura) al trainer y notebook `connectome_ft_wp.ipynb` (`make_ft_wp.py`): fine-tuning de Eswa, 2 épocas, lr 1e-4, tres losses (control WMSE / gwp / wp por secuencia), ~1,3 h. Se compara gwp contra el control, no contra Eswa. Probado en CPU.
- Cola de GPU (máximo 2 sesiones): base_x + base_y → al terminar x: `ft_wp` → al terminar y: subir dataset v4 y `sprint4_distill` → después `proxima_tanda` de Santiago.

## 2026-09-20 (sáb) — Resultados del sprint 4, la LSTM, kernel v3

- **Maestros base (exp 020):** la sorpresa es **y2, la LSTM 128x2: 0.6803 sola**, igual que la destilada Eswa y +0.004 sobre cualquier GRU normal. Las otras arquitecturas (200×1, 64×3, 160×2) rinden igual o peor que 128×2. Idea de Joaquín validada: la diversidad útil vino por el tipo de celda, no por el tamaño.
- **Alumnos X, Y, Z (exp 021):** 0.6802 / 0.6798 / 0.6802 con SWA. Ninguno rompe el techo individual (~0.680): destilar de maestros de ~0.677 no supera destilar de maestros de 0.68. Pero la **tabla de 1.940 combinaciones** subió: mejor cuaterna **0.6843** (era 0.6829), y **la LSTM está en las 40 mejores**. Sin ella, 0.6836.
- **Kernel v3:** LSTM en `solution.py` (celda por modelo, estados c, compuertas i,f,g,o) y en el exportador. Paridad float32 4e-5. `ens4_y2_Eswa_X_Zswa_int8.zip` (0.6843, ~38 min servidor) listo para subir como S7.
- Conclusión: la palanca que queda es **más LSTM** y **alumnos LSTM** (destilar el mejor ensamble en una LSTM). Proyección honesta: 0.6843 ≈ 0.654 en LB (puesto ~55–60); el top 20 (0.657) pide ~0.687 en valid.
- **S7 → LB 0.6542 (35 min, 690 MB).** Exactamente lo que valid predijo (+0.0018). Serie: 0.6471 → 0.6495 → 0.6524 → 0.6542 en tres días; cada paso vale +0.002/0.003 y el top 20 (0.657) está a +0.003.
- **Alumna LSTM LA: 0.6823 (mejor individual).** Idea de Joaquín ("si la LSTM es tan buena, destilarla también") validada: +0.002 sobre Eswa y sobre y2. Y sigue subiendo en la época 8 → se retoma desde el checkpoint hasta la 12 (agregado `--resume-epochs/--resume-swa-start/--resume-cache` al trainer, y corregido que el resume use los mismos datos del run original). La segunda alumna (LB) murió por un bug mío en la recarga del cache de soft targets (releía el archivo de 849 MB por cada secuencia → sin RAM); corregido y probado.
- Tabla con LA: y2+Eswa+X+LA **0.6848** (+0.0005 sobre S7): no justifica submission. El ensamble está saturado en ~0.685 con estos miembros; lo que mueve ahora es la calidad individual (LA con más épocas, mejores recetas) más que sumar miembros.
- El notebook `sprint4_lstm` nunca se había lanzado (confusión de anoche). Lanzados a las 16:00 en paralelo: **`connectome-sprint4-lstm`** (3 LSTM base: 128x2 seeds 2047/2048, 160x2 seed 2049; ~5,5 h) y **`connectome-sprint5-lstm-students`** (2 LSTM 128x2 destiladas de y2+Eswa+Dswa+X+Zswa, seeds 2061/2062, cache compartido; ~4 h; `make_sprint5.py`). Dataset `connectome-models` v5 con los alumnos X/Y/Z. Las pruebas de loss quedan para la semana próxima (cuota).
- **Cambio de plan (idea de Joaquín): sprint 4 = diversidad.** En vez de 5 alumnos de los mismos maestros, entrenar **6 maestros distintos** (semillas, tamaños y celdas diferentes) en dos grupos, y destilar un alumno por grupo: x = {GRU 128x2, GRU 200x1, GRU 64x3} → X; y = {GRU 160x2, **LSTM** 128x2, GRU 128x2 dropout 0.2} → Y; más Z = alumno de los 6 + Eswa + Dswa. Razón: los modelos "parientes" tocaron techo al promediar (0.683); la 192 fue el miembro más útil de los ensambles por ser distinta; y los alumnos destilados son mejores miembros que los crudos. Los alumnos son GRU 128x2 (lo único que corre el kernel); los maestros pueden ser cualquier cosa. Trainer y evaluador extendidos a LSTM / N capas (probado en CPU). Notebooks generados por `notebooks_kaggle/make_sprint4.py`: `sprint4_base_x` (~4,5 h), `sprint4_base_y` (~5,5 h), `sprint4_distill` (~6 h, requiere dataset `connectome-models` v4 con los 6 maestros). El sprint 3 (5 alumnos iguales) queda de lado.

## 2026-09-21 (lun) — Resultados del sprint 6 y de las LSTM base; plan para la cuota que queda

- **Sprint 6 (hiperparámetros nunca tocados, sobre la alumna LSTM):** `chunk 2000` (trozos más largos para el entrenamiento) dio 0.6824 = LA (0.6823) → no aporta, descartado. **`lr 5e-4` dio 0.6834 (ép. 7; SWA 0.6834): mejor modelo individual hasta ahora**, +0.0011 sobre LA. Mejora chica (el ruido entre semillas es ±0.001) pero coherente: sus últimas 4 épocas están todas por encima del máximo de LA. Receta nueva para alumnas LSTM: lr 5e-4, 8 épocas, SWA 5–8 (ya que el techo llega en la 7).
- **LSTM base l1/l2/l3 (sin maestros):** 0.6808 / 0.6794 / 0.6809 con SWA. Tercera, cuarta y quinta confirmación de que una LSTM cruda vale ~0.680 (lo que valía una GRU destilada). 160x2 (l3) no mejora a 128x2, igual que pasó con las GRU.
- **Resumen de dónde estamos:** mejor individual 0.6834 (alumna LSTM lr 5e-4), mejor ensamble medido 0.6848 (y2+Eswa+X+LA), LB 0.6542 (S7, 0.6843). Faltan ~+0.003 en LB para el top 20.
- **Recalibré el estimador de tiempo de `eval_ensemble.py`** con el único tiempo real que tenemos del kernel v3 (S7: 35 min). La fórmula anterior descartaba combinaciones que sí entran: ahora 4 LSTM128 (~42 min) o 3 LSTM128 + GRU192 (~47 min) cuentan como válidas. Esto abre ensambles de puras LSTM.
- **Dataset `connectome-models` v6** subido: LA (best/swa/checkpoint ép. 8 + metrics/run_config, para retomarla), l1/l2/l3, LR (lr 5e-4) y C2k (chunk 2000).
- **Cuota:** quedan ~4 h de GPU hasta el reinicio del jueves. Por eso el sprint 5b sale en versión **lite** (`connectome_sprint5b_lite.ipynb`, ~2 h): retomar LA de la época 8 a la 12 (seguía subiendo) + **tabla de 1.092 combinaciones con las 13 mejores piezas** (y2, Eswa, Dswa, X, Zswa, l1/l2/l3 SWA, LR SWA, C2k SWA, LA8, LA12, LA12 SWA). Con la tabla se arma la S8. La alumna LB (seed 2062, 12 ép.) y una alumna con la receta nueva (lr 5e-4) quedan para después del jueves.
- Nota de método: cada sprint ahora deja **modelos reutilizables** en el dataset de Kaggle + una tabla de combinaciones; el submit se elige de la tabla, nunca "a ojo".
- **Noche — resultado del sprint 5b lite (`connectome-sprint5b-lite`, 45 min):** la tabla de 1.092 combinaciones dio **0.6855 con Eswa + l1 + LR + LA8** (1 GRU + 3 LSTM), +0.0012 sobre S7. Las tres LSTM destiladas están en todas las combinaciones de arriba; la 192 (Dswa) quedó afuera por tiempo. **S8 armada** (`ens4_Eswa_l1swa_LRswa_LA8_int8.zip`): paridad int8 vs float32 igual en métrica, ~40 min anclado a S7. LB esperado ~0.655.
- **El resume de LA falló** (`FileNotFoundError: /kaggle/working/datasets/train.parquet`): al retomar, el trainer usaba la ruta por defecto de mi laptop en vez de la del checkpoint. Mi prueba local no lo detectó porque en la laptop esa ruta existe. Corregido (ahora toma rutas, batch y chunk del checkpoint) y probado con una prueba que espía qué archivos abre. Lección: la prueba de humo tiene que ejercitar **el mismo camino** que va a correr (acá probé destilación pero no resume).
- **S8 → LB 0.6541 (39m46s).** Peor que S7 por 0.0001 cuando valid prometía +0.0012. Es la primera vez que la calibración falla, y la explicación es simple: ya exprimimos valid demasiado. Elegir la mejor de 1.092 combinaciones (con alumnos cuyas épocas y maestros también se eligieron mirando valid) hace que el número de valid quede ~0.001 inflado. Conclusión dura: **recombinar los modelos que ya tenemos no da más**; diferencias < 0.002 en valid son ruido. Para mover el LB hace falta un cambio real (más datos, información nueva), no otra tabla.
- **Cambio de rumbo para LA:** en vez de darle 4 épocas más (esperaba +0.001 individual y casi nada en el ensamble), la GPU que queda va a una **alumna de segunda generación (sprint 7, G2a)**: LSTM 128x2 con la receta lr 5e-4, destilada del ensamble nuevo de 0.6855. Razón: en todos los casos anteriores el alumno siguió a la calidad de los maestros (0.6817→0.6805, 0.6843→0.6834); con maestros de 0.6855 espero un individual de ~0.684 y un ensamble de ~0.686. Requiere ≥3 h de cuota; si no, jueves.

## 2026-09-26 (sáb) — Análisis a fondo: dos puertas cerradas, una abierta

Después de que la S8 diera 0.6541 (peor que la S7 con mejor valid), paré de recombinar y me puse a medir **dónde está
la pérdida**, todo en la laptop con las predicciones de valid que ya teníamos guardadas (3,5 millones de filas puntuadas).

- **Lo más grande que encontré, y que resultó inalcanzable:** si uno pudiera ajustar *la escala de cada secuencia* de las
  predicciones, el ensamble pasaría de 0.6853 a **0.7053** (solo escala) o **0.7319** (escala + corrimiento). Son +0.047,
  seis veces la distancia que nos separa del puesto 1. Pero eso usa los targets. Probé recuperarlo sin targets de tres
  maneras (normalizar por las propias predicciones con toda la secuencia; hacerlo de forma causal con promedios
  acumulados; predecir la escala desde las features): **todas empeoran o no explican nada** (R² 0.05–0.15). La escala de
  cada secuencia varía x4 y es ruido no observable: es una normalización que pusieron los organizadores y no se puede
  adivinar. **Puerta cerrada, con números.**
- **Ajustar los pesos del promedio** (en vez de 1/N para cada modelo): ajustados en la mitad de las secuencias y medidos
  en la otra mitad dan **+0.0000 y −0.0003**. El promedio simple ya era óptimo. **Puerta cerrada.**
- **Por qué el ensamble satura:** los errores de los 4 miembros correlacionan **0.96–0.98**. Son casi el mismo modelo.
- **Dónde se juega la métrica:** las filas con |y| grande son el 36% de las filas pero el **68% del peso**.
- **La palanca que queda:** el **tamaño de la alumna**. Todas las alumnas destiladas fueron 128x2 y ninguna llegó al nivel
  de su ensamble maestro; las pruebas de redes más grandes que hicimos fueron sin destilación y con la receta vieja. La
  hipótesis es que copiar un promedio de 4 redes en una red chica no entra por falta de capacidad.
- **Verificado antes de gastar GPU:** exporté redes LSTM 192x2 y 256x2 con pesos al azar y las medí en el kernel:
  **20,5 y 31,1 min** de los 60 (2×192 = 39 min). Entran. El kernel y el exportador las soportan sin tocar nada.
- **Herramienta nueva `model/submission/combine_local.py`:** arma las tablas de combinaciones **en la laptop** con las
  predicciones guardadas (antes eran 22 min de GPU por tabla). El entrenador ahora guarda también las predicciones del
  modelo SWA (`valid_preds_swa.npz`). De acá en adelante la GPU se usa solo para entrenar.
- **Sprint 8 preparado:** dos notebooks en paralelo, alumnas **LSTM 192x2** (9 ép., ~4 h) y **LSTM 256x2** (7 ép., ~5,5 h),
  lr 5e-4, destiladas del ensamble de la S8 (0.6855). **Regla nueva para submitear: solo si la tabla gana ≥ 0.004 en valid**
  (lo de abajo es ruido; la S8 lo demostró).

## 2026-09-27 (dom) — Las alumnas grandes no sirvieron; vamos a las features cruzadas

- **Sprint 8, resultado:** LSTM 192x2 = 0.6834 (idéntico a la alumna de 128), LSTM 256x2 = 0.6843 (+0.0009, dentro del
  ruido entre semillas). El mejor ensamble que entra en tiempo queda en **0.6854**, o sea igual que antes. **La hipótesis
  del tamaño queda refutada:** copiar el ensamble en una red más grande no da más. Tercer sprint seguido sin avance real.
- Dato útil para el informe: el 256x2 entrenó **más rápido por época** que el 192x2 (19 vs 32 min). Con hidden potencia de
  dos, cuDNN usa kernels más rápidos. Si volvemos a entrenar redes grandes, conviene 256 antes que 192.
- **Lo que sigue (sprint 9), y por qué:** quedaba un solo eje de entrada sin probar. Los tres intentos de features que
  fracasaron eran **lags y deltas en el tiempo**, cosa que una red recurrente ya hace sola. Al armar las nuevas casi
  cometo el mismo error al revés: mi primera versión eran spread, mid y huecos entre niveles, que son **sumas y restas**
  de las entradas — y la primera capa de la red ya calcula cualquier suma pesada de las 112 entradas, así que no habrían
  aportado nada y la corrida era tiempo tirado. Las rehice con las únicas que una red aprende mal: **cocientes
  (desequilibrios de volumen), desvíos y máximos entre los 11 niveles, concentración, y productos**. Son 20, se calculan
  dentro de cada fila (causales) y son baratas de meter en el kernel si funcionan.
- Dos notebooks para correr en paralelo: **`connectome_sprint9_cross.ipynb`** (el importante, ~2,5 h) y
  **`connectome_sprint9_r256.ipynb`** (retoma la 256x2 de la época 7 a la 12; su curva seguía subiendo; ~2,5 h).
  Dataset `connectome-models` **v7** subido con la 256 (best/swa/checkpoint) y la 192 (swa).
- Probé en la laptop el camino "retomar + destilación", que nunca había corrido entero (el sprint 5b murió antes de llegar).
- **Criterio para la próxima submission:** la de features tiene que dar **≥ 0.687 sola** para que valga la pena tocar el
  kernel; el ensamble tiene que superar 0.6855 por **≥ 0.004** para subir algo.

## 2026-09-29 (mar) — Features cruzadas: no. Y el leaderboard se nos fue

- **Resultado del sprint 9A (features cruzadas): 0.6831 contra 0.6834 sin ellas. No aportan nada**, ni solas ni en
  ensamble. Es el **cuarto** intento de features que falla (lags/deltas en el tiempo ×3, y ahora combinaciones no
  lineales dentro de la fila). Conclusión firme: con estas 112 entradas ya está todo lo que el modelo puede aprovechar.
- El sprint 9B (retomar la 256) **nunca llegó a correr**: falló en la primera celda porque la versión 7 del dataset no
  se había terminado de subir (se me cortó el comando a mitad). Ya está arreglado, pero quedó sin ejecutar.
- **Estado del leaderboard (público, leído el 28/9):** 1.º 0.6677 · 20.º 0.6616 · 50.º 0.6589 · 100.º 0.6566 ·
  **nosotros 0.6542, puesto 168**. No bajamos por empeorar: en una semana todo el campo subió ~0.005 y nosotros
  estuvimos quietos. Está durísimo: del puesto 20 al 110 hay 0.005 de diferencia total.
- **Dato que importa:** hay competidores en 0.662–0.664 con **1, 2 o 5 submissions**. O sea que existe un enfoque que
  llega ahí directo. Nuestro estancamiento es **estructural**, no de ajuste fino: nuestra familia de modelos (RNN de 2
  capas sobre las 112 crudas + destilación + ensamble) satura en 0.6855 de valid = ~0.654 de leaderboard, y ya probamos
  y descartamos con números todo lo barato que quedaba.
- **Lo que queda sin probar, en orden de costo:** (1) entrenar el modelo final con **train + valid** (18 % más datos, se
  hace siempre al final; no se puede medir en local); (2) **normalización online de las entradas** (que el modelo vea
  cada secuencia en su propia escala; es la única idea barata que nunca corrimos); (3) **otra clase de modelo**
  (convolucional dilatado o atención) para romper la correlación de 0.96–0.98 entre los errores de todos nuestros
  modelos: es lo único con potencial de +0.01, y son varios días de trabajo en el kernel numba.

## 2026-09-29 (mar) — Dejamos de adivinar: fuimos a ver qué hacen los que están arriba

- Joaquín frenó la idea de entrenar con train+valid ("eso es para los últimos días, no es una estrategia; la
  gente subió por hacer cosas distintas"). Tiene razón, la sacamos del plan hasta noviembre.
- En vez de seguir probando variantes a ciegas, busqué información. **El competidor `dozken` (puesto ~51, 0.6588)
  tiene el repositorio público.** Su arquitectura es GRU 192x2 con proyección de 128 — *peor* que la nuestra — pero
  entrena con **"sequence-level weighted Pearson on soft-masked rows"**, y su archivo de notas advierte que calcular
  el Pearson por trozos chicos no se parece a la métrica real.
- **Ahí está la diferencia conceptual:** la métrica es una **correlación**, y una correlación **no mira la escala**.
  Nosotros entrenamos con error cuadrático, o sea que el modelo gasta capacidad en acertar la magnitud, algo que la
  competencia ignora. Lo habíamos probado una vez (exp 018) y cometimos exactamente el error que ellos marcan: peso
  0.05 y trozos de 512 filas.
- Implementé `--loss seqwp`: acumula los estadísticos de la correlación a lo largo de **toda la secuencia** (con
  olvido exponencial para que el trozo actual siga pesando), más un ancla chica de error cuadrático para que la escala
  de la salida no quede libre. Probado en CPU.
- De paso, por lo de "soft-masked rows", medí si se puede adivinar **qué filas puntúa la métrica**: **AUC 0.763** con
  una regresión logística sobre las 112 crudas (ningún `aN` solo pasa de 0.60). Son el 10,9 % de las filas, vienen en
  rachas de 2 a 4 y tienen |target| algo mayor. Pero **train no trae esa columna**, así que usarlo implica generar una
  máscara aproximada: queda como segunda idea, después de la loss.
- **Sprint 10 lanzado:** `connectome_sprint10_seqwp.ipynb` (sin destilar) y `connectome_sprint10_seqwp_dist.ipynb`
  (misma receta que la alumna LR, comparación exacta contra 0.6834). Si la loss da +0.004, cambia toda la estrategia.

### Ideas descartadas y por qué (para no repetirlas)
- TabPFN: no corre en 1 CPU a 96 µs/fila; sin memoria temporal.
- Reentrenar `events` con 20 épocas: loss de train estancada, 8 h GPU, no entra en tiempo.
- PCA sobre las features: la red ya aprende su proyección supervisada; pierde diferencias entre columnas correlacionadas.
- Transformar el target (asinh, potencia): la métrica ya dice qué target usar (clip ±2, peso |y|); +0.0008 en el mejor caso sobre predicciones.
- Retomar la 192x2 por más épocas: techo en ép. 7, lr ya en 0.00048.
- Schedule coseno a 10 épocas: sobreajusta desde la 7 (exp 017). Loss global WP / pesos |y|^1.5: no cambia nada (exp 018).
- Trozos (chunk) de 2.000 pasos en vez de 512: igual resultado, 30 % más lento (exp 024).
- **Alumnas más grandes (LSTM 192x2 y 256x2, destiladas, lr 5e-4):** 0.6834 y 0.6843 contra 0.6834 de la de 128x2. El tamaño no es el cuello (exp 026, 2026-09-27).
- **Features cruzadas no lineales dentro de la fila** (desequilibrios, desvíos, máximos, productos): 0.6831 vs 0.6834 sin ellas (exp 027). Cuarto intento de features que falla.
- **Features que son sumas o restas de las entradas** (spread, mid, huecos entre niveles): la primera capa de la red ya las calcula sola. Si se agregan features, tienen que ser no lineales.
- **Calibrar la escala/corrimiento de cada secuencia:** el oráculo vale +0.047 pero no se puede estimar ni de las predicciones ni de las features (2026-09-26). No volver a intentarlo salvo que aparezca información nueva.
- **Ajustar pesos del ensamble** (mínimos cuadrados por target): +0.0000 fuera de muestra. El promedio simple alcanza.
- **Elegir la mejor de muchas combinaciones por su valid:** con >1.000 combinaciones el valid queda ~0.001 inflado (caso S8: +0.0012 en valid → −0.0001 en el leaderboard).
- Modelos más anchos (160x2, 192x2) o más profundos (64x3) o más anchos y menos profundos (200x1): nunca superan a 128x2, ni en GRU ni en LSTM (exps 014, 020, 023).

## 2026-09-30 (mié) — La loss tampoco era

- **Sprint 10, resultado:** la loss de correlación a nivel de secuencia da **0.6837** contra **0.6834** de la misma
  receta con error cuadrático. Es la comparación más limpia que hicimos (dos corridas idénticas salvo la loss) y la
  diferencia es ruido. Sin destilación: 0.6804 contra 0.6808. Tampoco aporta al ensamble.
- O sea: **optimizar la correlación en vez del error cuadrático no cambia nada acá**, aunque la métrica sea una
  correlación. La ventaja de ese competidor viene de otro lado. Al menos ahora está medido y no hay que volver.
- **Patrón que ya no se puede ignorar:** cambiamos maestros, tamaño (128/192/256), features (4 variantes), receta de
  entrenamiento y ahora la función de costo — **todo aterriza en el mismo número**: ~0.680 sin destilar, ~0.684
  destilado, ~0.6855 el ensamble. Eso es la firma de un **techo de información** para esta familia de modelos con
  estas 112 entradas, no de un problema de optimización. Lo que queda tiene que cambiar *qué ve* el modelo o *cómo lo
  procesa*, no cómo se lo corrige.
- Siguiente por la hoja de ruta: **memoria larga (EMA)**, que es exactamente eso — darle información que hoy le es
  inaccesible porque el gradiente se corta cada 512 pasos. El notebook ya está listo (`connectome_sprint11_ema.ipynb`).



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

## 2026-10-01 (jue) — Ni memoria larga ni entrenar el doble

- **La grilla 2x2 del sprint 11 salió plana:** 8 ép. sin EMA 0.6834 · 8 ép. con EMA 0.6821 · 20 ép. sin EMA 0.6838 ·
  20 ép. con EMA 0.6838. Nada se mueve más de ±0.0004.
- **Dato útil igual:** las dos curvas hacen pico en la **época 9–10** y después bajan, y el SWA de las épocas 15–20
  (0.6808 y 0.6820) es **peor** que el de 5–8. O sea que 8–10 épocas es el óptimo y entrenar 20 es tirar 2 horas.
  Eso se aplica a todo lo que venga.
- **La medición que ordena todo:** calculé la correlación entre los **errores** de seis modelos. Están todos entre
  **0.975 y 0.987**, sin importar entradas, tamaño, épocas ni tipo de celda. El de memoria larga se equivoca donde se
  equivocan los demás. Por eso el ensamble está clavado: no estamos combinando modelos distintos, estamos promediando
  seis versiones del mismo.
- Esto **cambia el criterio de éxito**: veníamos midiendo cada idea por su score individual. Lo que hay que mirar es si
  **baja la correlación de errores**. Un modelo que empata en score pero se equivoca en otro lado vale más que uno que
  gana 0.001 y se equivoca igual.
- El mejor ensamble ahora da 0.6860 (ema+long+LR, 33 min), pero es el mejor de ~100 combinaciones y los modelos no son
  diversos: por la regla de ≥0.004 no se sube. **Seguimos en 0.6542 en el leaderboard.**
- **Lo único que queda en pie es el sprint 12 (capa lenta)**, y ahora con un criterio más honesto: si su correlación de
  errores con la familia baja de 0.97, hay diversidad real y el ensamble vuelve a moverse aunque el modelo solo empate.

## 2026-10-06/07 (mar/mié) — Dos resultados que cambian el diagnóstico

**1. La S9 dio 0.6533: peor que la S7 (0.6542), con mejor valid.** La serie completa es demoledora:

| | valid | leaderboard | brecha |
|---|---|---|---|
| S7 | 0.6843 | 0.6542 | 0.0301 |
| S8 | 0.6855 | 0.6541 | 0.0314 |
| S9 | 0.6857 | **0.6533** | **0.0324** |

La brecha **crece exactamente lo que ganamos en valid**. Eso es sesgo de selección acumulado: venimos eligiendo
la época por valid, los hiperparámetros por valid y la combinación por valid, treinta experimentos seguidos.
**Valid ya no sirve para rankear a resolución de 0.002.** Nuestro nivel real es el de la S7 y todo lo posterior
fue ruido.

**2. El sprint 13 contestó la pregunta de la diversidad, y la respuesta ordena todo.** Dos modelos entrenados
sobre mitades **disjuntas** (ni una secuencia en común) dan 0.6806 y 0.6815 — pierden apenas 0.002 con la mitad
de los datos — y sus **predicciones correlacionan 0.984**, igual que todo lo demás.

Verifiqué que no fuera un artefacto de medir sobre los errores (el error está dominado por el target, que es el
mismo para todos): la correlación entre las **predicciones** da lo mismo. Es real.

**Lo que eso significa:** no es que nuestros modelos se parezcan por usar la misma receta. Dos modelos que no
compartieron un solo dato de entrenamiento **aprenden la misma función**. Con estas 112 entradas y esta clase de
modelo, la función está esencialmente determinada: ya la encontramos, y lo que queda es ruido que ningún modelo
de esta familia puede predecir. Entrenar más modelos de este tipo, en cualquier variante, no va a mover nada.

**Consecuencias para el plan:**

- Queda **una sola palanca técnica**: cambiar de clase de modelo (convolucional dilatada o atención local). Si
  esa también aterriza en 0.684, el techo es del conjunto de features y no hay nada más que hacer.
- **Cambia el régimen de medición.** Si valid no puede rankear, retenerlo cuesta 1.873 secuencias de datos a
  cambio de un número que ya no decide nada. Propuesta: entrenar con train + 80 % de valid, dejar 375 secuencias
  como control de sanidad, y usar el leaderboard (5 submissions por día, usamos 8 en un mes) como juez real.

## 2026-10-07 (mié) — La convolucional no alcanza, pero es lo más distinto que logramos

- **Sprint 14:** la convolucional dilatada da **0.6739 y 0.6734** contra 0.6808 de la LSTM sin destilar. Peor sola.
- **Lo interesante:** su correlación con la familia recurrente es **0.962–0.972**, la más baja que medimos (entre
  LSTM es 0.976–0.988). Es la primera vez que un modelo nuestro mira algo distinto. Pero no alcanza: probando
  mezclarla con peso reducido (elegido en una mitad de valid, medido en la otra) da **−0.0001**. Está 0.007 abajo
  y esa desventaja se come la poca diversidad que aporta.
- **Dos salvedades que hacen injusta la comparación:** su curva **seguía subiendo** en la época 10 (0.6684 →
  0.6722, no había convergido) y corrió **sin destilación**, que en la LSTM vale +0.003. Por eso antes de cerrarla
  va una corrida con las dos cosas: 20 épocas y destilada (sprint 15B).
- **Y se implementó lo último barato que quedaba: el pesado por la máscara.** La métrica puntúa solo el 9,4 % de
  las filas, esa columna no viene en train, y una regresión logística sobre las 112 features la adivina con
  **AUC 0.794**: el decil más probable tiene 34,9 % de filas puntuadas contra 0,5 % del más bajo. Entrenar pesando
  hacia esas filas es el único cambio que alinea lo que optimizamos con lo que nos puntúan (sprint 15A).

## 2026-10-22 (jue) — El sprint 24 da cero y me tira abajo la regla con la que decidí todo el proyecto

- Corrieron tres de los cuatro (`div_b` quedó cancelado: se agotó la cuota). Primero medí solo `div_a` y `div_c`
  y vi que la correlación casi no bajaba. **Con `div_d` adentro el cuadro cambia, y vale separarlo bien:**

      div_a - div_c  (datos, maestros y lr distintos, las dos LSTM)   0.9849
      div_a - div_d  (lo mismo + LSTM contra GRU)                     0.9769
      div_c - div_d  (lo mismo + LSTM contra GRU)                     0.9768
      los 4 viejos entre si (solo cambia la semilla)                  0.9860

  **Cambiar datos y maestros no mueve nada. Cambiar la arquitectura sí.** Todo el 0.0065 de caída lo puso el GRU.
- Eso **refuta la hipótesis con la que lancé el sprint**: dije que la correlación de 0.986 venía de compartir los
  maestros. No. **Los 4 maestros correlacionan ~0.98 entre sí**, así que darle a cada alumno un par distinto es
  darle el mismo objetivo.
- **Y lo serio del día: los tres pasan la regla `s > S·r` con margen y la ganancia es cero.**

      div_a  s 0.6829  r 0.9902  S*r 0.6793  margen +0.0035  PASA
      div_c  s 0.6819  r 0.9900  S*r 0.6792  margen +0.0027  PASA
      div_d  s 0.6777  r 0.9835  S*r 0.6747  margen +0.0030  PASA

  Probé **las 35 cuaternas** de los 7 modelos (cuatro es lo que entra en los 60 min). La mejor da **0.6861**
  contra nuestro **0.6860**: **+0.0000**. Con pesos buscados a sorteo, +0.0000 también.
- **Es la segunda vez que la regla falla en la misma dirección**, así que no es casualidad: la Echo State Network
  pasaba por +0.0184 y la ganancia real fue −0.0019. La regla supone que los errores de los modelos son
  intercambiables y acá no lo son. **De ahora en más el único criterio que uso es promediar y medir.** Hay que
  releer con eso en mente las decisiones que tomé con la regla en sprints anteriores.
- **Se cierra la rama del ensamble**, la última abierta. El mapa: cambiar la arquitectura baja la correlación
  hasta 0.88 y da cero; cambiar datos, maestros y receta no la baja y da cero.
- **Lo que sí quedó, y vale por sí solo:** `div_a` sacó 0.6829 con el 70 % de los datos y 2 maestros, contra
  0.6834 del equivalente con el 100 % y 4. **El volumen de datos y la cantidad de maestros casi no importan.**
  Coincide con la prueba de train+valid del lunes (+17,7 % de datos = 0.0000). Lo que falta no es más de lo
  mismo: es **información que todavía no le damos a la red**.
- Queda **una sola cosa abierta**: la normalización adaptativa por secuencia (exp 040, sprint 23), que no agrega
  otro modelo sino que cambia **qué ve** la red. Verificada en la laptop desde el lunes (la variación de escala
  que ve la red baja de 4,07x a 1,09x) y nunca entrenada.
- **Cuota de GPU: 0.00 h en las dos cuentas, se repone el 2026-10-24.**

## 2026-10-21 (mié, noche) — Segunda cuenta de Kaggle y el sprint 24 corriendo

- **Nos quedamos sin GPU** (33,06 h usadas de 30; se repone el **sábado 24**, no el jueves como yo creía).
  `kaggle quota` lo dice exacto. De paso: hay **20 h de TPU sin tocar**, aunque adaptar el entrenamiento a TPU
  es trabajo y riesgo con LSTM y BPTT truncado.
- **Joaquín conectó una segunda cuenta de Kaggle (`joaqfio`)** y compartió los 4 datasets como colaborador.
  Verificado que los ve y que tiene las 30 h intactas. Entre las dos cuentas son 60 h semanales. La sesión de
  la cuenta principal quedó respaldada para poder volver.
- **Lancé el sprint 24: el ensamble diverso.** Cuatro alumnos donde por primera vez varío algo más que la
  semilla — datos distintos (70 % cada uno), maestros distintos (un par en vez de los cuatro) y recetas
  distintas (uno es GRU). Es lo único que ataca la causa que medimos hoy a la tarde.
- Actualicé `HANDOFF.md`, que estaba al 29 de septiembre y ya no servía para retomar nada.

## 2026-10-21 (mié, tarde) — Fuimos al leaderboard y cambió el criterio de todo el proyecto

- Joaquín insistió: "si estamos 400, algo no cierra". Tenía razón. Fui al leaderboard completo por primera vez
  en dos meses.
- **El campo es densísimo: de #400 a #100 hay 0.0077.** Con nuestra brecha de 0.0323, eso significa que
  **+0.002 en valid son 60 posiciones y +0.005 son 200**.
- **Esto corrige un error de criterio mío que venía de semanas:** descarté sistemáticamente mejoras de +0.002
  llamándolas ruido. Son ruido para *medirlas en valid*, no para la competencia.
- **Otro dato duro: los del top 100 llevan entre 20 y 193 submissions. Nosotros 8**, con 5 por día
  disponibles. Usan el leaderboard como instrumento; nosotros no.
- Verifiqué que no nos faltaran datos (la página dice "N≈200 features" y usamos 112): el parquet tiene 118
  columnas = 112 + 2 targets + 4 de control. No hay nada escondido.
- **Y medí lo que nunca había medido: cuánto correlacionan entre sí los modelos SIN destilar.**

      destilados    corr 0.9854   ensamble 0.6864   promediar vale +0.0021
      sin destilar  corr 0.9635   ensamble 0.6845   promediar vale +0.0051

  **Promediar vale 2,4 veces más sin destilar.** El mecanismo que sospechaba desde el sprint 15 queda
  confirmado: destilar a todos de los mismos maestros nos costó la mitad del ensamble.
- **Pero no alcanza como está:** los sin destilar que tenemos son del sprint 4 (8 épocas, receta vieja) y
  arrancan 0.005 más abajo, así que su ensamble queda debajo del nuestro. La comparación justa —sin destilar
  pero con la receta de ahora— nunca se hizo.
- **Y lo más importante que salió de esto:** nunca variamos NADA entre los modelos del ensamble salvo la
  semilla. Mismos datos, mismos maestros, misma receta. Las tres palancas estándar de diversidad (maestros
  distintos por alumno, datos distintos por modelo, recetas distintas) están sin usar.

## 2026-10-21 (mié) — La Echo State Network cierra la rama de arquitecturas

- Joaquín me mostró el cuadro de tipos de redes neuronales y preguntó si las había considerado todas. No: de
  ese cuadro habíamos probado cinco familias. La única que faltaba y tenía sentido era la **Echo State
  Network**: una red recurrente con los pesos **aleatorios y fijos**, donde solo se entrena una regresión
  lineal de salida. Interesaba justamente porque no se entrena, así que llega a otra solución por
  construcción. Corrió **en la laptop, en CPU, en 8 minutos** — útil porque no tenemos GPU hasta el jueves.
- **Dio 0.6099, con correlación 0.8756 contra el ensamble.** Esa correlación es **la más baja de cualquier
  competidor que construimos en dos meses** (árboles 0.927, convolucional 0.978, LSTM 0.985).
- **Pasa el umbral por el margen más grande de todos (+0.0184) y la ganancia honesta igual es −0.0019.**
- **Y acá apareció la regularidad que explica todo el proyecto.** El criterio `s > S·r` se puede escribir
  `s/r > S`. Midiendo las cuatro familias en las mismas 200 secuencias:

      otra LSTM            s/r = 0.6762
      LightGBM             s/r = 0.6892
      XGBoost              s/r = 0.6903
      Echo State Network   s/r = 0.6966
      el ensamble          S   = 0.6755

  **`s/r` es casi constante para cuatro familias radicalmente distintas, y el ensamble está justo en el medio
  de ese rango.** Cada modelo que logramos construir cae sobre la línea de empate: los más distintos son
  peores en la misma proporción, y las dos cosas se cancelan.
- Segundo efecto que completa el cuadro: aunque el margen sea nominalmente positivo, **elegir honestamente
  cuánto peso darle al modelo nuevo cuesta más ruido que lo que el margen vale.** Por eso los aportes medidos
  dan negativos en vez de +0.001.
- **La rama de arquitecturas queda cerrada con una medición.** No es que no hayamos encontrado el modelo
  distinto: lo encontramos (r = 0.876) y tampoco alcanza.

## 2026-10-20 (mar) — Un bug mío quemó 24 h de cuota de GPU

- **Las dos corridas murieron en el límite de 12 h de Kaggle sin entrenar una sola época.** Se pasaron 700
  minutos precalculando los targets del maestro, algo que normalmente tarda 30.
- **La causa: Kaggle monta el mismo dataset en dos rutas y una es ~70 veces más lenta.** 0,06 s por secuencia
  contra 4,5 s. Los notebooks viejos tenían la ruta rápida escrita a mano; cuando el sprint 19 falló porque el
  dataset no aparecía donde yo esperaba, lo "arreglé" con una función que busca el archivo por nombre — y
  devolvía el primer resultado, sin criterio. `chunk1024` agarró la buena y terminó en 128 min; las otras dos
  agarraron la mala.
- **Lección, y es mía:** cuando un arreglo reemplaza un valor fijo por una búsqueda, hay que decidir qué pasa
  si hay más de un resultado. La búsqueda parecía más robusta que la ruta a mano y era menos robusta.
- **Consecuencia concreta: no hay GPU hasta que Kaggle reponga la cuota semanal (jueves).** Dejé el
  relanzamiento automático preparado.
- **Lo que sí terminó: `chunk 1024` dio 0.6836 contra 0.6834.** Ruido. Duplicar hasta dónde puede aprender la
  red no cambia nada, igual que las features de memoria larga. La rama del horizonte queda cerrada.

## 2026-10-20 (lun, noche) — Encontré algo grande que estaba mal desde septiembre

- Joaquín me apretó porque bajamos al puesto 400 y le dije que iba a mirar una cosa más antes de contestar.
  **Apareció algo que estuvo mal desde el experimento 007.**
- **Cada secuencia tiene su propia escala y se la estábamos escondiendo a la red.** Medido sobre 150
  secuencias: **97 de las 112 features cambian de escala más de 2x entre secuencias**, 77 más de 3x, la
  mediana es **3,87x** y las peores llegan a **10x**. Y las normalizamos con una media y un desvío
  **globales**, calculados una sola vez. O sea que a la red le llegan números hasta 4 veces más grandes según
  en qué secuencia esté, y tiene que gastar capacidad en compensarlo sola.
- **Y conecta con el hallazgo más grande que teníamos sin explotar:** en septiembre medimos que corregir la
  escala de cada secuencia valdría **+0.047**, y nunca pudimos estimar esa corrección desde las features.
  Esto ataca lo mismo por el otro lado: en vez de corregir la salida, pone la entrada en la misma escala.
- **Implementado y verificado en la laptop:** con promedios móviles causales dentro de la secuencia, la
  variación de escala que ve la red baja de **4,07x a 1,09x**. Cuesta ~18 min por época.
- **Prioricé esto sobre todo lo demás:** frené el experimento del chunk 2048 y cancelé la 4.ª semilla de
  train+valid (que ya había dado cero con tres) para liberar las sesiones de GPU.
- Dicho con honestidad: no sé si va a mover el puntaje. Pero es un problema real, grande y medido, y es el
  primero en semanas que no es un ajuste fino sino algo que estaba mal.

## 2026-10-20 (lun, tarde) — Tercera semilla confirma el cero; queda un parámetro sin tocar

- **La tercera semilla de train+valid (final_d) dio 0.6724.** Las tres: 0.6718, 0.6746, 0.6724 → promedio
  **0.6729**. Los viejos, en esas mismas 200 secuencias, promedian **0.6730**. **+17,7 % de datos = 0.0000.**
- **Y la distancia entre semillas (0.0028) sigue siendo más grande que cualquier efecto que busquemos.** Esa
  es, en una línea, la razón por la que nada de lo que probamos en octubre se podía ver.
- **Revisé si sobraba algo en el presupuesto de inferencia.** Usamos 42 de 60 minutos, o sea que entra un
  quinto modelo. Medido: 4 modelos 0.6863, 5 modelos 0.6862. **−0.0001.** El quinto correlaciona 0.98 con los
  otros, así que promediar más no promedia nada nuevo. El presupuesto sobrante no es una oportunidad perdida.
- **Encontré un parámetro que nunca tocamos: el horizonte de aprendizaje (`--chunk`).** Las secuencias tienen
  20.000 filas y entrenamos cortando el gradiente cada **512**, lo que significa que **la red no puede
  aprender relaciones de más de 512 filas de distancia**. Está en 512 desde el 15 de septiembre y está fuera
  de la lista de "no volver a probar". Lancé 1024 y 2048.
- Lo honesto sobre la expectativa: las features de memoria larga dieron 0.0000 dos veces, lo que apunta a que
  la información vieja no sirve. Pero no es la misma prueba: esas features le dan masticado un promedio largo
  concreto, y el chunk le permite aprender sola qué estructura vieja importa.

## 2026-10-20 (lun) — Train+valid: +0.0002, y la prueba no alcanza para distinguirlo de cero

- **Los dos modelos con train+valid dieron 0.6718 y 0.6746** en las 200 secuencias que quedaron afuera. Los
  viejos, medidos en esas mismas 200, promedian **0.6730**; los nuevos, **0.6732**. **+0.0002.**
- **Lo honesto es decir que esta prueba no tiene potencia:** la distancia entre las dos semillas nuevas es
  **0.0028**, más grande que el efecto que buscábamos. Con dos modelos no se puede resolver un +0.002.
- **Y las nuevas correlacionan 0.9951 con el ensamble viejo**: +17,7 % de datos no las hizo distintas. Aporte
  honesto al ensamble: +0.0000, igual que todo lo demás.
- **Lancé dos semillas más (2213 y 2214)** para tener un ensamble de 4 modelos con train+valid, directamente
  comparable con los 4 viejos que produjeron el 0.6542 del leaderboard. **Esa comparación sí se puede hacer en
  el leaderboard**, que es el único juez que nos queda y el que venimos sin usar en serio: llevamos 8
  submissions en dos meses y tenemos 5 por día.
- Con esto el panorama es: todas las ramas cerradas con evidencia, y lo único que falta medir es si el +17,7 %
  de datos se traslada al test aunque no se vea en nuestras 200 secuencias.

## 2026-10-19 (dom) — XGBoost empata y correlaciona 0.993 con LightGBM; arranca el modelo final

- **XGBoost: 0.6405 contra 0.6390 de LightGBM.** Empate. Y lo que importaba más: su correlación con nuestro
  ensamble es **0.9279** contra 0.9271. Idéntica.
- **El número que cierra el tema: XGBoost y LightGBM correlacionan 0.9934 entre ellos.** Son el mismo modelo
  con otra implementación. Promediar los dos tampoco aporta (−0.0015). La rama de árboles queda cerrada con
  una medición, no con un argumento mío, que es lo que Joaquín pedía desde hace tres semanas.
- **Arrancó el modelo final con train + valid** (sprint 21), que es lo último que queda con mecanismo real:
  **+17,7 % de datos**. Se dejan afuera del entrenamiento las mismas 200 secuencias de valid que usamos para
  medir, así que queda un termómetro honesto. Verificado antes de correr que ninguna de esas 200 entra al
  entrenamiento.
- **El modelo final no elige nada por valid:** épocas fijas y SWA. Desde acá el único juez es el leaderboard,
  que es justo lo que necesitábamos: llevamos cuatro submissions donde valid subió +0.0020 y el leaderboard se
  movió −0.0002.
- Dos semillas en paralelo (2211 y 2212) para poder promediarlas sin elegir cuál es mejor.

## 2026-10-17 (vie) — La memoria larga da cero por segunda vez; XGBoost corriendo

- **Sprint 19: 0.6834 contra 0.6838. No aporta.** Las 11 columnas nuevas de memoria larga, esta vez elegidas
  **midiendo** su correlación con el error del ensamble en vez de a ojo, dieron lo mismo que las elegidas a ojo
  en septiembre: nada. Y la curva hace pico en la época 8 y después decae hasta 0.6768: con 33 entradas más la
  red sobreajusta antes.
- **Es la segunda vez que la memoria larga da exactamente cero.** La rama de features de entrada queda cerrada.
  El techo que yo había calculado (+0.001 a +0.002) resultó ser optimista: dio −0.0004.
- **XGBoost: corriendo.** Joaquín lo pidió tres veces y yo solo había argumentado por qué no esperaba mucho;
  esta vez se corre. Todo idéntico a LightGBM (10,6M filas, 190 columnas, 600 árboles, mismas 200 secuencias,
  misma semilla) y lo único distinto es la librería.
- **La primera corrida murió por memoria** y la causa es interesante: `xgb.DMatrix` guarda los datos como
  pares (índice, valor) de 8 bytes por número **aunque la matriz sea densa**, o sea 16 GB para nuestra matriz,
  más los 8 GB del array original, en una máquina de 33,7 GB. Arreglado con `xgb.QuantileDMatrix`, que guarda
  1 byte por número (~2 GB) y es exactamente el binneado que el método `hist` iba a usar igual. Es lo que
  LightGBM hace desde siempre con su `Dataset`, y por eso nunca tuvimos este problema con él.
- **Lo que se mide no es solo el puntaje.** Con la cuenta del proyecto, para que los árboles sumen +0.004
  necesitan subir el puntaje de 0.639 a 0.655 **o** bajar la correlación con el ensamble de 0.927 a 0.90.
  XGBoost arma los árboles por niveles y LightGBM por hojas, así que puede dar lo segundo aunque el puntaje
  quede igual. La correlación se mide acá cuando baje las predicciones.

## 2026-10-16 (jue, tarde) — Sprint 17: los dos dan lo esperado, o sea nada

- **Primera vez que subo y corro los notebooks yo**, con `kaggle kernels push`. Joaquín me lo pidió porque él
  estaba haciendo todo el trabajo mecánico sin ver resultados. Kaggle deja solo 2 sesiones de GPU a la vez.
- **17A, las columnas centinela: 0.6841 contra 0.6834.** +0.0007, ruido (el criterio era 0.6855). La
  distorsión era real y está medida — el desvío de a7 llegaba inflado 5 veces, y a7 limpia correlaciona 0.21
  con t1 contra 0.04 como se la pasábamos — pero **la red aprende sola a ignorar el centinela**. Las dos
  pruebas baratas que hice en la laptop ya lo habían dicho; esta vez acertaron.
- **17B, suma y diferencia: 0.6803 contra 0.6834. Peor, −0.0031.** Era el riesgo que estaba escrito en el
  propio notebook: la métrica puntúa t0 y t1 por separado, así que optimizar la parte común y la diferencia
  con sus propios pesos optimiza algo parecido pero no igual. Cerrado.
- **El sprint 19 falló por un problema de montaje de datos de Kaggle**, no por el código: no encontró
  `train_part1.parquet` con exactamente la misma configuración de inputs que los otros dos, que sí anduvieron.
  Relanzado.

## 2026-10-16 (jue) — Busqué de frente lo que queda, y queda poco

- **Medí la correlación entre el error del ensamble y los cambios de cada columna a retrasos de 1 a 200
  filas.** Hay señal real: lo más alto es **i1_v20 a 200 filas de retraso, −0.0343**, con ruido de ±0.003. Y es
  llamativo que la correlación más alta sea de **i1**, el instrumento correlacionado, al retraso más largo:
  es exactamente la relación "uno se mueve antes que el otro" que uno esperaría en un mercado.
- **Pero el techo es bajo y se calcula:** 0.034 contra un error que se lleva el 88 % de la varianza vale
  +0.0007 en la métrica. Todas las columnas juntas, descontando lo que se repite, +0.001 a +0.002. Consistente
  con el +0.0003 que había encontrado el corrector buscando a ciegas.
- **Descarté el calentamiento.** Cada secuencia reinicia el estado y tiene 99 filas de calentamiento; si eso
  costara, las primeras filas saldrían peor. Medido: las primeras están entre las **mejores** (0.69–0.70) y el
  peor tramo está en el medio. No hay nada que arreglar ahí.
- **Armé el sprint 19 (`--ema-plus`)** con las 11 columnas nuevas. Pero con dos avisos escritos en el propio
  notebook: la versión anterior de esta idea (las 40 features actuales) dio **exactamente 0.0000**, y el techo
  calculado es +0.001 a +0.002, que es justo el tamaño que viene sin trasladarse al leaderboard.

## 2026-10-15 (mié, madrugada) — Rompimos la pared de la correlación y del otro lado no había nada

- **Probé entrenar un corrector en vez de otro competidor:** un modelo que predice `y − predicción del
  ensamble`, o sea lo que las redes fallan. Se puede medir sin gastar GPU porque las redes nunca vieron valid,
  así que su error sobre valid es honesto.
- **Correlación del corrector con el ensamble: −0.14.** Todos los competidores que entrenamos en dos meses dan
  0.92 a 0.99. **Es lo único que construimos que no se parece a las redes**, y funciona exactamente como dice
  la cuenta `s > S·r`.
- **Pero la ganancia es +0.0003.** Y el dato que lo cierra: en la prueba chica (86 mil filas de ajuste) daba
  +0.0019, y al multiplicar por 6 los datos de ajuste **bajó** a +0.0003. O sea que el +0.0019 era ruido. Esta
  vez lo verifiqué antes de contarlo, que es lo que no hice con los árboles.
- **La conclusión es dura pero clara:** el error del ensamble se lleva el 88 % de la varianza del target y es
  impredecible desde estas 112 features. La pared de la correlación **no es un defecto de nuestra receta**: es
  que las redes ya extrajeron lo que hay. Agregar modelos al ensamble queda cerrado con evidencia.

## 2026-10-15 (mié, noche) — Los árboles mejoran solos pero no aportan; y un error mío cortó el otro notebook

- **Sprint 18: los árboles pasaron de 0.6253 a 0.6390** (+0.0137) con 3,3 veces más datos y las features de
  memoria larga. 71 min en CPU de Kaggle.
- **Pero la ganancia honesta en el ensamble es −0.0014.** El motivo está medido: **su correlación con el
  ensamble subió de 0.918 a 0.9271**. Las features de memoria larga, que son las que los hicieron mejores, son
  justamente las que los acercan a lo que hacen las redes recurrentes. El umbral subió de 0.620 a 0.6263 y se
  comió casi todo el margen.
- **Es la tercera vez que aparece el mismo patrón** (la convolucional destilada, los árboles con memoria, y
  antes cada red nueva): **mejorar un modelo lo vuelve más parecido al resto, y lo que gana en puntaje lo
  pierde en aporte.**
- **Me adelanté con esto y lo digo:** anuncié "+0.0019 si llegan a 0.640" usando una correlación de 0.918
  medida sobre 10 secuencias del modelo viejo. Medida bien, sobre 200 secuencias y el modelo nuevo, da 0.9271,
  y con ese número la ganancia prevista era +0.0009 desde el principio, no +0.0019.
- **Busqué si los árboles ganaban en alguna zona** (filas con predicción grande o chica, secuencias tranquilas
  o movidas): **no, es uniforme en todos lados**. Un peso por zona elegido honestamente da −0.0037.
- **El notebook de las columnas centinela falló por un error mío**, no por el experimento: la verificación de
  la prueba de humo leía la clave `columns` y el archivo la guarda como `names`. Se cortó en 13 s y nunca
  entrenó. Arreglado, y esta vez **ejecuté las verificaciones de verdad contra los archivos de la laptop** en
  vez de solo escribirlas. De paso apareció el mismo tipo de error en el otro notebook (buscaba `target_cd` en
  el config reducido del modelo en vez del completo del checkpoint): también arreglado y probado.

## 2026-10-15 (mié) — El leaderboard está clavado en 0.654

- **S10 (`ens4_mask_LR_LA_long_int8.zip`, valid 0.6863) dio 0.6540 en el leaderboard**, contra el 0.6542 de
  nuestra mejor submission. 41m52s reales, 690 MB.
- **Es la cuarta seguida sin moverse.** Desde el 19 de septiembre: valid subió +0.0020 y el leaderboard
  −0.0002. Las brechas: 0.0301 → 0.0314 → 0.0324 → 0.0323. La brecha crece justo lo que ganamos en valid.
- **O sea: todo lo que ganamos en valid en un mes es sesgo de selección**, no mejora real. Elegimos la época
  mirando valid, los hiperparámetros mirando valid y la combinación mirando valid; el leaderboard no se deja
  engañar por ninguna de las tres.
- **Regla nueva: no gastar submissions en variantes elegidas por valid.** Solo en modelos que sean distintos
  por una razón estructural. Eso deja exactamente dos cosas: los árboles (sprint 18, correlación 0.918) y
  entrenar con train+valid en noviembre, que ataca la brecha misma en vez del puntaje de valid.
- Detalle útil: el tiempo estimado para esta submission era 41 min y tardó 41m52s. **El modelo de costo de
  inferencia acierta**, así que sirve para planificar los 623 árboles con la máscara.

## 2026-10-13 (mar) — La cuenta que explica todo, y los árboles vuelven

Joaquín preguntó si habíamos probado XGBoost. Sí: LightGBM, que es la misma familia, en septiembre
(experimentos 010, 012, 013). Dio 0.6253 solo y lo descartamos **por tiempo de inferencia**. Pero nunca
medimos lo que después resultó ser el cuello de botella del proyecto entero, así que lo medí hoy con las
predicciones que habían quedado guardadas.

- **Los árboles correlacionan 0.918 con el promedio de nuestro ensamble.** Las redes entre ellas: 0.956–0.987.
  La convolucional del sprint 16, que era otra arquitectura: 0.978. **Los árboles son, por lejos, lo más
  distinto que construimos en todo el proyecto.**
- **Y salió la cuenta que explica los cinco sprints muertos.** Un modelo nuevo aporta al ensamble si y solo si
  su puntaje supera al del ensamble multiplicado por lo parecido que es: **s > S · r**. Con r = 0.985, una
  LSTM nueva tiene que igualar al ensamble completo para aportar algo — por eso nada se movía. La
  convolucional necesitaba 0.661 y dio 0.654: por eso dio exactamente +0.0000.
- **Los árboles de septiembre necesitaban 0.620 y dieron 0.625.** Justo en el límite, +0.0001. Pero como su
  correlación es baja, la pendiente ahí es empinada: si dieran 0.650 el ensamble gana **+0.0042**; con 0.660,
  **+0.0075**. No hace falta que se acerquen a las redes.
- **Medí el costo real de inferencia** porque la nota de septiembre y el docstring del kernel se contradecían
  por un factor 10. El kernel numba: **58 µs/fila por target**, 116 µs por los dos. El presupuesto total es 96
  y nuestras redes usan 66, así que quedan 30 µs = 156 árboles. No entran.
- **La salida es la máscara.** Corriendo los árboles solo en el 25 % de las filas con mayor probabilidad de ser
  puntuadas se atrapa el **68,7 % del peso de la métrica** y entran **623 árboles**. El resto de las filas se
  predice con las redes solas, como hoy.
- **Sprint 18 armado, y corre en CPU** (no gasta GPU). Tres cambios respecto de septiembre, ninguno es ajuste
  fino: **3,3x más datos** (10,6M filas contra 3,2M; septiembre usó el 1,5 % de train), **las 40 features de
  memoria larga** (un árbol no tiene estado, no puede calculárselas; es la ventaja principal de las redes), y
  **sacar 410 columnas muertas** (las 150 mejores tienen el 90,3 % de la ganancia, y sacarlas es lo que libera
  la memoria para los datos). En la prueba chica de la laptop, cuatro features de memoria larga aparecieron
  entre las más importantes de entrada.

## 2026-10-09 (vie) — Se cierra la rama de arquitecturas; aparece (y se desinfla) lo de las columnas centinela

- **Sprint 16: la convolucional convergida y sin destilar da 0.6738 y 0.6737.** Era el último cuadrante sin
  probar y salió como los anteriores. Las dos saturan: la de 128 canales hace pico en la época 21 y las últimas
  5 épocas dan −0.0007; la de 192 canales decae −0.0017, o sea que más tamaño tampoco ayuda acá.
- **Su correlación con la familia es 0.957–0.969, la más baja de todo el proyecto, y el aporte al ensamble es
  +0.0000.** La mejor combinación elegida honestamente (en la mitad A de valid, medida en la mitad B) es
  `LR+LA+long+b256` = 0.6862, idéntica con o sin las convolucionales.
- **Queda medido el techo de la arquitectura convolucional: ~0.674 sin destilar, 0.678 destilada, contra
  0.683–0.684 de la LSTM.** La rama se cierra con evidencia y no se vuelve. Son 4 corridas (sprints 14, 15B, 16)
  que dieron el mismo resultado desde ángulos distintos.
- **Después apareció algo que parecía grande y terminó chico.** Revisando las 112 columnas una por una, tres
  (a5, a6, a7) usan −5,1993 como marca de "dato faltante" y nosotros se lo pasábamos a la red como número.
  a7 tiene el centinela en el **44 %** de las filas: su desvío pasa de 0,583 a 2,934, así que al estandarizar
  la variación real le llega a la red comprimida a una quinta parte. Y a7 limpia correlaciona **0,21** con t1,
  la correlación más alta de una sola feature en todo el proyecto, contra **0,04** como se la pasamos hoy.
- **Lo probé dos veces sin GPU y las dos dieron cero** (lineal −0.0007, perceptrón por fila −0.0031 con ruido
  entre semillas de 0.0081). La red aprende sola la compuerta "si a7 vale −5,2, ignorála", y una vez aprendida
  la escala inflada no le molesta. Me entusiasmé con esto antes de medirlo; medido, la expectativa es baja.
  Se corre igual porque cuesta 2 h en una sesión que de otro modo queda ociosa.
- **Implementado `--target-cd`:** aprender la parte común (t0+t1)/2 y la diferencia (t0−t1)/2 en vez de t0 y t1
  por separado, que correlacionan −0,58 a −0,87. Verificado de punta a punta en la laptop: el modelo guardado
  devuelve t0 y t1 (diferencia máxima 4,9e−4 contra las predicciones del entrenador, que es la resolución del
  float16), así que el zip de submission y el kernel no cambian.
- **Sin subir todavía: `ens4_mask_LR_LA_long_int8.zip`** (0.6863 en valid, 41 min medidos, 1,02 MB).

## 2026-10-08 (jue) — Destilar y ser distinto están en conflicto

- **Pesar hacia las filas puntuadas (sprint 15A): 0.6840 contra 0.6834.** +0.0006, ruido. Que la métrica mire el
  9 % de las filas y nosotros entrenáramos parejo resultó no importar. Era lo último barato sin probar.
- **La convolucional bien entrenada (sprint 15B): 0.6782**, +0.004 respecto de la versión sin converger. Pero
  **su correlación con la familia recurrente subió de 0.962 a 0.975**. La destilación le dio score y le sacó lo
  único valioso que tenía: ser distinta. Aporte al ensamble: **+0.0000** medido honestamente (combinación elegida
  en una mitad de valid, reportada en la otra).
- **La lección, que no teníamos:** destilar mejora el score pero homogeneiza. Veníamos destilando todo desde el
  sprint 2, y eso explica en parte por qué toda la familia correlaciona 0.98. Nunca probamos el cuadrante
  **convergido y sin destilar**: la única corrida sin destilar (sprint 14) se cortó a las 10 épocas con la curva
  todavía subiendo.
- Mejor ensamble submiteable hoy: **mask+LR+LA+long = 0.6863** en valid, 41 min medidos. Zip armado
  (`ens4_mask_LR_LA_long_int8.zip`). Contra el 0.6843 que produjo nuestro mejor leaderboard (0.6542).

### Noche del 2026-09-30 — implementada la capa recurrente lenta (sprint 12)

Joaquín dejó corriendo las dos del sprint 11 y marcó, con razón, que eran "las últimas baratas" y no la apuesta
grande. La apuesta grande no necesita GPU libre ni que él esté despierto: necesita código. Así que la escribí.

`--slow-hidden 64 --slow-every K`: además de la red rápida de siempre, una **segunda red chica que corre una vez
cada K filas** sobre el promedio de los estados de la rápida, y devuelve un contexto lento a la salida. Las filas
de una ventana usan el contexto de la ventana **anterior**, así que es causal y se puede calcular online.

- **Costo en inferencia: +0,5 %** (1.152 MACs/fila sobre 254.208). Prácticamente gratis.
- Verificado en CPU antes de gastar GPU: estrictamente causal, y procesar por trozos da **exactamente** lo mismo
  que la secuencia entera (diferencia 0.00e+00). Entrena con destilación y con `--loss seqwp`; el checkpoint
  guarda los pesos de la lenta y su configuración.
- Notebooks: `connectome_sprint12_slow32.ipynb` y `_slow256.ipynb` (20 épocas, SWA 15–20, ~4 h cada uno).

Es el **primer experimento que cambia la arquitectura** en vez de ajustar algo alrededor de ella, y el único con
chance de romper la correlación de 0.96–0.98 entre los errores de todos nuestros modelos.

### Pendientes con potencial (viejos)
- Investigar `t1`: nuestra ventaja sobre el baseline es menor en `t1` (+0.04) que en `t0` (+0.09) (baseline t0 0.588 / t1 0.646; nosotros 0.677 / 0.686).
- Reconstruir niveles del libro con las medias por columna (hallazgo del QuantileTransformer) → spread, desequilibrio.
- **Bucle de generaciones:** mejor ensamble → alumna LSTM (lr 5e-4) → tabla → nuevo mejor ensamble → repetir. Cada vuelta ≈ +0.001. Después del jueves: 2 alumnas G2 en paralelo con semillas distintas.
- LA 9–12 y LB (12 ép.) quedan como opcionales de baja prioridad: la receta lr 5e-4 las reemplaza.
- Descomposición `d=(t0−t1)/2`, `c=(t0+t1)/2` como salidas/auxiliares.
- Entrenamiento final sobre train + valid con la receta fijada.
