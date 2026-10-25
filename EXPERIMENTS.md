# Registro de experimentos

> ⚠️ **2026-09-16: los organizadores CAMBIARON LA MÉTRICA.** Antes: WP por secuencia, promediada sobre secuencias.
> Ahora: **WP GLOBAL** — todas las filas scoreadas de todo el dataset juntas, una correlación ponderada por target, promedio de t0 y t1.
> Consecuencias: (1) `utils.BlockAccumulator` calcula la métrica vieja; la nueva está en `GlobalWP` (en `model/lgbm_deltas/` y `model/official_gpu/`).
> (2) La escala de las predicciones entre secuencias ahora importa. (3) Las filas con |target| grande dominan.
> **Baseline con la métrica nueva en los 10 grupos fijos: 0.6498** (t0 0.632, t1 0.668). Todos los números anteriores de esta tabla son con la métrica vieja.

Referencia: siempre comparar en los **mismos grupos de valid** (`--valid-groups 10 --seed 2026` → 10 grupos fijos).
El baseline oficial da **0.568** en esos 10 grupos (0.5896 en las 1.873 completas).

| id | fecha | modelo / cambio | train seqs | WP valid (10 g) | t0 | t1 | tiempo | máquina | notas |
|---|---|---|---|---|---|---|---|---|---|
| ref | 2026-09-12 | baseline.onnx oficial (GRU 128x2) | 10.607 | **0.568** | 0.531 | 0.605 | — | laptop | R² valid scored = −4.4: predice mal la escala, bien la dirección |
| 001 | 2026-09-12 | LSTM FirstTry tal cual (h64, 1 capa, MSE, 1 época, chunk 128) | 3 | 0.296 | 0.284 | 0.307 | 19 s | laptop | |
| 002 | 2026-09-12 | ídem | 20 | 0.336 | 0.295 | 0.377 | 25 s | laptop | |
| 003 | 2026-09-12 | ídem | 100 | 0.344 | 0.367 | 0.321 | 82 s | laptop | curva se aplana: el cuello es el entrenamiento, no los datos |
| 004 | 2026-09-12 | **loss alineada a la métrica** (clip ±2, peso \|y\|) en `first_try_torch.py`; resto igual | 100 | **0.404** | 0.419 | 0.389 | 82 s | laptop | +0.06 solo por la loss |
| 005 | 2026-09-12 | ídem | 500 | **0.458** | 0.423 | 0.492 | 366 s | laptop | con la loss nueva, más datos vuelve a ayudar |
| 006 | 2026-09-12 | ídem, **3 épocas** (run_20260912_190533) | 500 | **0.500** | 0.471 | 0.529 | ~18 min | laptop | +0.042 por épocas; train WP proxy 0.319 → no memoriza |
| 007 | 2026-09-15 | **`model/official_gpu/train_official_gpu_v1.py`** (nuevo): GRU 128x2, batch 8, chunk 256, loss wmse, clip 1.0, lr 1e-3, 2 épocas | 100 | **0.490** | 0.478 | 0.502 | ~9 min | laptop CPU | con solo 100 seqs iguala casi a exp 006 (500 seqs LSTM-64); ép.1 = 0.477. Script validado (resume, top-k, flags, loss wp) |
| 008 | 2026-09-15 | ídem, **primera corrida en Kaggle GPU (T4)**, batch 32, chunk 512, 1 época | 200 | 0.462 | 0.459 | 0.465 | **10 s** | Kaggle GPU | pipeline GPU validado: ~50× más rápido que CPU → 1 época con 10.607 seqs ≈ 8–10 min |
| **009** | 2026-09-15 | **Primera corrida completa en Kaggle**: GRU 128x2, batch 32, chunk 512, wmse, clip 1.0, lr 1e-3, lr-decay 0.9, 10 épocas (`full_gru128x2_10ep`) | **10.607 (todo)** | **0.5838** (ép. 4) | 0.573 | 0.595 | 64 min | Kaggle GPU T4 | **+0.016 sobre el baseline (0.568)**. Curva: 0.551 / 0.579 / 0.581 / **0.584** / 0.564 / 0.571 / 0.577 / 0.580 / 0.570 / 0.575 → meseta desde ép. 4; train loss sigue bajando (1.40→1.25) → empieza a sobreajustar. t1 siempre ~0.03 > t0. |

## Hallazgos de infraestructura (2026-09-12)

- Inferencia del baseline con ONNX Runtime: **82.7 µs/fila → 51.6 min** proyectados para un test del tamaño de valid. Al borde del límite de 60 min.
- Misma GRU reimplementada en numba: **27.7 µs/fila** (idéntica, diff máx 2e-5). Ver `connectome/src/gru_numba.py`. Abre margen para modelos más grandes.
- Targets: cuantizados (múltiplos de ~0.1254, escala varía por secuencia), corr(t0,t1) = −0.72 global, **nunca ambos > 0** en la misma fila, outliers hasta ±125 (3.1% fuera de ±2), 21.9% ≈ 0.
- Features: normalizadas (~[-2.4, 2.4]), sentinel −5.1993 en `i1_dv*` y `a7`.

### Análisis sin GPU del 2026-09-26 — dos puertas cerradas y dónde está el techo

Hecho en la laptop con las predicciones de valid ya guardadas (3.528.273 filas puntuadas de 1.873 secuencias).
Scripts: `model/submission/combine_local.py` (tabla de ensambles local) y los análisis del scratchpad.

| pregunta | resultado | conclusión |
|---|---|---|
| ¿Cuánto se gana si se **calibra la escala de cada secuencia**? (oráculo, usando los targets) | ensamble 0.6853 → **0.7053** solo con un factor por secuencia, **0.7319** con factor + corrimiento | Hay +0.047 teóricos ahí. Es **el techo más grande que existe**, mucho más que la distancia al puesto 1 (+0.008) |
| ¿Se puede estimar esa escala **sin ver los targets**? | normalizar por la media/desvío de las propias predicciones (con toda la secuencia o de forma causal): **0.6853 → 0.63–0.67, siempre peor** | No |
| ¿Se puede predecir el desvío del target de cada secuencia desde las features? | regresión sobre 500 secuencias, R² fuera de muestra: 0.08 (t0) y 0.15 (t1) con el desvío de las predicciones; **las features no agregan nada** (R² 0.05 y −0.01) | **No.** La escala por secuencia (varía x4 entre secuencias) es esencialmente ruido no observable → **el +0.047 es un espejismo, puerta cerrada** |
| ¿Sirve **ajustar los pesos** del promedio en vez de usar pesos iguales? | mínimos cuadrados ponderados por target, ajustado en la mitad de las secuencias y medido en la otra: **+0.0000** (4 modelos) y **−0.0003** (9 modelos) | No. El promedio simple ya es óptimo → puerta cerrada |
| ¿Cuán distintos son los miembros del ensamble? | correlación entre sus errores: **0.96–0.98** | La diversidad está agotada con esta familia de recetas; por eso el ensamble satura |
| ¿Dónde está el peso de la métrica? | filas con \|y\| grande: 35,7% de las filas pero **67,8% del peso**, WP 0.725 ahí; \|y\| chico: 14,9% de las filas, 4,1% del peso, WP 0.449 | Mejorar en las filas chicas casi no paga; todo se juega en las grandes |

**Lo que queda como palanca real:** el **tamaño de la alumna destilada**. Todas las alumnas fueron LSTM 128x2 y ninguna
alcanzó a su ensamble maestro (maestros 0.6843 → alumnas 0.6823 / 0.6834). Las pruebas de tamaño anteriores (160x2,
192x2, 200x1, 64x3) fueron **sin destilar y con lr 1e-3**, así que "alumna grande" nunca se probó → sprint 8.

**Tiempos MEDIDOS (no estimados) con pesos al azar, anclados a S7 = 35 min reales:** 1×LSTM192x2 = **20,5 min**,
1×LSTM256x2 = **31,1 min**, 2×LSTM192x2 = **39,0 min** (→ 256+128 ≈ 45). Paridad int8 vs PyTorch ~5e-4. Todo entra.
(La fórmula de `combine_local.py` sobreestima a las redes grandes: para 1×256 dice 37,6 min y la medición da 31,1.
Para decidir si un zip entra vale la medición de `time_anchored.py`, no la fórmula.)

### Correlación entre los errores: la medición que explica todo (2026-10-01)

Matriz de correlación entre los **errores** de 6 modelos sobre 271k filas puntuadas (t0):

| | ema | long | LR | LA | b256 | E |
|---|---|---|---|---|---|---|
| **ema** | 1.000 | 0.986 | 0.984 | 0.983 | 0.986 | 0.975 |
| **long** | | 1.000 | 0.986 | 0.984 | 0.987 | 0.976 |
| **LR** | | | 1.000 | 0.984 | 0.987 | 0.978 |
| **LA** | | | | 1.000 | 0.985 | 0.976 |
| **b256** | | | | | 1.000 | 0.976 |
| **E** | | | | | | 1.000 |

Todos entre **0.975 y 0.987**, sin importar entradas (112 vs 152), tamaño (128 vs 256), épocas (8 vs 20) ni celda
(la GRU E es la más distinta, 0.975–0.978, y por eso aparece en los mejores ensambles). **Cambiar las entradas no
genera diversidad**: el modelo con memoria larga se equivoca exactamente donde se equivocan los demás.

Esto reordena el criterio de éxito: un modelo nuevo sirve si **baja esa correlación**, aunque solo iguale en score.
Criterio para el sprint 12: correlación < 0.97 con la familia = primera diversidad real.

### Qué queda en el error del ensamble: búsqueda dirigida (2026-10-16)

El corrector buscó a ciegas con árboles y encontró +0.0003. Para descartar que una búsqueda a ciegas sobre
datos tan ruidosos se estuviera perdiendo algo chico pero sistemático, se midió **de frente** la correlación
entre el error del ensamble y los cambios de cada una de las 112 columnas a retrasos de 1 a 200 filas (120
secuencias de valid, ~370.000 filas puntuadas; el ruido a ese tamaño es ±0.003):

| columna | retraso | corr. con el error |
|---|---|---|
| **i1_v20** | 200 | **−0.0343** |
| i0_p9 | 50 | +0.0269 |
| i0_p15 | 50 | −0.0236 |
| i0_p2 | 20 | +0.0233 |
| i0_v4 | 100 | −0.0228 |
| i1_v4 | 50 | −0.0224 |

**Hay señal real, pero es chica.** Lo más alto (0.034) está 11 veces por encima del ruido, y la correlación
más alta de todas es de **i1**, el instrumento correlacionado, a 200 filas de retraso — justo el tipo de
relación "uno se mueve antes que el otro" que uno esperaría. Ninguna de esas columnas tiene hoy features de
memoria larga.

**El techo:** una correlación de 0.034 contra un error que se lleva el 88 % de la varianza vale
`0.034 × 0.87 ≈ 0.030` de correlación parcial, o sea `sqrt(0.686² + 0.030²) − 0.686 = +0.0007` en la métrica.
Todas juntas, descontando lo que se repite: **+0.001 a +0.002**. Consistente con el +0.0003 del corrector.
Se prueba en el sprint 19 (`--ema-plus`).

### El calentamiento NO es un problema (2026-10-16)

Cada secuencia reinicia el estado de la red y tiene 99 filas de calentamiento. Si a la red le costara agarrar
el ritmo, las primeras filas puntuadas saldrían peor. Medido sobre las 3.528.273 filas puntuadas de valid:

| posición en la secuencia | filas | puntaje |
|---|---|---|
| 100 – 1.000 | 73.470 | 0.69–0.70 |
| 1.000 – 2.000 | 179.267 | 0.6625 |
| 2.000 – 5.000 | 520.742 | 0.6951 |
| 5.000 – 10.000 | 876.499 | 0.6731 |
| 10.000 – 15.000 | 887.974 | 0.6967 |
| 15.000 – 20.000 | 901.580 | 0.6859 |

**No hay efecto de calentamiento:** las primeras filas (0.69–0.70) están entre las mejores, y el peor tramo
está en el medio (1.000–2.000 y 5.000–10.000). La variación es ruido entre tramos, no una curva que sube.
Queda descartado tocar la inicialización del estado o el manejo del calentamiento.

### El corrector: romper la pared de la correlación no alcanza (2026-10-15)

Todo lo que entrenamos hasta ahora fueron **competidores**: modelos que predicen el target por su cuenta y
después se promedian. Todos terminan correlacionando 0.92–0.99 con el ensamble. La alternativa estructural es
entrenar un **corrector**: un modelo que predice `y − predicción del ensamble`, o sea el error.

Se puede medir sin gastar GPU y sin tocar el entrenamiento, porque **las redes nunca vieron valid**: el error
del ensamble sobre valid es un error honesto, fuera de muestra. Se ajustó un LightGBM (190 columnas, 600
árboles) sobre las filas puntuadas de 300 secuencias de valid y se midió en **otras 300**:

| | prueba chica (60+60 secuencias, 86k filas) | prueba grande (300+300, 502k filas) |
|---|---|---|
| correlación del corrector con el ensamble | **−0.25** | **−0.14** |
| puntaje del ensamble solo | 0.6716 | 0.7195 |
| mejor mezcla (peso fijo) | 0.6735 (+0.0019, peso 0.25) | 0.7198 (**+0.0003**, peso 0.10) |
| ganancia honesta (peso elegido en una mitad, medido en la otra) | +0.0000 | **+0.0004** |

**Lo primero es un resultado real: es lo único en todo el proyecto que NO se parece al ensamble** (−0.14
contra 0.92–0.99 de todos los competidores). La construcción funciona exactamente como dice la teoría.

**Lo segundo es el resultado que importa: del otro lado de la pared no hay nada.** Con 6 veces más datos de
ajuste la ganancia *bajó* de +0.0019 a +0.0003, o sea que el +0.0019 de la prueba chica era ruido. El error
del ensamble se lleva el **88 %** de la varianza del target y es, hasta donde estas 112 features permiten ver,
**impredecible**.

**Qué significa para el proyecto:** la pared de la correlación no es un defecto de nuestra receta que se pueda
esquivar con ingeniería. Las redes ya extrajeron lo que estas features permiten extraer; lo que queda es
ruido. Esto cierra, con evidencia, toda la familia "agregar otro modelo al ensamble".

### No hay ninguna zona donde los árboles ganen (2026-10-15)

Antes de cerrar la rama se buscó si los árboles eran mejores en algún pedazo identificable de las filas, para
usarlos solo ahí. Dos cortes, los dos disponibles en inferencia, sobre 388.058 filas puntuadas:

| corte | ensamble | árboles | correlación | margen (s − S·r) |
|---|---|---|---|---|
| por \|predicción\| del ensamble, quintil 1 | 0.5575 | 0.5145 | 0.903 | +0.0111 |
| … quintil 3 | 0.4710 | 0.4449 | 0.920 | +0.0117 |
| … quintil 5 | 0.8759 | 0.8322 | 0.936 | +0.0124 |
| por volatilidad de la secuencia, quintil 1 | 0.6552 | 0.6245 | 0.927 | +0.0168 |
| … quintil 5 | 0.6521 | 0.6173 | 0.931 | +0.0101 |

**Es uniforme: los árboles pierden por lo mismo en todos lados y correlacionan lo mismo en todos lados.** No
hay región donde convenga darles más peso. Un peso por quintil elegido honestamente da **−0.0037**, peor que
el peso único (−0.0014): más libertad para elegir = más sesgo, que es la misma lección de siempre.

### El sprint 23 cierra el último frente: la correlación más baja que logramos, y +0.0001 (2026-10-25)

La normalización adaptativa por secuencia era lo único que quedaba sin probar, y era distinta de todo lo demás:
no agrega otro modelo del mismo tipo, cambia **qué ve** la red. Corrió una de las dos configuraciones.

| | puntaje solo | r con el ensamble |
|---|---|---|
| `adapt8000` (SWA) | 0.6801 | **0.9832** |
| `adapt8000` (best) | 0.6794 | 0.9817 |
| LR / LA / long / b256 | 0.6823 – 0.6843 | 0.9939 – 0.9955 |
| `div_a` / `div_c` / `div_d` (sprint 24) | 0.6777 – 0.6829 | 0.9835 – 0.9902 |

**Funcionó como diversificador: es la correlación más baja con el ensamble de cualquier red que entrenamos**
(0.9832 contra 0.9939 del mejor de los viejos), y contra los modelos individuales queda en 0.972–0.979 cuando
los cuatro viejos están en 0.9860 entre sí. La hipótesis de que cambiar la entrada diversifica más que cambiar
los datos o los maestros **se confirmó**.

Y no sirvió igual. Probadas **las 70 cuaternas** de los 8 modelos disponibles (cuatro es lo que entra en los
60 min de inferencia):

| cuaterna | valid |
|---|---|
| `long`+`b256`+`adapt8000`+`div_a` | 0.6861 |
| `LR`+`long`+`b256`+`adapt8000` | 0.6861 |
| **`LR`+`LA`+`long`+`b256` (la nuestra)** | **0.6860** |
| **ganancia** | **+0.0001** |

La nuestra queda **4ª de 70**, dentro del ruido de la mejor. Con cinco modelos (que no entran en presupuesto):
+0.0002.

**Esto cierra el último frente técnico que teníamos identificado.** Y deja el resultado más limpio de todo el
proyecto sobre por qué: en cuatro sprints bajamos la correlación de 0.9955 a 0.9832 con la palanca más
agresiva que existe —cambiar la representación de entrada— y la ganancia fue +0.0001. **La correlación de
~0.98 no es un defecto de nuestra receta: es cuánta información distinta hay para extraer de estos datos con
una red recurrente de este tamaño.**

**La corrida que murió, y la lección sobre la guarda.** `adapt2000` gastó **695 minutos solo precalculando los
soft targets** y lo mató el límite de 12 h antes de la primera época. `adapt8000`, con el **mismo código**,
tardó **37 minutos** en ese mismo paso: 18,8 veces menos. La guarda de velocidad midió 0,05 s por secuencia al
arrancar y la dejó pasar, porque **el frenazo apareció durante la corrida, no al principio**. La guarda está
bien puesta pero mide el arranque; para cubrir esto haría falta un control de presupuesto **dentro** del bucle
de soft targets que aborte si la proyección pasa de las 12 h. Queda anotado: es la cuarta corrida que perdemos
por suerte de máquina.

### El +0.047 de septiembre es un oráculo, no una puerta (2026-10-24)

Desde el 2026-09-26 el número más grande del proyecto era este: **si supiéramos la escala de cada secuencia, el
ensamble pasaría de 0.6853 a 0.7053. +0.047**, seis veces la distancia al puesto 1. Lo había cerrado como
imposible probando **una** sola cosa —estimar la escala desde las propias predicciones— y quedó anotado como
"el techo más grande que existe". Esta vez lo abrí de verdad, en la laptop, y resultó que **estaba mal
etiquetado desde el principio**.

**Primero: no es escala.** El oráculo de septiembre aplicaba un factor **y un corrimiento** por secuencia. Si se
aplica solo el factor que iguala la escala del target (σ_y/σ_q), el resultado es:

| oráculo | efecto |
|---|---|
| igualar la escala exacta (σ_y/σ_q) | **−0.0566** |
| solo el factor óptimo de regresión | −0.0314 |
| solo el corrimiento óptimo | +0.0046 |
| **factor + corrimiento** | **+0.0470** |

**Igualar escalas empeora.** El +0.047 sale de la **interacción** entre las dos piezas, no de ninguna de las dos.
Y la pieza que de verdad manda es el corrimiento, que es `b_s = my_s − a_s·mq_s`, donde `my_s` es el sesgo
direccional de los targets de la secuencia: a dónde se movió en agregado. Eso es casi saber la respuesta.

**Segundo: las dos piezas son impredecibles.** Con un resumen de las primeras 2.000 filas de cada secuencia
(media, desvío y variación media de las 112 features, más 6 números de nuestras propias predicciones, que en
inferencia también tenemos), ridge ajustado en la mitad de las secuencias y medido en la otra:

| qué se intenta predecir | R² fuera de muestra |
|---|---|
| el factor `a_s` | **+0.003** |
| el corrimiento `b_s` | **+0.003** |
| la escala del target `σ_y(s)` | +0.20 / +0.26 (solo +0.08 con las primeras 2.000 filas) |

Y aplicado de verdad, nada mejora: el mejor caso es +0.0001, que es el límite donde la corrección se apaga.
Para ver si alguna de las dos mitades bloquea a la otra, las combiné con el oráculo:

| | resultado |
|---|---|
| factor ORÁCULO + corrimiento predicho | −0.0316 |
| factor predicho + corrimiento ORÁCULO | +0.0035 |

Ninguna mitad sola llega a nada. **El +0.047 necesita las dos piezas exactas a la vez: es un oráculo, no una
puerta.** La conclusión de septiembre era correcta; el motivo era otro, y ahora está medido.

**Tercero, y esto sí es un hallazgo: la confianza por secuencia se puede predecir, y bien.** La correlación
`r_s` de nuestro ensamble dentro de cada secuencia (mediana 0.702, p10 0.333, p90 0.865 — varía muchísimo) sale
con **R² = +0.51 solo con las features del libro y +0.57 agregando el resumen de nuestras propias
predicciones**, fuera de muestra. Es la señal más fuerte que encontramos en todo el proyecto: **sabemos de
antemano en qué secuencias vamos a predecir bien.**

Lo que no sabemos es qué hacer con eso. Usarla como multiplicador —encoger las predicciones donde la confianza
es baja— **empeora, y empeora incluso con el oráculo de `r_s` (−0.0244)**. Tiene sentido con la métrica que nos
puntúan: la WP global premia mantener una escala consistente entre secuencias, y encoger unas y no otras rompe
esa consistencia. Queda anotado como lo único con señal fuerte sin explotar.

**Consecuencia para el sprint 23, y es una corrección a lo que escribí el 2026-10-20.** La ficha del exp 040
dice que la normalización adaptativa "conecta con el oráculo de +0.047, atacándolo desde la entrada en vez de la
salida". **Esa conexión es falsa**: el oráculo no es de escala, así que el sprint 23 no hereda ese techo. Su
justificación queda en pie por sí sola y es más modesta —97 de las 112 features cambian de escala más de 2x
entre secuencias y eso le dificulta el aprendizaje a la red— pero hay que leerla sin el +0.047 detrás.

### La regla `s > S·r` falla, y el sprint 24 es la prueba (2026-10-22)

Corrimos tres de los cuatro alumnos del sprint 24 (`div_b` quedó cancelado por falta de cuota). Cada uno vio
**datos distintos** (70 % de las secuencias, semilla propia), **maestros distintos** (un par de los 4) y
**receta distinta**; `div_d` además es **GRU** en vez de LSTM.

**Lo primero: qué palanca bajó la correlación y qué palanca no.**

| par | qué cambia entre ellos | correlación |
|---|---|---|
| `div_a` – `div_c` | datos, maestros, lr — **las dos son LSTM** | **0.9849** |
| `div_a` – `div_d` | lo mismo **+ LSTM contra GRU** | **0.9769** |
| `div_c` – `div_d` | lo mismo **+ LSTM contra GRU** | **0.9768** |
| los 4 viejos entre sí | solo la semilla | 0.9860 |

**Cambiar datos y maestros no mueve la correlación (0.9849 contra 0.9860: 0.0011). Cambiar la arquitectura sí
la mueve (0.977).** La familia nueva promedia 0.9795, 0.0065 abajo — y todo ese 0.0065 lo puso el GRU.

Eso **refuta la hipótesis con la que lanzamos el sprint**, que decía que la correlación venía de compartir los
maestros. No: los 4 maestros son ellos mismos casi idénticos (~0.98 entre sí), así que darle a cada alumno un
par distinto es darle **el mismo objetivo**.

**Lo segundo, y es lo grave: los tres pasan la regla y la ganancia es cero.**

| modelo | puntaje `s` | corr. con el ensamble `r` | `S·r` | margen | veredicto de la regla |
|---|---|---|---|---|---|
| `div_a` | 0.6829 | 0.9902 | 0.6793 | **+0.0035** | PASA |
| `div_c` | 0.6819 | 0.9900 | 0.6792 | **+0.0027** | PASA |
| `div_d` | 0.6777 | 0.9835 | 0.6747 | **+0.0030** | PASA |

Y la medición de verdad, sobre las 3.528.273 filas puntuadas, probando **todas** las cuaternas de los 7 modelos
(cuatro es lo que entra en los 60 min de inferencia):

| ensamble | valid | leaderboard |
|---|---|---|
| mejor cuaterna de las 35: `div_a`+LR+long+b256 | 0.6861 | 0.6538 |
| **el nuestro: LR+LA+long+b256** | **0.6860** | **0.6537** |
| **ganancia real** | **+0.0000** | |

Con pesos no uniformes, buscados con 300 sorteos: **+0.0000** en la cuaterna. Con los 7 juntos y pesos, +0.0002
—elegidos mirando el mismo valid con el que se miden, así que está sobreestimado, y siete modelos no entran en
el presupuesto de todos modos.

**La conclusión que vale para lo que queda del proyecto: `s > S·r` no sirve como criterio de decisión.** Es la
segunda vez que falla en la misma dirección y ya no es casualidad:

| modelo | margen sobre `S·r` | ganancia real |
|---|---|---|
| Echo State Network (exp 041) | +0.0184 | −0.0019 |
| `div_a` / `div_c` / `div_d` (exp 042) | +0.0027 a +0.0035 | +0.0000 |

La regla sale de suponer que los errores de los modelos son intercambiables; acá no lo son. **De ahora en más
la única evidencia que cuenta es promediar y medir.** Las decisiones que tomamos con la regla en sprints
anteriores hay que leerlas con eso en mente.

**Esto cierra la rama del ensamble**, la última que quedaba después de la de arquitecturas. El mapa de todo lo
intentado: cambiar la **arquitectura** baja la correlación hasta 0.88 y da cero; cambiar **datos, maestros y
receta** no la baja y da cero.

**Y deja un hallazgo lateral que vale por sí solo:** `div_a` sacó 0.6829 con el **70 % de los datos y 2
maestros** contra 0.6834 del equivalente con el 100 % y 4. **El volumen de datos y la cantidad de maestros casi
no importan**, lo mismo que ya había dicho la prueba de train+valid (+17,7 % de datos = 0.0000). Lo que falta no
es más de lo mismo: es **información que todavía no le damos a la red**, y el único candidato medido que queda
es la normalización por secuencia (exp 040).

### El leaderboard real y lo que vale cada milésima (2026-10-21)

Fuimos al leaderboard completo por primera vez. **El campo es densísimo:**

| puesto | puntaje |
|---|---|
| 1 | 0.67380 |
| 50 | 0.66517 |
| 100 | 0.66256 |
| 200 | 0.65934 |
| 300 | 0.65685 |
| 400 | 0.65491 |
| **nosotros** | **0.65400** |

**De #400 a #100 hay 0.0077.** Con nuestra brecha medida valid → leaderboard de 0.0323:

| mejora en valid | puesto |
|---|---|
| +0.001 | ~396 |
| +0.002 | ~352 |
| +0.003 | ~294 |
| +0.005 | ~214 |
| +0.008 | ~118 |

**Esto corrige un error de criterio que venía arrastrando:** descarté sistemáticamente mejoras de +0.002
llamándolas "ruido". Son ruido *para medirlas en valid*, pero valen **60 posiciones**.

Otro dato del leaderboard: los equipos del top 100 llevan entre 20 y 193 submissions. **Nosotros llevamos 8**,
teniendo 5 por día disponibles.

Y se verificó una sospecha: la página dice "N≈200 features" y nosotros usamos 112. El parquet tiene 118
columnas = 112 features + 2 targets + 4 de control. **No hay nada escondido.**

### Destilar nos costó la mitad del ensamble (2026-10-21)

Nuestro ensamble de 4 gana apenas **+0.0021** sobre el mejor modelo solo. En una competencia normal promediar
4 modelos vale entre +0.005 y +0.015. La causa estaba a la vista desde el sprint 15 y nunca la medimos de
frente: **los 4 están destilados de los MISMOS 4 maestros**, así que son casi el mismo modelo.

Medido sobre todo valid, comparando con 4 modelos **sin destilar** que teníamos guardados del sprint 4:

| familia | mejor solo | correlación entre ellos | ensamble de 4 | lo que gana promediar |
|---|---|---|---|---|
| destilados | 0.6843 | **0.9854** | **0.6864** | +0.0021 |
| sin destilar | 0.6794 | **0.9635** | 0.6845 | **+0.0051** |

**El mecanismo queda confirmado: sin destilar, promediar vale 2,4 veces más.** Pero los que tenemos sin
destilar son del sprint 4 (8 épocas, receta vieja, sin SWA) y arrancan 0.005 más abajo, así que su ensamble
termina *debajo* del nuestro. Mezclar las dos familias da +0.0004: la correlación cruzada es 0.9739.

**La comparación justa nunca se hizo:** sin destilar pero con la receta de ahora. Proyectando con el factor
medido (×1.00974 contra ×1.00424):

| si cada modelo diera | ensamble de 4 | leaderboard | puesto |
|---|---|---|---|
| 0.6808 (una LSTM sin destilar bien entrenada) | 0.6874 | 0.6551 | ~391 |
| 0.6820 | 0.6886 | 0.6563 | ~331 |
| 0.6835 | 0.6902 | 0.6579 | ~262 |

**Y lo que falta probar, que es lo que de verdad sigue:** nunca variamos **nada** entre los modelos del
ensamble. Todos vieron los mismos datos, los mismos maestros y la misma receta; lo único distinto fue la
semilla. Las tres palancas estándar de diversidad están sin usar:

1. **maestros distintos por alumno** (subconjuntos disjuntos de los 4) — conserva el +0.003 de destilar sin
   que todos converjan al mismo objetivo
2. **datos distintos por modelo** (bagging: 70 % de las secuencias, distinto para cada uno)
3. recetas distintas (tamaños y tasas de aprendizaje), que es lo único que sí variamos a medias

**Las tres se probaron al día siguiente, juntas (exp 042): los datos y los maestros no movieron la correlación; solo la movió cambiar la arquitectura, y la ganancia fue +0.0000. La sección del 2026-10-22 dice por qué, y por qué la regla `s > S·r` con la que decidimos todo esto no sirve.**

### La regularidad que cierra la rama de arquitecturas (2026-10-21)

Con la Echo State Network quedan cinco familias medidas **en las mismas 200 secuencias de valid**, contra el
mismo ensamble de 4 redes (S = 0.6755). La columna que importa es `s/r`, porque el criterio `s > S·r` se puede
escribir **`s/r > S`**:

| modelo | puntaje (s) | correlación (r) | **s/r** | margen (s − S·r) | aporte honesto |
|---|---|---|---|---|---|
| otra LSTM (train+valid) | 0.6729 | 0.9951 | **0.6762** | +0.0007 | +0.0000 |
| LightGBM | 0.6390 | 0.9271 | **0.6892** | +0.0127 | −0.0014 |
| XGBoost | 0.6405 | 0.9279 | **0.6903** | +0.0137 | −0.0015 |
| Echo State Network | 0.6099 | 0.8756 | **0.6966** | +0.0184 | −0.0019 |
| *el ensamble* | *0.6755* | | | | |

**`s/r` es casi constante — entre 0.676 y 0.697 — para cuatro familias de modelos radicalmente distintas**
(una LSTM entrenada por gradiente, árboles con boosting por hojas, árboles por niveles, y un reservorio
recurrente aleatorio que no se entrena). Y el ensamble está en 0.6755, justo en el medio de ese rango.

Dicho de otra forma: **cada modelo que logramos construir cae exactamente sobre la línea de empate.** Los que
son más distintos (r más bajo) son peores en la misma proporción, y las dos cosas se cancelan.

Y hay un segundo efecto que termina de explicar los números negativos: aunque el margen sea nominalmente
positivo (+0.013 a +0.018), **elegir honestamente cuánto peso darle al modelo nuevo cuesta más ruido que lo
que el margen vale**. Por eso los aportes medidos son −0.0014, −0.0015 y −0.0019 en vez de +0.001.

**Conclusión: la rama de arquitecturas queda cerrada con medición, no con argumento.** No es que no hayamos
encontrado el modelo distinto: lo encontramos (r = 0.876) y tampoco alcanza.

### El bug que costó 24 h de cuota de GPU (2026-10-20)

**Kaggle monta el mismo dataset en DOS rutas** y una es ~70 veces más lenta para leer:

| ruta | velocidad leyendo secuencias |
|---|---|
| `/kaggle/input/datasets/<usuario>/<dataset>/archivo.parquet` | **0,06 s** por secuencia |
| `/kaggle/input/<dataset>/archivo.parquet` | **4,5 s** por secuencia |

Los notebooks viejos tenían la ruta rápida escrita a mano. Cuando el sprint 19 falló porque Kaggle no montaba
el dataset en la ruta esperada, lo arreglé con una función `buscar()` que localiza el archivo por nombre —
pero devolvía `hits[0]`, o sea **cualquiera de las dos rutas**, sin criterio.

Resultado: `chunk1024` agarró la rápida y terminó en 128 min; `adapt2000` y `chunk2048` agarraron la lenta,
se pasaron 700 minutos **solo precalculando los targets del maestro** y murieron en el límite de 12 h de
Kaggle **sin entrenar una sola época**. Entre las dos se llevaron 24 de las 30 horas semanales de GPU.

Arreglado: `buscar()` ahora prefiere explícitamente la ruta que contiene `/datasets/` e imprime cuál eligió.
**Lección: cuando un arreglo reemplaza un valor fijo por una búsqueda, hay que definir qué pasa si hay más de
un resultado.** La búsqueda parecía más robusta que la ruta a mano y era menos robusta.

### Cada secuencia tiene su propia escala y se la estábamos escondiendo a la red (2026-10-20)

Medido sobre 150 secuencias de valid, mirando cuánto cambia el desvío de cada feature de una secuencia a otra:

| | |
|---|---|
| features que cambian de escala **más de 2x** entre secuencias | **97 de 112** |
| más de 3x | 77 |
| mediana | **3,87x** |
| las peores (`i1_p9`, `i0_p9`) | **10x** |

Y desde el experimento 007 normalizamos con **una media y un desvío globales**, calculados una sola vez sobre
una muestra de train. A la red le llegan números hasta 4 veces más grandes según en qué secuencia esté, y
tiene que gastar capacidad en darse cuenta sola y compensarlo.

**Esto conecta con el hallazgo del oráculo.** En septiembre medimos que multiplicar las predicciones de cada
secuencia por el factor correcto valdría **+0.047**, y nunca pudimos estimar ese factor desde las features. La
normalización adaptativa ataca el mismo problema por el otro lado: en vez de corregir la salida, pone la
entrada de todas las secuencias en la misma escala.

`--adapt-span N` normaliza con una media y un desvío móviles **dentro de la secuencia**, causales (solo miran
el pasado), inicializados en los valores globales para que las primeras filas queden como hoy. Verificado en
la laptop, mirando lo que ve la red:

| | desvío típico | variación entre secuencias |
|---|---|---|
| normalización global (hoy) | 0.74 | **4,07x** |
| adaptativa (memoria 2.000) | 0.98 | **1,09x** |

Costo: ~18 min por época (la corrida pasa de ~2 h a ~5 h). En inferencia son dos cuentas por fila y feature,
en el mismo lugar donde el kernel ya resta la media y divide por el desvío.

### El presupuesto de inferencia sobrante tampoco sirve (2026-10-20)

Nuestro mejor ensamble usa 42 de los 60 minutos. Quedan 18 minutos libres, o sea lugar para un quinto modelo.
Medido con `combine_local.py` sobre todo valid:

| ensamble | valid | minutos |
|---|---|---|
| LR+LA+long+mask (4 modelos) | **0.6863** | 41,8 |
| LR+LA+long+mask+l1 (5 modelos) | 0.6862 | 51,0 |

**Agregar un quinto modelo da −0.0001.** Es la misma pared de siempre: el quinto correlaciona 0.98 con los
otros cuatro, así que promediar más no promedia nada nuevo. El presupuesto sobrante no es una oportunidad
desaprovechada; simplemente no hay con qué llenarlo.

### El leaderboard está clavado en 0.654 (2026-10-15)

Cuatro submissions desde el 19 de septiembre, con el valid subiendo todo el tiempo y el leaderboard quieto:

| submission | valid | leaderboard | brecha |
|---|---|---|---|
| S7 (19/09) | 0.6843 | **0.6542** | 0.0301 |
| S8 (21/09) | 0.6855 | 0.6541 | 0.0314 |
| S9 (06/10) | 0.6857 | 0.6533 | 0.0324 |
| S10 (09/10) | **0.6863** | 0.6540 | 0.0323 |

Valid subió **+0.0020**; el leaderboard se movió **−0.0002**. Antes de S7 los deltas se trasladaban 1:1 y la
brecha era constante en 0.0301; desde S7 la brecha crece exactamente lo que ganamos en valid.

**Conclusión:** todo lo que ganamos en valid en el último mes es sesgo de selección acumulado (la época, los
hiperparámetros y la combinación, todo elegido mirando valid). Para que el leaderboard se mueva hace falta algo
que sea mejor **por una razón que valid no eligió**: o un modelo estructuralmente distinto (los árboles,
r = 0.918) o atacar la brecha misma (entrenar con train+valid en noviembre).

### La regla que explica todos los sprints muertos (2026-10-13)

Para un ensamble con puntaje **S** y un modelo nuevo con puntaje **s** y correlación **r** con las predicciones
de ese ensamble, la mezcla `e + a·n` tiene puntaje `(S + a·s) / sqrt(1 + 2·a·r + a²)`. Derivando en `a = 0`:

> **el modelo nuevo aporta  ⟺  s > S · r**

No importa que el modelo nuevo sea "distinto" en abstracto, ni que tenga otra arquitectura. Importa que su
puntaje supere al del ensamble **multiplicado por lo parecido que es**. Medido sobre 10 secuencias de valid
con las predicciones guardadas (22.198 filas puntuadas), con el ensamble de 4 redes en S = 0.6752:

| modelo nuevo | r con el ensamble | necesita pasar de | dio | medido en el ensamble |
|---|---|---|---|---|
| otra LSTM de la familia | 0.985 | 0.665 | ~0.684 | tendría que igualar al ensamble entero |
| convolucional (sprint 15B) | 0.978 | 0.661 | 0.654 | **+0.0000** (le faltaron 0.007) |
| **árboles LightGBM (exp 012)** | **0.918** | **0.620** | 0.625 | **+0.0001** (justo en el límite) |

**Esto explica los sprints 12, 13, 14, 15 y 16 de una sola vez.** Con r = 0.985, cualquier red nueva tiene que
igualar al ensamble completo para aportar algo; por eso nada se movía, y por eso la convolucional, que era
genuinamente otra arquitectura, tampoco: 0.978 sigue siendo demasiado parecido.

**Y señala dónde mirar.** La correlación de los árboles con el ensamble es 0.918, lo más bajo de todo el
proyecto, y como la pendiente del aporte crece con `1 − r`, ahí un avance chico vale mucho:

| si los árboles dieran | peso óptimo | ganancia en el ensamble |
|---|---|---|
| 0.625 (lo de septiembre) | 0.05 | +0.0001 |
| 0.640 | 0.23 | +0.0019 |
| 0.650 | 0.38 | **+0.0042** |
| 0.660 | 0.58 | **+0.0075** |
| 0.670 | 0.80 | **+0.0117** |

No hace falta que los árboles se acerquen a las redes: con que suban 0.025 es la mejora más grande en un mes.

### El costo real de los árboles en inferencia (medido 2026-10-13)

La nota del 2026-09-16 decía "~100 µs/fila" y el docstring de `lgbm_numba.py` decía "5-10 µs". Medido con el
kernel numba sobre el modelo de 600 árboles de septiembre: **58 µs/fila por target**, o sea **116 µs/fila**
para t0 y t1. El docstring estaba mal por un factor 10; la nota del diario era correcta.

Presupuesto: 96 µs/fila (60 min para 37,46M filas). Nuestro ensamble de 4 redes usa ~66 µs/fila (41 min
medidos) → quedan **30 µs/fila = 156 árboles por target**. Con 156 árboles el modelo no llega.

**La salida es la máscara.** La métrica puntúa el 9,3 % de las filas y el modelo de la máscara (AUC 0.794) sabe
cuáles son. Corriendo los árboles solo en las filas más probables:

| árboles corridos en el … % de las filas | atrapa … de las puntuadas | … del PESO de la métrica | árboles que entran |
|---|---|---|---|
| 10 % | 38,1 % | 42,7 % | 1.557 |
| 20 % | 57,7 % | 62,0 % | 779 |
| **25 %** | 64,9 % | **68,7 %** | **623** |
| 33 % | 74,3 % | 77,2 % | 472 |
| 100 % | 100 % | 100 % | 156 |

Con el 25 % de cobertura entran los 600 árboles completos y se cubre el 68,7 % del peso de la métrica. Las
filas no cubiertas se predicen con las redes solas, como hoy. Esto se implementa **después** de saber si los
árboles mejoraron (sprint 18); sin mejora no hay nada que meter en el kernel.

### Las columnas centinela (2026-10-09) — un defecto real de nuestro pipeline, con efecto no medible

Buscando qué quedaba sin tocar, se revisaron **las 112 columnas de entrada una por una** buscando valores
repetidos que inflaran el desvío. Aparecieron exactamente 3, todas con el mismo valor **−5,1993**, que es la
marca de "este dato no existe":

| columna | filas con el centinela | desvío real | desvío que ve la red | inflado |
|---|---|---|---|---|
| a7 | **44,0 %** | 0,583 | 2,934 | **5,0x** |
| a6 | 17,5 % | 0,808 | 2,154 | 2,7x |
| a5 | 10,9 % | 0,816 | 1,903 | 2,3x |

Como estandarizamos dividiendo por ese desvío, **la variación real de a7 le llega a la red comprimida a una
quinta parte**. Y a7 no es una columna cualquiera: sobre las filas puntuadas, **sin** los centinelas
correlaciona **0,21** con t1 (la correlación más alta de una sola feature medida en todo el proyecto); tal como
se la pasamos, correlaciona **0,04**. Esto estuvo así en **todas** las corridas del proyecto: la bandera
`--sentinel-flags` existe desde agosto y nunca se usó en una corrida en serio.

**Pero las dos pruebas baratas dicen que no cambia nada:**

| prueba | crudo | con banderas | diferencia |
|---|---|---|---|
| regresión lineal pesada, ajustada en 6M filas de train, medida en las 374.199 filas puntuadas de valid | 0.5872 | 0.5865 | −0.0007 |
| perceptrón por fila (2 capas de 128), 1,5M filas, 2 semillas | 0.5480 | 0.5449 | −0.0031 (ruido entre semillas: 0.0081) |

La lineal no sirve para esta pregunta y hay que decirlo: la columna cruda vale `a7_limpio*(1−S) − 5,1993*S`,
que es una combinación lineal de las dos columnas que armaría la codificación con banderas, así que el modelo
lineal absorbe el desvío inflado en su coeficiente y no sufre el problema. El perceptrón **sí** lo sufre (es no
lineal y usa entradas estandarizadas) y tampoco se movió. La explicación más probable: la compuerta
"si a7 = −5,2, ignorála" es fácil de aprender porque el centinela es un valor único y afilado, y una vez
aprendida, la escala inflada es una constante que la primera capa reescala sola.

**Conclusión honesta:** el defecto es real y está medido, el efecto no se detecta con las herramientas baratas,
y la resolución de esas pruebas (±0,008) no alcanza para descartar el +0,002 que nos importaría. Se corre igual
(sprint 17A) porque cuesta 2 h en una sesión que de otro modo queda ociosa, pero **la expectativa es baja**.

### El modelo de la máscara (2026-10-07) — `model/official_gpu/mask_coef.json`

La métrica solo puntúa las filas con `is_scored` (**9,4 %**), vienen en rachas cortas, y **esa columna no existe
en train**. Veníamos entrenando parejo sobre las 20.000 filas de cada secuencia.

Regresión logística sobre las 112 features crudas, ajustada en 150 secuencias de valid y medida en otras 60:
**AUC 0.794** fuera de muestra. La separación es fuerte:

| decil de probabilidad | filas realmente puntuadas |
|---|---|
| más bajo | 0,5 % |
| medio | 12,0 % |
| más alto | **34,9 %** |

`--mask-coef mask_coef.json --mask-power 1.0` cambia el peso de cada fila a `|y| × (p / p_medio)^potencia`.
Con potencia 1 una fila del decil alto pesa ~6 veces más que una del decil bajo. Es el único cambio que alinea
lo que optimizamos con lo que nos puntúan. Se prueba en el sprint 15A.

### Sprint 12 (2026-09-30) — capa recurrente LENTA: lo primero que cambia la arquitectura

Implementado en el entrenador: `--slow-hidden N --slow-every K`. Además de la red rápida de siempre (una pasada
por fila), corre una **segunda red de N unidades una vez cada K filas**, alimentada con el **promedio de los
estados de la rápida** durante esas K filas; su estado vuelve como contexto concatenado a la salida. Las filas de
una ventana usan el contexto calculado al cerrar la ventana **anterior**: es causal y se puede calcular online en
el kernel (un acumulador + una red chica cada K filas).

Ataca las dos causas posibles del techo: (1) el BPTT truncado en 512 pasos impide aprender dependencias de miles
de filas; (2) los errores de todos nuestros modelos correlacionan 0.96–0.98, así que hace falta un modelo con otra
estructura para que el ensamble vuelva a sumar.

**Costo en inferencia: 1.152 MACs/fila = +0,5 %** sobre la LSTM 128x2 (254.208). Prácticamente gratis.

Verificado en CPU antes de gastar GPU: estrictamente causal (cambiar el futuro no altera el pasado), y procesar
por trozos da **exactamente** el mismo resultado que la secuencia entera (diferencia 0.00e+00). Entrena con
destilación y con `--loss seqwp`; el checkpoint guarda los pesos de la lenta y su configuración.

Dos corridas (`make_sprint12.py`), 20 épocas, SWA 15–20, destiladas de los 4 maestros: `slow32` (lenta cada 32
filas, seed 2121) y `slow256` (cada 256 filas: contexto mucho más lento, seed 2122).
**Si gana ≥ 0.687 hay que implementar la capa lenta en `solution.py` antes de poder submitear.**

### Sprint 10 (2026-09-29) — la loss: entrenamos optimizando lo que NO nos puntúan

Hallazgo por investigación, no por prueba y error: el competidor **dozken** (puesto ~51, LB 0.6588) tiene
[repositorio público](https://github.com/dozken/wunder). Su arquitectura es **GRU 192x2 + proyección de 128**
(peor que la nuestra: nuestra LSTM 192x2 da 0.6834 en valid) pero su entrenamiento usa
**"sequence-level weighted Pearson on soft-masked rows"** (`--seq-loss --loss-mask soft`), y su archivo de
trabajo advierte: *"per-chunk Pearson calculations diverge from the actual competition metric"*.

El punto: **la métrica es una correlación ponderada, que es invariante a la escala**. Entrenando con WMSE, el
modelo gasta capacidad en acertar la magnitud de la predicción, que la métrica ignora por completo. Lo habíamos
probado (exp 018) pero mal, y de la forma exacta que ese archivo desaconseja: peso 0.05 sobre trozos de 512 filas.

**`--loss seqwp`** (nuevo en el entrenador): acumula los seis estadísticos suficientes de la correlación
(Σw, Σwy, Σwp, Σwy², Σwp², Σwyp) **a lo largo de toda la secuencia**, con olvido exponencial (`--seqwp-decay`,
0.98) para que el trozo actual siga aportando gradiente. La normalización refleja la secuencia entera, no un
trozo. `--wmse-anchor` agrega un poco de error cuadrático para que la escala de la salida no quede libre.

Dos corridas (`make_sprint10.py`): `seqwp` (sin destilar, ancla 0.1, seed 2101) y `seqwp_dist` (misma receta
exacta que la alumna LR pero con esta loss, ancla 0.2, seed 2102) → **comparación limpia contra LR = 0.6834**.
**Resultado (exp 028): 0.6837 vs 0.6834. No aporta.** Optimizar la correlación en vez del error cuadrático no cambia
nada en este problema, ni sola ni en ensamble.

También se midió, por el "soft-masked rows": **`is_scored` (qué filas puntúa la métrica) es parcialmente
predecible desde las features** — AUC 0.763 fuera de muestra con una regresión logística sobre las 112 crudas;
ningún `aN` solo pasa de 0.60. Las filas puntuadas son el 10,9 %, vienen en rachas de 2–4 filas y tienen
|target| algo mayor (0.655 vs 0.608). **`train.parquet` NO trae `is_scored`** (solo valid), así que para usarlo
habría que generar una máscara aproximada. Queda como segunda idea si la loss funciona.

### Sprint 9 (2026-09-27) — features cruzadas: el último eje de entrada sin probar

Los tres intentos de features que fracasaron (2026-09-13 y 09-16/17) eran **lags y deltas en el tiempo**, que una red
recurrente ya construye sola. Y **sumas o restas entre columnas tampoco pueden servir**: la primera capa de la red ya es
una combinación lineal de las 112 entradas, así que agregarlas como entradas no aporta información (esto casi se me pasa:
la primera versión de `cross_features` eran todas sumas y restas — spread, mid, huecos — y habría sido una corrida tirada).

Lo que una red con 112 entradas correlacionadas sí aprende mal: **cocientes, desvíos, máximos y productos**. `--cross-features`
del entrenador agrega 20 de esas, calculadas dentro de cada fila (causales, baratas en inferencia): desequilibrio de volumen
compra/venta como cociente (top, 3 niveles, total), fracción del volumen del lado comprador, desvío de precios y volúmenes
entre los 11 niveles, profundidad relativa al spread, spread relativo al mid, máximo y concentración del volumen en un nivel,
desequilibrio del segundo instrumento y su diferencia con el primero, y dos productos. Comparación limpia: misma receta que
la alumna LR (**0.6834**), solo cambian las entradas.

### Estimador de tiempo de `eval_ensemble.py` recalibrado (2026-09-21)

Único tiempo real medido con el kernel v3 int8: **S7 (LSTM128x2 + 3 GRU128x2 = 825k MACs/fila) = 35m05s = 56 µs/fila** en el servidor.
La fórmula vieja (`2 + 0.091·kMAC`, ×1.15, tope 55) le daba 77 µs y marcaba "NO entra" a combinaciones que sí entran.
Nueva: `µs = 8 + 0.058·kMAC`, entra si ≤ 50 min (10 de margen). Consecuencias: **4 LSTM128 ≈ 42 min, 3 LSTM128 + GRU192 ≈ 47 min, 2 LSTM128 + LSTM160 + GRU128 ≈ 44 min: todas entran.**

### Kernel de inferencia v3 (2026-09-19) — LSTM

`solution.py` acepta GRU y LSTM por modelo (`cell_per_model`; arrays rellenados a 4·Hmax compuertas; estados c0/c1 para LSTM;
compuertas LSTM en el orden PyTorch i,f,g,o sin reordenar). `build_submission.py` exporta ambos (`load_any` devuelve la celda; `make_net` para la referencia).
Costo LSTM 128x2 = 254k MACs (4/3 de una GRU 128x2). Medido anclado a S4: y2+Eswa+X = 30 min, **y2+Eswa+X+Zswa = 38,5 min**.
`kernel_bench/time_anchored.py` automatiza la medición anclada.

### Kernel de inferencia v2 (2026-09-18) — `model/submission/solution.py`

Diagnóstico con benchmarks round-robin (laptop Ryzen 7735U, 3×GRU128x2, un núcleo fijo; `model/submission/kernel_bench/kbench6.py`):
el kernel v1 **ya vectorizaba** (AVX2 FMA), pero (a) cada fila de la matriz era una reducción horizontal corta (112–128 elementos),
(b) los pesos float32 (760 KB/modelo) no entran en la L2 de 512 KB, y (c) exp/tanh de libm son escalares (~37 % del tiempo).

| variante (3×128) | µs/fila | vs v1 | nota |
|---|---|---|---|
| v1: filas float32 + libm | 85.2 | 1.00 | kernel de S3/S4 |
| 8 filas por bloque | 91.7 | 0.93 | peor |
| traspuesto (K,3H), bloques de 4 k | 85.4 | 1.00 | solo, nada |
| int16 por fila | 83.0 | 1.03 | solo, nada |
| tanh racional (Eigen 13/6) en vez de exp/tanh | 79.2 | 1.08 | |
| traspuesto + int16 + tanh racional | 64.8 | 1.31 | |
| **traspuesto + int8 + tanh racional** | **54.0** | **1.58** | **elegido** |
| traspuesto + int8 + tanh racional, bloques de 8 k | 59.2 | 1.44 | peor que bloques de 4 |

Precisión: int8 (escala simétrica por fila de salida) **no cambia la métrica**: 0.6831 / 0.6831 / 0.6831 (float32 / int16 / int8) en 120 seqs de valid
(`model/submission/kernel_bench/quant_eval.py`, torch). Con el pipeline oficial (20 seqs): float32 0.7334, int8 0.7335; paridad vs PyTorch 1.8e-4 (float32 v2) / 9.5e-2 (int8).
**Medición limpia con el callback oficial (2026-09-18 01:00, núcleo fijo, datos reales, anclada a S4 = 33 min reales en el servidor):**
S4 v1 2×128 = 70.8 µs (hoy; ayer 57: la laptop quedó más lenta) → ens3 v1 = 112.7 (52 min servidor), **ens3 v2 int8 = 63.6 (~30 min)**, **B+Cswa+Dswa v2 int8 = 83.8 (~39 min)**.
Proyección: 4×128 ≈ 40 min, 5×128 ≈ 50 min. Overhead DataPoint + loop: 1.4 µs; llamada numba: 1.3 µs (7 args). Fórmula nueva de `eval_ensemble.py`: µs ≈ 2 + 0.091·kMAC → 2×128 = 37, **3×128 = 54 (39 min servidor)**,
4×128 = 71 (51 min), 2×128+192 = 73 (52 min). `build_submission.py --bits 8|16|32`.

## Experimentos del equipo (2026-09-13, `model/`) — GRU 1 capa 32 unidades, 100 seqs train, 2 épocas, chunk 128, MSE cruda, CPU

Calibración: baseline oficial evaluado en los mismos grupos de valid de cada experimento (2026-09-15).

| Experimento | Valid (50 g) | Baseline en esos g | GRU base | Variante | Δ variante − base (5 rondas) | Veredicto |
|---|---|---|---|---|---|---|
| `feature_select` top_k lags/deltas (1 corrida) | seed 2026 | **0.560** | 0.441 | top_5 0.448 / top_10 0.434 / top_20 0.367 / top_3 0.400 | +0.006 / −0.008 / −0.075 / −0.042 | muchas entradas empeoran |
| `feature_select` top3 + `i0_v21_delta_5` (5 rondas) | seed 2027 | **0.637** | 0.533–0.574 | 0.518–0.559 | media −0.008, mejora 1/5 | no ayuda a esta escala |
| `a_boundary_flags` `aN_is_min/max` (5 rondas) | seed 2027 | **0.637** | ídem | 0.483–0.567 | media −0.011, mejora 1/5 (ronda 5: −0.049) | no ayuda a esta escala |
| `model_target` feedback de predicciones (5 rondas) | propio | **0.552** | 0.424–0.473 | 0.428–0.475 | media +0.001, mediana +0.010, mejora 3/5 | empate |

Lectura: la GRU-32 del equipo queda ~0.10 por debajo del baseline en todos los conjuntos; nuestra LSTM-64 (exp 006) queda −0.07. Nadie supera el baseline aún. Las features no dieron ganancia robusta con red chica + pocos datos + MSE cruda; quedan para re-probar a escala. Regla desde ahora: **reportar siempre la distancia al baseline en los mismos grupos**.

Ranking de correlación (`explore/feature_select`, 100 seqs): top para t0 = `i0_p3_lag_1` (0.13), `i0_p8_lag_1`, `i0_p0_lag_1`, `i0_v21_delta_5`, `i0_p3_delta_50` (~0.08). Los `lag_1` son casi redundantes con la columna cruda (autocorr alta); los `delta_50` de precios sí son información nueva.

## Experimentos con la métrica GLOBAL (desde 2026-09-16)

| id | fecha | modelo / cambio | train | GLOBAL WP (10 g) | t0 | t1 | por-seq (vieja) | máquina | notas |
|---|---|---|---|---|---|---|---|---|---|
| ref-G | 2026-09-16 | baseline.onnx oficial | 10.607 | **0.6498** | 0.632 | 0.668 | 0.568 | laptop | referencia nueva |
| 010 | 2026-09-16 | LightGBM deltas (60 seqs×300 filas, 150 árboles) — prueba | 60 | 0.485 | 0.533 | 0.438 | 0.420 | laptop | `model/lgbm_deltas/train_lgbm_deltas_v1.py` validado |
| 011 | 2026-09-16 | mezcla 0.3·LGBM(010) + 0.7·GRU(exp 007, CPU 100 seqs) | — | 0.581 | 0.603 | 0.559 | 0.519 | laptop | GRU sola: 0.558 global → la mezcla suma +0.023 |
| 009-G | 2026-09-16 | **GRU exp 009 re-medida con métrica global** | 10.607 | 0.6474 | 0.679 | 0.616 | 0.584 | Kaggle | **−0.002 vs baseline** (con la vieja era +0.016). t0 +0.05 mejor que baseline, t1 −0.05 peor |
| 012 | 2026-09-16 | **LightGBM deltas, todo train** (10.607×300 filas = 3,2M, 560 feats, 600 árboles/target) — `lgbm_deltas_all` | 10.607 | 0.6253 | 0.649 | 0.602 | 0.549 | Kaggle CPU | 70 min. Top feats: i0_p3, i0_p15, i0_p0, i0_p21, i0_p14, deltas de i0_v4; pocas deltas en el top 25 |
| **013** | 2026-09-16 | **mezcla 0.3·LGBM(012) + 0.7·GRU(009)** | — | **0.6528** | 0.682 | 0.623 | 0.587 | Kaggle CPU | **+0.003 vs baseline, +0.005 vs GRU sola**; 0.2→0.6519, 0.4→0.6526, 0.5→0.6513. Confirmar en ≥200 grupos |

### Evaluación en TODO valid (métrica global) — 2026-09-16/17

| modelo | seqs | GLOBAL WP | t0 | t1 | fuente |
|---|---|---|---|---|---|
| **baseline oficial** | **1.873** | **0.6171** | 0.588 | 0.646 | `eval_full_valid_baseline.log` (172 min laptop). Métrica vieja: 0.5896 = README oficial ✓ |
| baseline oficial | 700 (primeras) | 0.6213 | — | — | `eval_full_valid_700seqs.log` (laptop, se cortó a las 700) |
| GRU exp 009 (`gru_only.zip`) | 700 (primeras) | **0.6748** | — | — | ídem → **+0.053 vs baseline** (en los 10 grupos parecía −0.002: los 10 grupos engañan) |
| notebook "Challenge" (compañero) `full/base`: GRU 128x2, 112 entradas, misma receta, 10 ép. | 1.873 | **0.6722** (ép. 6) | 0.666 | 0.679 | log del notebook; curva 0.672/0.672/0.672/0.666/0.664 en ép. 6–10 |
| notebook "Challenge" `full/events` (2.880 feats) | 1.873 | 0.6673 (ép. 5, en curso) | 0.661 | 0.674 | −0.005 vs base |
| notebook "Challenge" piloto (1.000 seqs, 3 ép.): base / temporal / events / complete | 1.873 | 0.6547 / 0.6522 / 0.6530 / 0.6497 | | | **ninguna feature pack supera a base** (3er experimento que lo confirma) |

Distancia al baseline en todo valid: **+0.056** (full/base de Santiago) / +0.053 (exp 009 en 700 seqs). **LB real: 0.6426 (puesto 123)** → el delta en valid se traslada ~0.9× al test. Regla: +0.01 en valid ≈ +0.009 en LB. Serie completa valid → LB: S3 0.6779→0.6471, S4 0.6796→0.6495, S6 0.6825→0.6524, S7 0.6843→0.6542: **LB ≈ valid − 0.030, deltas 1:1**. Para el top 20 (0.657) hacen falta ~+0.016 más en valid (≈ 0.689).

## Submissions

Leaderboard público (2026-09-16): baseline = **0.5935** (puestos 146–182, gente que subió el zip original). Top 1 = 0.6622 (+0.069). Puesto 100 ≈ 0.642 (+0.049).
Nuestro valid de 10 grupos es más fácil que el test (baseline 0.6498 ahí): comparar siempre **distancias al baseline**, no valores absolutos.

| # | fecha | zip | contenido | local (10 g, global) | µs/fila laptop | LB público | notas |
|---|---|---|---|---|---|---|---|
| S1 | 2026-09-16 | `submissions/gru_only.zip` | GRU 128x2 exp 009 en numba (`model/submission/solution.py`) | 0.6474 (baseline 0.6498) | 37.5 (50 con scorer) | *pendiente* | paridad 1e-4 vs PyTorch; determinista; compila 1.7 s |
| S2 | — | `submissions/gru_lgbm300.zip` | 0.3·LGBM(300 árboles) + 0.7·GRU | 0.6508 | 99 → **no entra** (62 min) | — | hay que bajar el tiempo de los árboles dentro del kernel unificado |
| **S-Santiago** | 2026-09-17 | `Archivos subidos/trainposta.zip` (notebook `trainposta`) | GRU 128x2 `full/base` de temporal_experiment.py (misma receta que exp 009), ép. 6 | **0.6731 en 1.873 seqs** | — | **0.6426 → puesto 123** | Calibración valid→LB: 0.6731 − 0.6171 (baseline valid) = +0.056; LB 0.6426 − 0.5935 = +0.049. **Se traslada ~0.9×.** `full/events` (2.880 feats, 10 ép.) 0.6716: −0.0015, y no entra en tiempo (170 µs/fila). Loss train de events estancada desde ép. 4 → 20 épocas no la salva. |

| 015 | 2026-09-17 | **GRU 128x2 + dropout 0.1 + wd 1e-5**, seed 2028, 8 ép., SWA 5–8, valid all (`gru128_drop_seed2028`) | 10.607 | 0.6743 (ép. 7); **SWA 0.6762** | 0.671 | 0.681 | 6.5 min/ép. | Kaggle T4 | mejor 128x2 hasta ahora; dropout + SWA suman ~+0.003 |
| — | 2026-09-17 | 192x2 SWA 5–8 | — | **0.6765** | 0.672 | 0.681 | | | SWA +0.003 sobre mejor época |
| **ENS** | 2026-09-17 | Tabla completa en `model/submission/results/ensemble_eval_sprint1.json` (1.873 seqs). Mejores: B+C+Dswa **0.6817**, A+B+C 0.6806, B+Cswa 0.6796, A+B 0.6779 | | | | | | | **µs/fila MEDIDOS (laptop, kernel numba mixto)**: 2×128 = 57, 3×128 = 90 (56 min, al límite), 128+128+192 = 122 (no entra). Solo entran ensambles de 2. |
| **S3** | 2026-09-17 | `submissions/ens2_exp009_santiago.zip` (A+B) | promedio GRU exp009 + GRU Santiago full/base (2×128x2, `solution.py` multi-GRU, clip ±20 en el de Santiago) | 40 seqs: 0.7154 (A 0.7099, B 0.7101); 256 seqs (informe): +0.004 | 45 → 28 min | **0.6471 → puesto 117** | +0.0045 LB vs modelo solo; valid predijo +0.004 → **calibración confirmada** |
| 014 | 2026-09-17 | **GRU 192x2** seed 2027, 8 ép., lr-decay 0.9, SWA 5–8, valid all (`gru192_seed2027`, notebook sprint) | 10.607 | curva 0.6627/0.6687/0.6603/0.6677/0.6688/0.6706/**0.6735**/0.6724 | 0.669 | 0.678 | 18 min/ép. | Kaggle T4 | solo ≈ Santiago (0.6731); techo en ép. 7. Para ensamble. SWA pendiente |
| **S4** | 2026-09-17 | `submissions/ens2_B_Cswa.zip` | Santiago + gru128_drop_seed2028 SWA (2×128) | **0.6796** (1.873) | 57 → 33 min (servidor) | **0.6495** | +0.0017 valid vs S3 → +0.0024 LB (estimado 0.6486); paridad 1.3e-5 |
| **016** | 2026-09-18 | **Destilación** (`distill_128`, sprint 2): GRU 128x2 dropout 0.1 wd 1e-5, seed 2029, 8 ép., SWA 5–8, loss = WMSE(y) + 1.0·WMSE(soft) con soft = promedio de B+Cswa+Dswa (0.6817), pesos \|y\| | 10.607 | curva 0.6709/0.6749/0.6756/0.6763/0.6782/0.6782/0.6784/**0.6790**; **SWA 0.6805** | 0.676 | 0.685 | 18 min/ép. (teachers en cada chunk) | Kaggle T4 | **Mejor modelo individual (+0.004 sobre Cswa)**. Curva más suave y por encima de C en todas las épocas. Un 128 solo = lo que valía un ensamble de 2. Ensambles: Dswa+Eswa 0.6818, Eswa+Gswa 0.6814 (`results/sprint2/results/ensemble_eval_sprint2.json`, solo pares) |
| 017 | 2026-09-18 | Schedule **coseno** 1e-3→1e-5, 10 ép., dropout, SWA 7–10, seed 2030 (`cosine_128`) | 10.607 | 0.6747 (ép. 6); SWA 0.6717 | 0.670 | 0.680 | 11 min/ép. | Kaggle T4 | no aporta: empeora desde ép. 7 (0.6733/0.6708/0.6698/0.6688) → el SWA de las últimas promedia modelos peores. Descartado |
| 018 | 2026-09-18 | Loss **WMSE \|y\|^1.5 + 0.05·(1−WP global)**, dropout, SWA 5–8, seed 2031 (`gwp_128`) | 10.607 | 0.6737 (ép. 8); SWA 0.6761 | 0.671 | 0.681 | 11 min/ép. | Kaggle T4 | igual que C (0.6762): la loss alineada ya era suficiente. Descartado |
| **S5** | 2026-09-18 | `submissions/ens3_B_Cswa_Dswa_int8.zip` | Santiago + Cswa + Dswa(192) (0.6817 en 1.873), **kernel v2 int8** | 0.7308 (10 seqs) | 83.8 → ~39 min servidor | ~0.651 (top 70) | primera submission con el kernel v2: calibra tiempo y score del int8 en el servidor |
| **ENS-2** | 2026-09-18 | Tabla de 162 combinaciones (1–4 modelos) de A,B,Cswa,Dswa,E,Eswa,F,Gswa en 1.873 seqs (`model/submission/results/eval_combos/`) | | Dswa+Eswa+F+Gswa **0.6829**; Dswa+Eswa+Gswa **0.6825**; Dswa+Eswa 0.6818; Eswa 0.6805 | | | | Kaggle T4 | techo ~0.683 con los modelos actuales: el 4.º modelo suma +0.0004 |
| **S6** | 2026-09-19 | `submissions/ens3_Dswa_Eswa_Gswa_int8.zip` | 192 SWA + destilada SWA + gwp SWA (**0.6825** en 1.873), kernel v2 int8 | 0.7272 (10 seqs) | 52 → ~40 min servidor | **0.6524 → puesto 65** | +0.0029 valid sobre S4 → +0.0029 LB: calibración 1:1. Kernel int8 confirmado en el servidor |
| **020** | 2026-09-19 | **Sprint 4, 6 maestros diversos** (idea de Joaquín), receta común 8 ép. / drop 0.1 / wd 1e-5 / SWA 5–8, semillas propias. x1 GRU 128x2, x2 GRU 200x1, x3 GRU 64x3, y1 GRU 160x2, **y2 LSTM 128x2**, y3 GRU 128x2 d0.2 (`results/sprint4_{x,y}/`) | 10.607 | x1 0.6741/**0.6768** · x2 0.6711/0.6730 · x3 0.6748/0.6757 · y1 0.6718/0.6747 · **y2 0.6788/0.6803** · y3 0.6743/0.6760 (mejor/SWA) | | | 1,2–2 h c/u | Kaggle T4 ×2 en paralelo | **La LSTM 128x2 sola da 0.6803 = Eswa sin destilar, +0.004 sobre cualquier GRU normal.** 1 capa×200 y 3×64 rinden menos que 128x2. 160x2 no mejora a 128x2 |
| **021** | 2026-09-19 | **Alumnos X (x1-3), Y (y1-3), Z (los 6 + Eswa + Dswa)**, GRU 128x2, soft targets precalculados (trainer v4 corregido), semillas 2051–2053 (`results/sprint4_distill/`) | 10.607 | X 0.6797/**0.6802** · Y 0.6782/0.6798 · Z 0.6786/**0.6802** | | | 1,6 h c/u (+6–18 min precálculo) | Kaggle T4 | ningún alumno supera el techo individual (~0.680). Tabla de 1.940 combinaciones (15 modelos, ≤4): **y2+Eswa+Dswa+X 0.6843, y2+Eswa+X+Zswa 0.6843, y2+Eswa+X 0.6841**; sin y2 la mejor cuaterna es 0.6836. **La LSTM está en las 40 mejores combinaciones** |
| **S7** | 2026-09-19 | `submissions/ens4_y2_Eswa_X_Zswa_int8.zip` | LSTM y2 + Eswa + X + Zswa (**0.6843**), **kernel v3** (GRU+LSTM, int8) | 0.7352 (10 seqs) | 55.2 → 38 min estimado, **35m05s real** (690 MB) | **0.6542** | +0.0018 valid vs S6 → +0.0018 LB: calibración 1:1 otra vez. Primera submission con LSTM en el servidor: kernel v3 OK |
| **022** | 2026-09-19 | **Alumna LSTM LA** (128x2, seed 2061) destilada de y2+Eswa+Dswa+X+Zswa, soft targets precalculados (`results/sprint5/`) | 10.607 | curva 0.6625/0.6765/0.6773/0.6794/0.6789/0.6789/0.6815/**0.6823**; SWA 0.6821 | 0.678 | 0.686 | 1,7 h (+14 min precálculo) | Kaggle T4 | **Mejor modelo individual (+0.002 sobre Eswa/y2)**; sigue subiendo en la ép. 8 (las GRU se plantan en la 6–7) → retomar hasta 12. Tabla de 255 combos: **y2+Eswa+X+LA 0.6848**, y2+Eswa+LA 0.6846 (3 modelos). La alumna LB **murió por bug mío** (recarga del cache releía el npz de 849 MB por secuencia → Killed por RAM; corregido) |
| 023 | 2026-09-19/20 | **LSTM base l1, l2, l3** sin maestros (`sprint4_lstm`, `results/sprint4_lstm/`): l1 128x2 seed 2047, l2 128x2 seed 2048, l3 **160x2** seed 2049 | 10.607 | l1 0.6794/**0.6808** · l2 0.6778/0.6794 · l3 0.6794/0.6809 (mejor/SWA) | | | 1,5–2 h c/u | Kaggle T4 | **LSTM cruda ≈ 0.680 de forma consistente** (y2 0.6803, l1 0.6808, l3 0.6809; l2 algo peor por la semilla). 160x2 no mejora a 128x2 (igual que en GRU). Sirven como miembros de ensamble, no como mejor modelo |
| 024 | 2026-09-20/21 | **Sprint 6 — trozos (chunk) de 2.000 pasos** en vez de 512, alumna LSTM 128x2 destilada de y2+Eswa+Dswa+X+Zswa, seed 2071, 10 ép., SWA 7–10 (`results/sprint6/student_chunk2000`) | 10.607 | 0.6820 (ép. 9) / SWA **0.6824** | 0.678 | 0.686 | 2,5 h | Kaggle T4 | = LA (0.6823, chunk 512): la memoria más larga en el entrenamiento **no aporta** (ni en t1). Descartado; queda chunk 512 |
| **025** | 2026-09-20/21 | **Sprint 6 — lr 5e-4** en vez de 1e-3, misma alumna LSTM, seed 2072, 10 ép., SWA 7–10 (`results/sprint6/student_lr5e-4`) | 10.607 | curva 0.6728/0.6772/0.6801/0.6804/0.6819/0.6832/**0.6834**/0.6818/0.6824/0.6819; SWA **0.6834** | 0.679 | 0.687 | 2,5 h | Kaggle T4 | **Mejor modelo individual (+0.0011 sobre LA)**. Techo en la ép. 7 y luego oscila (ruido entre semillas ≈ ±0.001, así que es una mejora chica pero coherente: las 4 últimas épocas están todas ≥ 0.6818). Receta nueva por defecto para alumnas LSTM: **lr 5e-4, 8 ép., SWA 5–8** |
| **ENS-3** | 2026-09-21 | **Sprint 5b lite: tabla de 1.092 combinaciones (≤4) de 13 piezas** (y2, Eswa, Dswa, X, Zswa, l1/l2/l3 SWA, LR SWA, C2k SWA, LA8, LA, LAswa) en 1.873 seqs (`results/sprint5b/results/ensemble_eval_sprint5b.json`) | | **Eswa+l3swa+LRswa+LA8 0.6855**, Eswa+l1swa+LRswa+LA8 0.6855 (Δ 0.00006), l3swa+LRswa+LA8 0.6853 (3 modelos), LRswa+LA8 0.6847 (2) | | | 22 min | Kaggle T4 | **Las 3 LSTM destiladas + 1 GRU forman el mejor ensamble.** Dswa (192) ya no entra en tiempo. El techo subió de 0.6848 a 0.6855. El resume de LA a 12 épocas **falló** (bug: tomaba la ruta local de datos, corregido y probado) → LA = LA8 en la tabla |
| **S8** | 2026-09-21 | `submissions/ens4_Eswa_l1swa_LRswa_LA8_int8.zip` | Eswa + l1 SWA + LR SWA + LA8 (**0.6855** en 1.873; GRU + 3 LSTM), kernel v3 int8 | 0.7337 (10 seqs; float32 0.7336) | 64.9 → 39m46s real (691 MB) | **0.6541** | valid decía +0.0012 y el LB dio −0.0001: **primera vez que un delta no se traslada**. Causa probable: elegir la mejor de 1.092 combinaciones sobre valid (y alumnos cuyos maestros y épocas también se eligieron con valid) infla el valid ~0.001. Regla nueva: **una diferencia < 0.002 en valid no justifica submission**. S7 sigue siendo la mejor. Elegida la versión con l1 (128) en vez de l3 (160): misma métrica, 4 min menos |
| **026** | 2026-09-26/27 | **Sprint 8 — alumnas GRANDES** (la hipótesis: el cuello es el tamaño de la alumna, no los maestros). Mismos 4 maestros que la S8 (0.6855), lr 5e-4, dropout 0.1, wd 1e-5. **big192**: LSTM 192x2, 9 ép., SWA 6–9, seed 2081 (4,8 h). **big256**: LSTM 256x2, 7 ép., SWA 5–7, seed 2082 (2,3 h) | 10.607 | big192 0.6826 / SWA **0.6834** · big256 0.6838 / SWA **0.6843** (curva 0.6742/0.6796/0.6819/0.6826/0.6822/0.6834/0.6838, **seguía subiendo**) | | | | Kaggle T4 ×2 en paralelo | **Hipótesis refutada.** 192x2 = 0.6834 = exactamente lo mismo que la alumna 128x2; 256x2 gana +0.0009, dentro del ruido entre semillas. Mejor ensamble que entra: **b256+LR 0.6854** y b192+LR+LA 0.6854 = igual que el 0.6855 que ya teníamos. El tamaño no es el cuello. Dato curioso: el 256x2 entrenó **más rápido por época** que el 192x2 (19 vs 32 min): hidden potencia de dos usa kernels de cuDNN más rápidos |
| **027** | 2026-09-27 | **Sprint 9 — features cruzadas**: 20 entradas NO lineales dentro de la fila (desequilibrios como cociente, desvíos y máximos entre los 11 niveles, concentración del volumen, i0 vs i1, productos). LSTM 128x2, lr 5e-4, 8 ép., SWA 5–8, seed 2091, mismos 4 maestros. 132 entradas (`results/sprint9_connectome-sprint9-cross/`) | 10.607 | 0.6824 / SWA **0.6831** (curva 0.6733/0.6771/0.6751/0.6812/0.6812/0.6816/0.6817/0.6824) | 0.680 | 0.687 | 2,5 h | Kaggle T4 | **Sin ganancia: la misma receta sin las features da 0.6834.** Tampoco aporta al ensamble (cross+b256+LR 0.6856 vs b256+LR+LA 0.6857). **Cuarto intento fallido de features**: ni lags/deltas en el tiempo, ni combinaciones lineales, ni combinaciones no lineales dentro de la fila. Las 112 entradas crudas ya contienen lo que el modelo puede usar |
| **028** | 2026-09-29/30 | **Sprint 10 — loss de correlación a nivel de secuencia** (`--loss seqwp`, estadísticos acumulados por secuencia con olvido 0.98 + `--wmse-anchor`). LSTM 128x2, lr 5e-4, 8 ép., SWA 5–8. **seqwp** (sin destilar, ancla 0.1, seed 2101) y **seqwp_dist** (destilada, ancla 0.2, seed 2102 = misma receta que la alumna LR salvo la loss) | 10.607 | seqwp 0.6790 / SWA **0.6804** · seqwp_dist 0.6836 / SWA **0.6837** | 0.680 | 0.688 | 1,3 h c/u | Kaggle T4 ×2 | **Sin ganancia, con la comparación más limpia que hicimos:** seqwp_dist vs LR (idénticas salvo la loss) = **0.6837 vs 0.6834**, +0.0003 = ruido. Sin destilar: 0.6804 vs 0.6808 de l1. Tampoco aporta al ensamble (sq+b256+LR 0.6855 vs b256+LR+LA 0.6857). **La hipótesis de la loss queda descartada**: la ventaja de ese competidor viene de otro lado |
| **029** | 2026-09-30/10-01 | **Sprint 11 — grilla 2x2: memoria larga (EMA) x duración del entrenamiento.** `--ema-features` (40 entradas: desvío respecto de promedios móviles de 100/1.000/10.000 filas de 12 columnas + volatilidad lenta; 152 entradas en total) y 20 épocas con SWA 15–20. Alumnas LSTM 128x2 lr 5e-4 destiladas, seeds 2111/2112 | 10.607 | **8 ép. sin EMA 0.6834** (ref) · 8 ép. con EMA **0.6821** · 20 ép. sin EMA **0.6838** · 20 ép. con EMA **0.6838** | 0.680 | 0.688 | 3 h + 2,5 h c/u | Kaggle T4 ×2 | **Las dos ideas, descartadas.** Nada se mueve más de ±0.0004, salvo EMA a 8 épocas que es peor. Las curvas hacen pico en la **época 9–10** y después bajan: el SWA de las épocas 15–20 (0.6808 / 0.6820) es **peor** que el de 5–8. **Conclusión operativa: 9–10 épocas es el óptimo; 20 es tiempo tirado.** El mejor ensamble subió a 0.6860 (ema+long+LR, 33 min) pero es elegir entre ~100 combinaciones: por la regla de ≥0.004 no se sube |
| **030** | 2026-10-06 | **Sprint 13 — mitades DISJUNTAS de los datos.** Dos alumnas LSTM 128x2 lr 5e-4, 10 ép., SWA 7–10, entrenadas sobre 5.303 secuencias cada una **sin una sola en común** (seeds 2131/2132) | 5.303 c/u | mitad1 0.6806/**0.6806** · mitad2 0.6809/**0.6815** · promedio de las dos **0.6832** | | | 1,5 h c/u | Kaggle T4 ×2 | **El resultado más importante del proyecto.** Con la mitad de los datos pierden solo 0.002, y sobre todo: **sus predicciones correlacionan 0.984 entre sí**, igual que todo el resto de la familia. Dos modelos que no compartieron ni una secuencia de entrenamiento aprenden **la misma función**. No es falta de diversidad por receta: es que con estas 112 entradas y esta clase de modelo, la función está esencialmente determinada |
| **S9** | 2026-10-06 | `submissions/ens3_long_LR_LA_int8.zip` (long + LR + LA, **0.6857** en valid) | | 0.7091 (6 seqs) | 43,3 → 31m52s real (686 MB) | **0.6533** | **Valid dejó de transferir.** Serie: S7 valid 0.6843 → LB 0.6542 (brecha 0.0301) · S8 0.6855 → 0.6541 (0.0314) · S9 0.6857 → **0.6533** (0.0324). La brecha **crece** justo lo que ganamos en valid: es sesgo de selección acumulado (época elegida por valid, hiperparámetros elegidos por valid, combinación elegida por valid). **Valid ya no puede rankear modelos a resolución de 0.002** |
| **S10** | 2026-10-09 | `submissions/ens4_mask_LR_LA_long_int8.zip` (mask + LR + LA + long, **0.6863** en valid) | | — | 41,0 estimado → **41m52s real** (690 MB) | **0.6540** | **Cuarta submission seguida con el leaderboard plano.** Valid subió 0.0020 desde S7 y el LB se movió −0.0002. Serie de brechas: S7 0.0301 · S8 0.0314 · S9 0.0324 · S10 **0.0323**. Todo lo ganado en valid desde el 19 de septiembre es sesgo de selección. **Regla nueva: no gastar submissions en variantes elegidas por valid; solo en modelos que sean distintos por una razón estructural.** El tiempo estimado acertó a 52 s: el modelo de costo sirve para planificar los árboles |
| **031** | 2026-10-07 | **Sprint 14 — OTRA CLASE DE MODELO: convolucional causal dilatada** (`--arch tcn`, 128 canales × 5 capas, dilataciones 1/4/16/64/256, campo receptivo 683 filas, 260k operaciones por fila ≈ las 254k de la LSTM 128x2). **Sin destilar a propósito.** Dos pasos de aprendizaje: 5e-4 (seed 2141) y 1.5e-3 (seed 2142) | 10.607 | lr 5e-4 0.6722 / SWA **0.6739** · lr 1.5e-3 0.6683 / SWA **0.6734** | 0.670 | 0.678 | 2 h c/u | Kaggle T4 ×2 | **No alcanza, pero es lo más distinto que logramos:** su correlación con la familia recurrente es **0.962–0.972**, la más baja medida (entre LSTM es 0.976–0.988). Aun así no aporta al ensamble: con el peso óptimo elegido en la mitad A da **−0.0001** en la mitad B. **Dos salvedades:** la curva **seguía subiendo** en la época 10 (0.6684 → 0.6722) y corrió **sin destilación**, que vale +0.003. Por eso se repite bien entrenada (sprint 15B) antes de cerrarla |
| **032** | 2026-10-07/08 | **Sprint 15A — pesar el entrenamiento hacia las filas que la métrica puntúa** (`--mask-coef`, AUC 0.794, potencia 1.0). LSTM 128x2 destilada, lr 5e-4, 10 ép., SWA 7–10, seed 2151 | 10.607 | 0.6840 / SWA **0.6840** | 0.680 | 0.688 | 2 h | Kaggle T4 | **+0.0006 sobre la alumna LR (0.6834): dentro del ruido entre semillas.** Alinear el entrenamiento con lo que la métrica mira no cambia nada medible. Su correlación con la familia es 0.974–0.986, igual que todo |
| **033** | 2026-10-07/08 | **Sprint 15B — la convolucional bien entrenada**: 20 épocas + destilación (seed 2152) | 10.607 | 0.6782 / SWA **0.6782** (curva **todavía subiendo** en la ép. 20: 0.6776/0.6778/0.6782) | 0.674 | 0.682 | 4 h | Kaggle T4 | **Sube +0.004 respecto de la versión sin destilar, pero su correlación con la familia sube de 0.962 a 0.975.** La destilación la empuja a imitar a las LSTM: gana score y pierde lo único que la hacía valiosa. Ensamble: **+0.0000** honesto (elegido en una mitad de valid, medido en la otra). **Lección: destilar y ser distinto están en conflicto** |
| **034** | 2026-10-08/09 | **Sprint 16 — el cuadrante que faltaba: convolucional CONVERGIDA y SIN destilar.** 40 épocas, SWA 33–40. Dos tamaños: 128 canales (seed 2161) y 192 canales (seed 2162) | 10.607 | 128 ch: mejor 0.6738 / SWA 0.6724 · 192 ch: mejor 0.6737 / SWA 0.6694 | 0.670 | 0.678 | 5 h y 7 h | Kaggle T4 ×2 | **Rama de arquitecturas CERRADA con evidencia.** Las dos saturan: la de 128 hace pico en la época 21 y las últimas 5 épocas dan −0.0007; la de 192 decae −0.0017 (más tamaño no ayuda, igual que en la LSTM). Correlación con la familia **0.957–0.969**, la más baja de todo el proyecto, y **aun así el aporte honesto al ensamble es +0.0000** (combinación elegida en la mitad A de valid, medida en la mitad B: `LR+LA+long+b256` da 0.6862 con o sin las convolucionales). Queda medido el techo de la arquitectura: **~0.674 sin destilar, 0.678 destilada, contra 0.683–0.684 de la LSTM** |
| **035** | 2026-10-15 | **Sprint 18 (CPU) — los árboles con 3,3x más datos.** LightGBM: 10.607 × 1.000 = **10,6M filas**, **190 columnas** (las 150 de mayor ganancia + las 40 de memoria larga), 600 árboles/target, 200 secuencias de valid | 10.607 | **0.6390** (t0 0.6384, t1 0.6395) | — | — | 71 min | Kaggle CPU | **El árbol mejoró de verdad: +0.0137 sobre el 0.6253 de septiembre.** Pero su correlación con el ensamble **subió de 0.918 a 0.9271**, porque las features de memoria larga lo acercan a lo que hacen las redes recurrentes. Umbral 0.6755 × 0.9271 = 0.6263: lo pasa por +0.0127, pero la **ganancia honesta medida es −0.0014** (peso elegido en la mitad A de valid, medido en la mitad B; el óptimo real es ~0.05 y vale +0.0001). Las 40 features de memoria larga se llevaron solo el **10,2 %** de la ganancia. **Tercera vez que aparece el mismo patrón: mejorar un modelo lo vuelve más parecido al resto** |
| **037** | 2026-10-16 | **Sprint 17A — las columnas centinela** (`--sentinel-flags`): a5/a6/a7 con −5,1993 → 0 más 3 columnas indicadoras, 115 entradas. Receta idéntica a la alumna LR, seed 2171 | 10.607 | mejor 0.6836 / SWA **0.6841** | — | — | 136 min | Kaggle T4 | **Ruido: +0.0007 sobre el 0.6834 de LR**, con el criterio puesto en 0.6855. La distorsión era real y está medida (el desvío de a7 inflado 5x, y a7 limpia correlaciona 0.21 con t1 contra 0.04 cruda), pero **la red aprende sola a ignorar el centinela**. Confirma lo que ya habían dicho las dos pruebas locales (lineal −0.0007, perceptrón −0.0031): esta vez el diagnóstico barato acertó |
| **038** | 2026-10-16 | **Sprint 17B — aprender suma y diferencia** (`--target-cd`): la red predice (t0+t1)/2 y (t0−t1)/2, y la cabeza se transforma de vuelta al guardar. Receta idéntica a LR, seed 2172 | 10.607 | mejor 0.6800 / SWA **0.6803** | — | — | 106 min | Kaggle T4 | **PEOR: −0.0031.** Era el riesgo anotado en el propio notebook: la métrica puntúa t0 y t1 por separado, y optimizar la suma y la diferencia con sus propios pesos optimiza algo parecido pero no igual. Queda cerrado |
| **036** | 2026-10-16/17 | **Sprint 19 — memoria larga elegida midiendo** (`--ema-plus`): 11 columnas nuevas con promedios móviles, elegidas por su correlación con el ERROR del ensamble (i1_v20, i0_p9, i0_p15, i0_p2, i0_v4, i1_v4, i0_p20, i0_v12, i0_v6, i0_v9, i1_v9). 73 features de memoria larga en vez de 40, 185 entradas en vez de 152. LSTM 128x2 destilada, 20 épocas, seed 2191 | 10.607 | mejor **0.6834** / SWA 0.6776 (pico en la época 8, después baja) | — | — | ~4 h | Kaggle T4 | **−0.0004 contra el 0.6838 de `student_ema` y `student_long`: no aporta.** La curva hace pico en la época 8 (0.6834) y decae hasta 0.6768: con 33 entradas más, la red sobreajusta antes. **Segunda vez que la memoria larga da cero, ahora con las columnas elegidas midiendo en vez de a ojo.** La rama de features de entrada queda cerrada con evidencia |
| **037** | 2026-10-18 | **Sprint 20 — XGBoost** (`--lib xgboost`, por niveles, max_depth 6 ≈ 64 hojas). Todo lo demás idéntico a LightGBM | 10.607 | **0.6405** (t0 0.6398, t1 0.6412) | — | — | ~1,5 h | Kaggle CPU | **+0.0015 sobre LightGBM: empate.** Correlación con el ensamble **0.9279** contra 0.9271 de LightGBM: igual. **Y entre ellos correlacionan 0.9934**, o sea que son el mismo modelo con otra implementación. Aporte honesto **−0.0015**; promediando los dos árboles, −0.0015. **La rama de árboles queda cerrada con medición, no con argumento.** La 1.ª corrida murió por memoria: `xgb.DMatrix` guarda 8 bytes por número aunque la matriz sea densa (16 GB); arreglado con `xgb.QuantileDMatrix` (1 byte, ~2 GB) |
| **038** | 2026-10-19/20 | **Sprint 21 — el modelo final: train + valid.** 12.280 secuencias (+17,7 % de datos) dejando afuera las 200 de evaluación. Receta idéntica a la alumna LR, épocas fijas y SWA, sin elegir nada por valid. Semillas 2211 y 2212 | 12.280 | final_a SWA **0.6718** · final_b SWA **0.6746** (medidos en las 200 secuencias) | — | — | ~3 h c/u | Kaggle T4 | **+0.0002 contra los modelos viejos medidos en las mismas 200 secuencias** (promedio viejos 0.6730, promedio nuevos 0.6732). **Pero la prueba no tiene potencia para resolverlo:** la distancia entre las dos semillas es **0.0028**, más grande que el efecto que buscábamos (+0.002). Correlación entre las dos nuevas 0.9903, y de las nuevas con el ensamble viejo **0.9951**: son los mismos modelos. Aporte honesto al ensamble **+0.0000**. Se agregan 2 semillas más (2213, 2214) para poder armar un ensamble de 4 comparable al viejo y **medirlo en el leaderboard**, que es el único juez que queda |
| **039** | 2026-10-20 | **Sprint 22 — el horizonte de aprendizaje** (`--chunk 1024` en vez de 512) | 10.607 | mejor 0.6834 / SWA **0.6836** | — | — | 128 min | Kaggle T4 | **+0.0002: ruido.** Duplicar hasta dónde puede aprender la red no cambia nada, igual que las features de memoria larga. La corrida de `--chunk 2048` murió a las 12 h por el bug de rutas de abajo, sin llegar a entrenar |
| **040** | 2026-10-20 | **Sprint 23 — normalizar las entradas DENTRO de cada secuencia** (`--adapt-span` 2000 y 8000). Media y desvío móviles causales por secuencia, inicializados en los valores globales. Receta idéntica a la alumna LR, semillas 2231 y 2232 | 10.607 | *pendiente* | — | — | ~5 h c/u | Kaggle T4 | **(RESUELTO 2026-10-25, ver exp 044: dio 0.6801 y +0.0001 al ensamble.)** Era lo más grande que teníamos sin probar. 97 de las 112 features cambian de escala más de 2x entre secuencias (mediana 3,87x, peores 10x) y las normalizábamos globalmente. Verificado en la laptop: la variación de escala que ve la red baja de **4,07x a 1,09x**. ~~Conecta con el oráculo de +0.047 de septiembre, atacándolo desde la entrada en vez de la salida.~~ **(2026-10-24: esa conexión es FALSA, el oráculo no es de escala. Ver exp 043. La justificación del sprint 23 queda sola y es más modesta.)** Criterio: 0.6855. Generador: `notebooks_kaggle/make_sprint23.py` |
| **041** | 2026-10-21 | **Echo State Network** (reservorio de 400 unidades con pesos recurrentes **aleatorios y fijos**, radio espectral 0.95, fuga 0.3; solo se entrena la salida, que es una regresión lineal con ridge). 600 secuencias de train, evaluada en las mismas 200 de valid | 600 | **0.6099** | — | — | 8 min | **laptop, CPU** | **Correlación con el ensamble 0.8756: la más baja de cualquier competidor que construimos** (árboles 0.927, convolucional 0.978, LSTM 0.985). Pasa el umbral `s > S·r` por +0.0184, el margen más grande de todos, y aun así la **ganancia honesta es −0.0019**. Cierra la rama de arquitecturas: cinco familias distintas, correlaciones de 0.88 a 0.99, todas dan cero |
| **042** | 2026-10-21 | **Sprint 24 — ensamble diverso de verdad.** 4 alumnos, cada uno con **datos distintos** (70 % de las secuencias, 7.424 de 10.607, semilla propia), **maestros distintos** (un par de los 4, no los 4 juntos) y **receta distinta** (lr, dropout, y uno GRU en vez de LSTM). 12 épocas, SWA 9–12 | 7.424 c/u | **0.6829** / **0.6819** / **0.6777** | — | — | ~1,7 h c/u | Kaggle T4 (cuenta `joaqfio`) | **Ataca la causa medida el mismo día:** nuestros 4 modelos están destilados de los mismos maestros, correlacionan 0.9854 y promediarlos vale +0.0021, contra +0.0051 de 4 modelos sin maestros compartidos. **Nunca habíamos variado nada entre los modelos salvo la semilla.** El puntaje individual va a ser más bajo (70 % de datos, 2 maestros): lo que decide es la correlación entre los cuatro. Si baja a ~0.95 con individual ~0.681, el ensamble va a ~0.690 → LB ~0.658 → puesto ~235. Generador: `notebooks_kaggle/make_sprint24.py`. **RESULTADO: corrieron 3 de 4 (`div_b` cancelado por cuota). La correlación bajó 0.0065, pero TODO lo puso el cambio de arquitectura de `div_d` (GRU): el par de LSTM con datos y maestros distintos quedó en 0.9849 contra 0.9860. Los tres PASAN la regla `s > S·r` con margen y la mejor cuaterna de los 7 gana +0.0000. Ver la sección de abajo: la regla falla.** |
| **043** | 2026-10-24 | **¿El +0.047 de septiembre se puede alcanzar?** Descomposición del oráculo afín por secuencia y predicción de cada pieza desde las features (resumen de las primeras 2.000 filas: media, desvío y variación media de las 112, más 6 números de nuestras propias predicciones). Ridge ajustado en 936 secuencias, medido en 937 | — | **no se alcanza** | — | — | 6 min | **laptop, CPU** | **El +0.047 es un oráculo conjunto y sus dos piezas son impredecibles por separado: R² = +0.003 cada una.** Oráculo del factor + corrimiento predicho: −0.0316. Factor predicho + oráculo del corrimiento: +0.0035. Y el oráculo de *igualar la escala* (σ_y/σ_q) da **−0.0566**: igualar escalas empeora, así que el +0.047 nunca fue un problema de escala. **Lo que sí se predice es la confianza por secuencia r_s, R² = +0.57 fuera de muestra** —la señal más fuerte del proyecto— pero usarla como multiplicador empeora incluso con el oráculo (−0.0244). Scripts: `analisis/escala_paso{2,3,4}.py` |
| **044** | 2026-10-25 | **Sprint 23 — normalización adaptativa por secuencia** (`--adapt-span 8000`), la única palanca que cambia *qué ve* la red en vez de agregar otro modelo. 10 épocas, SWA 7–10, receta de la alumna LR, semilla 2232 | 10.607 | **0.6801** (SWA) / 0.6794 (best) | — | — | 7,7 h | Kaggle T4 | **Funcionó como diversificador y no alcanzó igual. r con el ensamble 0.9832: la más baja de cualquier red que entrenamos** (los viejos están en 0.9939–0.9955). Probadas las 70 cuaternas de los 8 modelos: la mejor da 0.6861 contra nuestro 0.6860, **+0.0001**, y la nuestra queda 4ª de 70. Con 5 modelos +0.0002, pero no entran en los 60 min. **Cierra el último frente técnico.** La corrida hermana `adapt2000` murió: 695 min solo en los soft targets contra 37 min de ésta con el mismo código — suerte de máquina otra vez, y la guarda no lo agarró porque mide el arranque. Script: `analisis/s23_mide.py` |
| 019 | 2026-09-18/19 | **Sprint 3, 1.ª corrida: 5 alumnos destilados de Dswa+Eswa+F+Gswa** con soft targets precalculados (trainer v4) | 10.607 | **FALLÓ** (los 5) | | | | Kaggle T4 | `CUDA out of memory` en el precálculo (forward de 20.000 pasos de una vez). Fix: por trozos de 2.000. Relanzado con prueba de humo |

## Próximos pasos

1. ~~Loss alineada a la métrica~~ ✅ hecho (exp 004).
2. ~~Más épocas~~ ✅ 3 épocas (exp 006). Falta probar: lr, gradient clipping, más épocas aún.
3. Modelo más grande (h128, 2 capas).
4. ~~`train_official_gpu_v1.py` + primera corrida grande en Kaggle~~ ✅ (exp 009: 0.584 > baseline 0.568).
5. ~~Evaluar en todo valid~~ ✅ (eval_ensemble.py). ~~Restricción dominante: tiempo de inferencia (solo 2 modelos entran)~~ → **kernel v2 (int8): entran 3×128 en ~30 min y 4×128 en ~40.** ~~Destilación~~ ✅ **funciona (exp 016: 0.6805 solo)**; ~~loss global / coseno~~ ✗ no aportan (017, 018). → ~~Sprint 3: 5 alumnos de los mismos maestros~~ (falló por OOM; reemplazado) → **Sprint 4 (2026-09-19): 6 maestros diversos (x: GRU 128x2 / 200x1 / 64x3; y: GRU 160x2 / LSTM 128x2 / GRU 128x2 d0.2) → alumnos X (x), Y (y), Z (todos + Eswa + Dswa) GRU 128x2 → cuaterna final.**
6. `solution.py` con GRU en numba + normalización → verificar paridad con PyTorch → medir µs/fila → primer submit.
7. Regularización (dropout / weight decay): train loss baja pero valid no → sobreajuste desde ép. 4.
8. Evaluar en más grupos de valid durante el entrenamiento (`--valid-groups 50`) para que la selección de época sea menos ruidosa.
