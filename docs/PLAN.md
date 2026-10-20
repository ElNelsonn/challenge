# Plan hasta el 15 de noviembre

Escrito el 2026-10-07. Se actualiza cuando cambia algo. La idea es no tener que decidir sobre la marcha:
cada resultado posible ya tiene definido qué viene después.

## 1. Dónde estamos, con números

| | 2026-09-28 | 2026-10-07 | se movió |
|---|---|---|---|
| 1.º | 0.6677 | **0.6711** | +0.0034 |
| 20.º | 0.6616 | **0.6655** | +0.0039 |
| 50.º | 0.6589 | **0.6623** | +0.0034 |
| 100.º | 0.6566 | **0.6600** | +0.0034 |
| **nosotros** | 0.6542 (puesto 168) | **0.6542 (puesto 290)** | **0** |

**El campo sube ~0.0004 por día.** Quedarnos quietos nos cuesta unos 15 puestos diarios. Nuestro mejor valid es
0.6855 (ensamble) y la relación medida es **leaderboard ≈ valid − 0.031**, así que:

| objetivo | leaderboard | valid necesario | nos falta |
|---|---|---|---|
| top 100 | 0.6600 | ~0.691 | **+0.006** |
| top 50 | 0.6623 | ~0.693 | +0.008 |
| top 20 | 0.6655 | ~0.697 | +0.011 |

Y esos números suben solos mientras pasa el tiempo. **Objetivo realista: top 100.** Top 20 requiere que la
apuesta de arquitectura funcione muy bien y que el campo se frene.

Dato de contexto: los de arriba llevan 20 a 100 submissions; nosotros 8. No es casualidad: iteran más rápido.

## 2. La bifurcación de esta semana (sprint 13)

Están corriendo dos modelos sobre mitades **disjuntas** de los datos. Lo que decide el camino **no es el score**
de cada uno (van a dar peor, ~0.680), sino la **correlación entre sus errores**. Hoy toda nuestra familia está
entre 0.975 y 0.987.

### Rama A — la correlación baja de 0.97

Encontramos la palanca: la diversidad viene de los datos. Plan inmediato (≈10 h de GPU, una semana):

1. Entrenar **4 modelos sobre 4 cuartos disjuntos** (2.650 secuencias cada uno), semillas distintas. ~1 h c/u.
2. Tabla de ensambles local (`combine_local.py`) con los 4 + los que ya tenemos.
3. Si el ensamble de 4 supera 0.6895 en valid (≥ +0.004 sobre 0.6855) → **submission**.
4. Si funciona, escalar: 6 u 8 particiones, y combinar con destilación (un alumno que copie al ensamble de
   particiones, que es como meter 8 modelos en el presupuesto de 1).

**Techo esperado de esta rama:** +0.004 a +0.008 en valid. Alcanza para top 100, no para top 20.

### Rama B — la correlación sigue en 0.98

La diversidad no viene de los datos, y queda descartado el último camino barato. Pasamos a la apuesta grande,
que es **cambiar de familia de modelo**. Dos candidatos, los dos entran en el presupuesto de inferencia:

- **Convolucional dilatada (TCN):** 4 capas de 64 canales con kernel 3 y dilataciones 1/4/16/64 ≈ 49k MACs por
  fila (la LSTM 128x2 usa 254k). Se evalúa incrementalmente con buffers circulares.
- **Atención local:** una ventana de 64 pasos con claves cacheadas sobre la salida de la LSTM ≈ 8k MACs por fila.

Pasos (≈1 semana de trabajo mío + 5 h de GPU):

1. Implementar la arquitectura en el entrenador y verificar causalidad + equivalencia por trozos (como hicimos
   con la capa lenta).
2. Entrenar 10 épocas con la receta congelada (lr 5e-4, destilación, SWA 7–10).
3. **Medir la correlación de sus errores con la familia actual.** Es el número que importa, incluso si el modelo
   solo empata.
4. Si la correlación baja de 0.95 → implementar en `solution.py` (numba) y armar submission con el ensamble.

**Techo esperado:** es lo único con potencial de +0.01, y también lo único que puede no funcionar después de
una semana de trabajo.

## 3. Lo que se hace pase lo que pase

- **Submission de calibración (esta semana).** Nuestro mejor ensamble actual da 0.6860 en valid contra el 0.6843
  que produjo el 0.6542. No va a cambiar el puesto, pero **vuelve a medir la relación valid → leaderboard**, que
  es la herramienta con la que decidimos todo y hace 2 semanas que no la verificamos. Cuesta una submission de
  las 5 diarias que no usamos.
- **Usar más submissions.** Llevamos 8 en un mes. El leaderboard es la única medición no contaminada que tenemos
  y la estamos desperdiciando por miedo a "gastarla".
- **Entrenamiento final con train + valid** (idea de Joaquín: solo al final, no antes). 12.480 secuencias en vez
  de 10.607. La semana del 8 al 15 de noviembre, con la receta ya congelada. Esperable +0.002 a +0.004, y no se
  puede medir en local: el leaderboard es el juez.
- **El informe.** Tenemos 29 experimentos documentados con sus resultados negativos y por qué se descartaron.
  Eso ya está escrito en `EXPERIMENTS.md` y `DIARIO.md`; falta ordenarlo como informe final.

## 4. Calendario

| semana | rama A | rama B | fijo |
|---|---|---|---|
| 7–13 oct | 4 cuartos + tabla + submission | implementar TCN y verificar | submission de calibración |
| 14–20 oct | escalar a 6–8 particiones + destilación | entrenar TCN y medir correlación | |
| 21–27 oct | combinar lo mejor de las dos ramas; si algo funcionó, exprimirlo con 2–3 submissions | | |
| 28 oct – 3 nov | congelar la receta ganadora | | |
| 4–10 nov | reentrenar con train + valid | | |
| 11–15 nov | submission final y verificación de tiempo | | informe |

## 5. Lo que NO hay que volver a probar (está medido)

Features a mano (4 variantes: lags/deltas, packs de eventos, combinaciones dentro de la fila, memoria larga) ·
modelos más grandes (160/192/256/200x1/64x3) · más épocas (el pico está en la 9–10) · la loss de correlación ·
pesos ajustados del ensamble · calibrar la escala por secuencia (el oráculo vale +0.047 pero es inestimable) ·
schedule coseno · LightGBM (no entra en tiempo) · elegir la mejor de muchas combinaciones mirando valid.

## 6. Sobre el repositorio público del competidor

`github.com/dozken/wunder` (puesto ~51) nos dio una pista concreta: su loss. La probamos bien y **no aportó**
(exp 028). Sirve para saber qué están haciendo otros, pero no para copiar: su arquitectura es peor que la
nuestra y su ventaja viene de otro lado. Lo nuestro tiene que salir de nuestras propias mediciones.
