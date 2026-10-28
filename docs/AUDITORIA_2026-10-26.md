# Auditoría del challenge — 26 de octubre de 2026

El estancamiento es real, pero el diagnóstico anterior mezcla referencias distintas y convierte resultados negativos de pruebas concretas en conclusiones demasiado generales. Antes de gastar otra tanda de GPU hay que corregir esas referencias. En esta sesión no apareció una mejora de puntaje demostrada; sí quedaron corregidos riesgos del código y documentados errores que podían orientar mal los próximos experimentos.

Trabajé sobre el repositorio y los archivos locales reales. Recalculé cinco modelos y sus combinaciones sobre las 3.528.273 filas puntuadas de las 1.873 secuencias, contrasté el cache completo contra el parquet, ejecuté el baseline en los diez grupos fijos y examiné 24 secuencias de train y 24 de valid. No lancé entrenamiento, subí archivos ni hice submissions.

**1. La métrica del entrenador es correcta; parte del handoff no lo es.**

La [documentación oficial vigente](https://wundernn.io/connectome/docs/data_overview) confirma WP global, con targets y predicciones recortados a ±2, pesos |target| y promedio entre los dos targets. Los archivos originales del starter pack describen la versión anterior, por secuencia. No hay que cambiar el entrenamiento a esa versión antigua.

| Referencia | WP global | WP por secuencia | Verificación |
|---|---:|---:|---|
| Baseline, diez grupos fijos | 0,649812330 | 0,567949279 | Ejecutado en esta sesión |
| Baseline, valid completo | 0,617052286 | 0,589594483 | Registro histórico local de evaluación completa |

El handoff asignaba **0,589595** al baseline global sobre todo valid. Es la métrica vieja. El JSON histórico y el registro central ya contenían la distinción correcta. No repetí la evaluación completa del baseline, que es costosa.

Para la comprobación nueva ejecuté el mismo grafo ONNX fila por fila, reiniciando los estados por secuencia. La versión instalada de ONNX Runtime no acepta una propiedad usada por el wrapper original; el script de auditoría configura directamente la sesión y conserva la misma recurrencia.

**2. El ensamble analizado no coincide con el ZIP asociado a la última entrega.**

El archivo BUILD.json de `ens4_mask_LR_LA_long_int8` enumera:

- student_mask **SWA**;
- studentLR **SWA**;
- studentLA **best**;
- student_long **best**.

Los cuatro son LSTM de **128 unidades y dos capas**. Reconstruí sus pesos desde esos checkpoints: todos los arrays exportados coinciden exactamente con weights.npz del artefacto local. Su kernel coincide por SHA-256 con el kernel actual.

En cambio, los análisis recientes usan **LR best + LA best + long best + b256 SWA**. Cambian tanto un miembro como la variante de LR. Por lo tanto, llamar a esa combinación «la entrega actual» y atribuirle su tiempo medido es incorrecto. La asociación entre ese ZIP y el resultado público proviene del diario; no accedí a la cuenta de submissions para confirmarla.

| Combinación, con clipping después del promedio | WP global | t0 | t1 |
|---|---:|---:|---:|
| LR best + LA best + long best + b256 SWA | 0,686047560 | 0,682347946 | 0,689747173 |
| LR best + LA best + long best + mask SWA | 0,686266804 | 0,682718939 | 0,689814669 |

La segunda fila aproxima el artefacto: aún usa LR best en lugar de LR SWA. **No es una evaluación exacta del ZIP.** Su ventaja de +0,000219 tiene un intervalo del 95 % de **−0,000250 a +0,000651**, remuestreando secuencias completas 1.000 veces. No demuestra una mejora.

**3. El presupuesto de inferencia estaba mal calibrado.**

El handoff ancla 41 min 52 s a tres redes de 128 más una de 256. El artefacto local que corresponde a esa referencia tiene cuatro de 128. La tabla de «seis minutos por LSTM 128» deriva de esa atribución equivocada.

Comparé ambos builds usando el callback, el mismo núcleo de CPU, dos secuencias reales completas y tres repeticiones por build. Hice dos mediciones: la primera coexistió parcialmente con otras tareas de auditoría; la segunda repitió la comparación con menos carga.

| Medición local | Costo relativo de 3×128 + 1×256 frente a 4×128 | Proyección usando 41 min 52 s como ancla |
|---|---:|---:|
| Primera | 1,584× | 66,3 minutos |
| Segunda | 1,829× | 76,6 minutos |

Son proyecciones condicionadas al ancla histórica, no tiempos medidos en el servidor. La variación local impide prometer un tiempo preciso. **No está demostrado que la combinación con b256 entre en 60 minutos.** La cuaterna de 128 sí tiene el antecedente de entrega del diario. No conviene diseñar otro ensamble contando con veinte minutos libres en la combinación con b256.

**4. Hay un error de causalidad en los análisis de confianza y escala.**

En escala_paso3 y escala_paso4, Q ya contiene únicamente filas puntuadas. La expresión `Q[t][:2000]` toma hasta 2.000 de esas filas; no toma las predicciones de los primeros 2.000 pasos reales.

| Punto final de ese resumen por secuencia | Paso real |
|---|---:|
| Percentil 10 | 10.962,2 |
| Mediana | 19.650 |
| Percentil 90 | 19.987 |

En el **100 %** de las secuencias el resumen termina después del paso 2.000. En el 63,5 % utiliza todas las filas puntuadas de la secuencia porque no hay 2.000. Además, depende de una máscara que no llega al modelo durante inferencia.

El resumen X de las features originales sí se construye con los primeros 2.000 pasos. El problema está en el resumen adicional de predicciones. Así, el R²≈0,57 que lo incorpora **no demuestra confianza predecible al inicio de una secuencia**. Incluso un resumen correctamente calculado hasta el paso 1.999 solo se puede usar desde ese momento en adelante, sin modificar predicciones anteriores. Esto afecta la interpretación de esos experimentos; no encontré evidencia de que ese resumen se haya desplegado en el ZIP.

**5. El supuesto orden real del libro no quedó demostrado.**

Los tests anteriores ordenan valores anonimizados y llaman «mejor bid real» a su máximo. Si cada columna tiene una transformación diferente, esa comparación no identifica el orden de los precios originales. La estabilidad del ranking normalizado tampoco resuelve el problema.

En la nueva muestra de 24 secuencias de valid, p10 es el máximo normalizado del lado bid de i0 en el 99,62 % de las filas y p14 es el mínimo normalizado del ask en el 97,96 %. Al centrar cada columna por separado, la concentración máxima de esos índices baja a aproximadamente 60 % y 59 %. El ranking depende fuertemente de los offsets de las columnas. Este centrado usó secuencias completas como diagnóstico; no es una propuesta de feature causal.

También aparecieron varios órdenes medios entre las secuencias muestreadas: tres para bids de i0 y dos para asks. No se reproduce universalmente el «12/12 y orden fijo» del muestreo anterior.

Hay además un salto lógico en la explicación anterior: una capa lineal puede deshacer normalizaciones **afines conocidas o aprendibles**, pero no una transformación arbitraria por cuantiles. El valor extremo ±5,1993 es compatible con saturación de una transformación gaussiana por cuantiles; por sí solo no identifica todo el preprocesamiento. La correspondencia precio-volumen tampoco está verificada.

Por ahora no llamaría spread verdadero, microprecio ni desequilibrio real a cocientes de esas columnas. Las cross_features existentes son combinaciones numéricas que se pueden probar, pero su interpretación física no está acreditada. Cambiar p0 por p10 no es una corrección demostrada.

Probé también una explicación alternativa: que todos los precios de un lado fueran copias transformadas de una sola variable. Los cambios temporales están muy sincronizados, pero las relaciones entre niveles no son deterministas ni lineales; aplicar la CDF normal tampoco los reduce a una sola variable. **No hay evidencia para comprimir los 44 precios a cuatro columnas.**

**6. La validación partida anterior sigue teniendo filtraciones de selección.**

En ensamble_todo2 se escoge primero best o SWA de cada corrida por su resultado en TODO valid. Después se elige la combinación en la mitad A y se mide en B. B ya participó en una selección. Además, los checkpoints best originales y muchas decisiones anteriores fueron elegidos usando ese valid.

Partir por secuencia es correcto y mejora el análisis, pero no vuelve a crear datos nuevos. Sus resultados sirven como evidencia exploratoria; no son una prueba independiente de que «ninguna combinación puede mejorar» o de que la familia alcanzó un límite de información.

Otro detalle: varios scripts recortan cada modelo antes de promediar. La entrega promedia primero y la métrica recorta el resultado. En la cuaterna documentada la diferencia de WP es solo **0,000000170**: no explica el estancamiento, aunque había que comprobarla.

La pérdida usada por target_cd también cambia el objetivo al calcular pesos y clipping sobre suma/diferencia. Su resultado negativo evalúa esa receta completa; no demuestra que la mera reparametrización sea inútil. El registro histórico ya mencionaba este riesgo.

**7. La regla de correlación fue aplicada e interpretada incorrectamente.**

Para un solo target, sin activación del clipping, la derivada al incorporar una fracción α del modelo q al ensamble e es:

`d corr_w(y, (1−α)e + αq) / dα en α=0 = (σq,w / σe,w) × [corr_w(y,q) − corr_w(y,e) × corr_w(e,q)]`.

Todos los términos deben usar **los mismos pesos del target**. Después se promedian las dos derivadas. No requiere suponer errores intercambiables. Usar una correlación sin ponderar del target 0, mezclarla con el promedio de dos scores y anticipar una sustitución de modelos no cumple esas condiciones.

Además, una derivada positiva solo dice que una cantidad suficientemente pequeña puede ayudar; no garantiza que ayude un peso del 20 % o reemplazar un cuarto del ensamble. En las predicciones auditadas, long tiene derivada aproximada **+0,000326**, pero darle 20 % adicional produce **−0,000039**. La prueba por diferencias finitas pequeñas concuerda aproximadamente con la derivada; el clipping explica parte de la diferencia residual.

La decisión práctica sigue siendo mezclar y medir con el presupuesto real y una evaluación separada. Lo que no corresponde es concluir que la identidad matemática falló.

**8. Qué intenté mejorar hoy y qué quedó corregido.**

Probé dos calibradores globales baratos usando únicamente las dos predicciones del ensamble: uno lineal y otro cuadrático. Ajusté cada uno en cuatro quintas partes de las secuencias y lo evalué en la quinta restante, rotando cinco veces.

| Transformación | WP global cruzado | Diferencia | Intervalo del 95 % de la diferencia |
|---|---:|---:|---:|
| Referencia | 0,686047560 | — | — |
| Lineal, dos salidas e intercepto | 0,686002098 | −0,000045 | [−0,000255; +0,000148] |
| Cuadrática | 0,685997237 | −0,000050 | [−0,000326; +0,000221] |

No hay una mejora útil. Estos folds también son exploratorios porque el ensamble base ya fue elegido usando valid. No recomiendo desplegar esos calibradores.

Corregí tres rutas del código que podían aceptar un modelo con normalización adaptativa y aplicarle mean/std fijos: el exportador, el cargador de teachers y el evaluador de ensambles. Ahora rechazan ese pipeline con una explicación, hasta que se implemente la inferencia causal correspondiente. Antes, incluso la comparación del exportador contra su referencia podía dar una falsa sensación de paridad porque ambos omitían la adaptación.

También cambié la identidad del cache de soft targets: antes dependía solo del basename de cada checkpoint, por ejemplo best_model.pt. Dos maestros distintos con ese nombre podían reutilizar el mismo cache. Ahora usa SHA-256 del contenido. Los caches antiguos se invalidan una vez. No encontré prueba de que este riesgo hubiera contaminado entrenamientos históricos; el arreglo previene el caso.

Las **siete pruebas de regresión pasan**, incluida la igualdad de salidas entre exportador y teacher con normalización fija. El kernel y los ZIP existentes permanecen iguales.

**9. Cómo seguir, en orden.**

Primero hay que fijar una referencia exacta: evaluar el artefacto de cuatro redes de 128 con sus variantes reales, incluida LR SWA, y compararlo con el mejor ZIP histórico. Guardar hashes, secuencias evaluadas, score por target y tiempo. No atribuirle el resultado del ensamble con b256.

Después prepararía una sola comparación controlada, no otra búsqueda de decenas de combinaciones. La hipótesis a investigar es el manejo separado de los eventos de ambos instrumentos: medir cuándo se actualiza cada bloque y probar relojes causales de actualización antes de construir una arquitectura más compleja. La sincronización observada de los precios permite definir esos eventos sin inventar profundidad. Los packs de eventos grandes y el análisis de diferencias de i1 ya se intentaron; esto debe probar una representación pequeña y explícita, no repetir esos packs.

El primer piloto debería comparar la misma LSTM 128×2 con entradas originales frente a originales más indicadores y tiempo desde la última actualización de i0/i1. Mismos grupos, semilla, épocas, pérdida y SWA; sin destilar para no mezclar el cambio de entrada con la presión de los maestros. La señal es una hipótesis, no una ganancia prevista.

Para obtener una comparación nueva, reservar secuencias de train y entrenar ambos candidatos desde cero excluyéndolas. Los maestros antiguos también vieron train: usarlos sobre ese nuevo holdout mediante destilación rompería su independencia. Conservar valid completo para continuidad histórica, sin presentarlo como un conjunto nunca usado. Fijar el protocolo antes de mirar el resultado y repetir una segunda semilla solo si aparece una diferencia consistente.

Antes de una corrida grande: comprobar causalidad por prefijos, reinicio de estado, igualdad entre evaluación completa y por chunks, exportación y costo por fila. Priorizar una ganancia pareada con intervalo por secuencia y presupuesto verificado. Un +0,0002 seleccionado entre muchas alternativas no justifica proclamar avance.

Como contexto actualizado, el [leaderboard público](https://wundernn.io/connectome/leaderboard), consultado el 26/10, mostraba 0,67433 en el primer puesto, 0,66593 en el 50 y 0,66303 en el 100. La brecha frente al 0,6542 histórico es de 0,00883 para el top 100; no se puede convertir una mejora pequeña de valid en una posición prometida usando una resta fija.

**Archivos y reproducción.**

Los números completos están en auditoria_resultados.json, auditoria_precios.json y auditoria_tiempos.json. El respaldo local incluye los scripts, las correcciones y las pruebas. El informe también quedó copiado al directorio docs del repositorio y enlazado desde los handoffs.

Desde la raíz del repositorio challenge, con Python local y las dependencias ya instaladas:

```powershell
python analisis\auditoria_20261026.py --baseline --output 'C:\Users\Joaquin\Documents\Codex\2026-10-26\est\outputs'
python analisis\auditoria_precios_20261026.py --output 'C:\Users\Joaquin\Documents\Codex\2026-10-26\est\outputs'
python -m unittest discover -s tests -p test_pipeline_guards.py -v
```

El baseline de diez grupos se volvió a ejecutar; el número de baseline completo proviene del registro histórico identificado arriba. No se verificó en esta sesión la cuota actual de Kaggle, el estado de la cuenta de competición ni el tiempo de servidor de un ZIP nuevo. Los cambios quedaron locales, sin commit ni push.
