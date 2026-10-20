# Benchmarks del kernel de inferencia (2026-09-18)

Scripts usados para diseñar el kernel v2 de `solution.py`. Todos cargan los pesos reales de
`submissions/ens3_A_B_Cswa/weights.npz` (formato v1, float32) y miden µs/fila con 20.000 filas aleatorias.

- `kbench6.py [n_modelos] [núcleo] [reps]` — round-robin de 10 variantes (filas / 8 filas / traspuesto × float32 / int16 × libm / tanh racional), fijado a un núcleo con `psutil`. Imprime el mínimo por variante y la paridad contra v1.
- `kbench7.py` — segunda vuelta: int8 traspuesto, bloques de 8 k, overhead de llamada según cantidad de argumentos.
- `quant_eval.py [n_seqs]` — efecto de int16/int8 (escala por fila) en la métrica GLOBAL real, con torch en CPU sobre valid.
- `asm_check.py` — cuenta instrucciones AVX (`vfmadd231ps ymm`) en el asm que genera numba, para saber si vectoriza.

Resultado (3×GRU128x2, Ryzen 7735U): v1 85 µs → v2 (traspuesto + int8 + tanh racional) 54 µs. Métrica idéntica.
Los números absolutos dependen de la carga de la máquina; comparar siempre dentro de una misma corrida round-robin.
