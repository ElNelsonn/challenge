r"""Genera los notebooks del sprint 22 (2026-10-20): el horizonte de aprendizaje (chunk), que nunca tocamos.

QUÉ ES. Las secuencias tienen 20.000 filas. Entrenar una red recurrente sobre 20.000 pasos de una sola vez no
entra en memoria, así que se parte en trozos: la red procesa 512 filas, actualiza los pesos, **corta el
gradiente**, y sigue con las 512 siguientes llevándose el estado pero no el gradiente. Eso se llama BPTT
truncado y el tamaño del trozo es `--chunk`.

**La consecuencia es concreta: la red solo puede APRENDER relaciones de hasta 512 filas de distancia.** Si lo
que predice t0 ahora depende de algo que pasó 1.500 filas atrás, el gradiente nunca conecta las dos cosas y la
red no puede aprenderlo, aunque su estado interno sí arrastre la información.

Está en 512 desde el experimento 008, el 15 de septiembre, y **nunca lo variamos**. Es el último parámetro
estándar sin probar: está fuera de la lista de "no volver a probar" del plan.

LA EVIDENCIA EN CONTRA, PARA NO ENTUSIASMARSE. Las features de memoria larga (promedios móviles de 100, 1.000
y 10.000 filas) dieron **0.0000 dos veces**, lo que sugiere que la información vieja no sirve. Pero no es la
misma prueba: esas features le **dan masticado** un promedio largo concreto, mientras que un chunk más largo
le permite **aprender sola** qué estructura vieja importa. Son dos cosas distintas y la segunda nunca se probó.

  * connectome_sprint22_chunk1024.ipynb — el doble de horizonte
  * connectome_sprint22_chunk2048.ipynb — el cuádruple

Todo lo demás idéntico a la alumna LR (0.6834): LSTM 128x2 destilada, lr 5e-4, 10 épocas, SWA 7-10, batch 32.
Criterio: sirve si pasa de 0.6855.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint22.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TEACHERS = ["gru128_distill_seed2029_swa.pt", "lstm128x2_l1_seed2047_swa.pt",
            "lstm128x2_studentLR_seed2072_swa.pt", "lstm128x2_studentLA_best.pt"]
CORRIDAS = {"chunk1024": (1024, 2221), "chunk2048": (2048, 2222)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


BUSCAR = ("import os, time, glob, torch\n"
          "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
          "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
          "print('memoria de la GPU: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))\n"
          "\n"
          "def buscar(nombre, espera=600):\n"
          "    \"\"\"Busca el archivo por nombre debajo de /kaggle/input: los datasets grandes a veces tardan\n"
          "    en montarse o aparecen en otra ruta.\"\"\"\n"
          "    t0 = time.time()\n"
          "    while True:\n"
          "        hits = glob.glob(f'/kaggle/input/**/{nombre}', recursive=True)\n"
          "        if hits:\n"
          "            return hits[0]\n"
          "        if time.time() - t0 > espera:\n"
          "            print('NO aparecio:', nombre, '| esto es lo que hay montado:')\n"
          "            for d in sorted(glob.glob('/kaggle/input/*')) + sorted(glob.glob('/kaggle/input/*/*')):\n"
          "                print('   ', d)\n"
          "            raise FileNotFoundError(nombre)\n"
          "        time.sleep(15)\n"
          "\n"
          "TRAIN = ' '.join(buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet'))\n"
          "VALID = buscar('valid.parquet')\n"
          f"TEACHERS = ' '.join(buscar(n) for n in {TEACHERS})\n"
          "print('train:', TRAIN)\nprint('valid:', VALID)\n"
          "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")

for nombre, (chunk, seed) in CORRIDAS.items():
    cells = [
        md(f"# Connectome — sprint 22: **horizonte de aprendizaje {chunk}** (hoy 512)\n\n"
           "**Inputs:** train part1/part2, valid, `Connectome models`. GPU. **Save & Run All.** "
           f"~{3 if chunk == 1024 else 4} h.\n\n"
           "---\n\n"
           "## Qué es esto\n\n"
           "Las secuencias tienen **20.000 filas**. Entrenar una red recurrente sobre 20.000 pasos de una sola "
           "vez no entra en memoria, así que se parte en trozos: la red procesa un trozo, actualiza los pesos, "
           "**corta el gradiente**, y sigue con el siguiente llevándose el estado pero no el gradiente.\n\n"
           "**La consecuencia es concreta: la red solo puede APRENDER relaciones de hasta el tamaño del "
           "trozo.** Si lo que predice t0 ahora depende de algo que pasó 1.500 filas atrás, el gradiente nunca "
           "conecta las dos cosas y la red no puede aprender esa relación, aunque su estado interno sí arrastre "
           "la información.\n\n"
           f"Ese tamaño está en **512 desde el 15 de septiembre** (experimento 008) y **nunca lo variamos**. "
           f"Acá se prueba con **{chunk}**.\n\n"
           "## La evidencia en contra, para no entusiasmarse\n\n"
           "Las features de memoria larga (promedios móviles de 100, 1.000 y 10.000 filas) dieron **0.0000 dos "
           "veces**, lo que sugiere que la información vieja no sirve.\n\n"
           "Pero **no es la misma prueba**: esas features le **dan masticado** un promedio largo concreto, "
           "mientras que un trozo más largo le permite **aprender sola** qué estructura vieja le importa. Es la "
           "diferencia entre darle la respuesta y darle la posibilidad de buscarla.\n\n"
           "| | |\n|---|---|\n| modelo | LSTM 128x2, destilada de los 4 maestros, semilla " + str(seed) + " |\n"
           "| receta | **idéntica** a la alumna LR: lr 5e-4, decaimiento 0,9, 10 épocas, SWA 7–10, batch 32 |\n"
           f"| único cambio | `--chunk {chunk}` en vez de 512 |\n"
           "| comparar contra | **la alumna LR = 0.6834** |\n\n"
           "**Criterio:** sirve si pasa de **0.6855**. Es el último parámetro estándar que quedaba sin tocar: "
           "está fuera de la lista de \"no volver a probar\" del plan.\n\n"
           f"**Aviso de memoria:** el trozo más largo multiplica por {chunk // 512} la memoria de las "
           "activaciones. La prueba de humo de abajo lo verifica antes de empezar la corrida larga, así que si "
           "no entra nos enteramos en 3 minutos y no en 3 horas."),

        code(BUSCAR),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" +
             (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),

        md("## Prueba de humo (~3 min)\n\n"
           "Corre una época con pocas secuencias **con el batch y el trozo de la corrida larga**, que es lo "
           "que decide si entra en memoria, y verifica cuánta memoria de GPU llegó a usar."),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm "
             f"--hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --batch 32 --chunk {chunk} "
             f"--seed 1 --model-seed 1 --distill-teachers $TEACHERS --distill-cache '' --run-name smoke\n"
             "import os, json, torch\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLO'\n"
             "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
             f"assert m['config']['chunk'] == {chunk}, f\"el trozo quedo en {{m['config']['chunk']}}\"\n"
             "print('Prueba de humo OK | trozo:', m['config']['chunk'], '| batch:', m['config']['batch'])\n"
             "print('memoria de GPU usada como maximo: %.1f GB de %.1f' % (\n"
             "    torch.cuda.max_memory_allocated()/1e9, torch.cuda.get_device_properties(0).total_memory/1e9))\n"
             "print('valid WP:', round(m['best_valid_wp'], 4))\n"),

        md(f"## La alumna con horizonte {chunk}"),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups all --valid-groups all --output-dir /kaggle/working/results --save-valid-preds "
             f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
             f"--epochs 10 --swa-start 7 --batch 32 --chunk {chunk} --distill-weight 1.0 "
             f"--distill-teachers $TEACHERS --distill-cache /kaggle/tmp/soft_S22.npz "
             f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"),

        md("## Resumen"),
        code("import json, os\n"
            f"d = '/kaggle/working/results/{nombre}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
             "    mejor = max(m['best_valid_wp'], m.get('swa_valid_wp') or 0)\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva:', [round(v, 4) for v in h])\n"
             f"    print('trozo usado:', m['config']['chunk'], '(la referencia uso 512)')\n"
             "    print()\n"
             "    print('referencia EXACTA: alumna LR, misma receta con trozo 512 = 0.6834')\n"
             "    print('veredicto:', 'SIRVE' if mejor >= 0.6855 else ('ruido' if mejor >= 0.6815 else 'PEOR'))\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint22_{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, f"(trozo {chunk}, semilla {seed})")
