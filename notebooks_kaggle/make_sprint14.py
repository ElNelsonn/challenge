r"""Genera los dos notebooks del sprint 14 (2026-10-07): OTRA CLASE DE MODELO (convolucional dilatada).

Por qué es lo único que queda. Dos mediciones nuestras lo justifican:

  1. Sprint 13: dos modelos entrenados sobre mitades DISJUNTAS de los datos (ni una secuencia en común) dan
     0.6806 y 0.6815, y sus predicciones correlacionan **0.984**. Dos modelos que no compartieron un solo dato
     aprenden la misma función: dentro de la familia recurrente ya está todo exprimido.
  2. La señal vive en un horizonte MUY corto: suavizar el target destruye la correlación (0.663 -> 0.499 ->
     0.260 al promediar 20 y 100 filas). Una convolucional dilatada mira explícitamente los últimos 683 pasos
     con filtros aprendidos, en vez de comprimirlos en un estado. Está mejor matcheada a una señal local.

**SIN destilación, a propósito.** Destilar la ataría a imitar a la familia recurrente, que es justamente de lo
que queremos escapar. La referencia es l1, la LSTM 128x2 sin destilar: **0.6808**.

Dos corridas que cambian SOLO el paso de aprendizaje, porque la receta de lr la afinamos para la LSTM y no
tenemos idea de qué le va bien a esta arquitectura. Así, si falla, sabemos que falla la arquitectura y no que
la entrenamos mal.

  * connectome_sprint14_tcn_lr5e4.ipynb  — 128 canales x 5 capas, lr 5e-4 (~2 h)
  * connectome_sprint14_tcn_lr15e3.ipynb — igual, lr 1.5e-3 (~2 h)

Costo en inferencia: 260.352 operaciones por fila, prácticamente igual que la LSTM 128x2 (254.208).

Uso (desde challenge/):  python notebooks_kaggle/make_sprint14.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TRAIN = ("/kaggle/input/datasets/joaquinfiorina/connectome-train-part1/train_part1.parquet "
         "/kaggle/input/datasets/joaquinfiorina/connectome-train-part2/train_part2.parquet")
VALID = "/kaggle/input/datasets/joaquinfiorina/connectome-valid/valid.parquet"
BASE = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
        f"--output-dir /kaggle/working/results --save-valid-preds --lr-decay 0.9 --weight-decay 1e-5 "
        f"--arch tcn --tcn-channels 128 --tcn-layers 5 --tcn-kernel 3 --dropout 0.1 --epochs 10 --swa-start 7")
RUNS = {"lr5e4": ("5e-4", 2141), "lr15e3": ("1.5e-3", 2142)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for name, (lr, seed) in RUNS.items():
    cells = [
        md(f"# Connectome — sprint 14: **convolucional dilatada**, paso {lr}\n\n"
           "**Inputs:** train part1/part2, valid. GPU T4 x2. **Save & Run All.** ~2 h.\n\n"
           "**Correr los dos notebooks del sprint 14 a la vez.**\n\n"
           "**Por qué esto es lo único que queda.** Dos mediciones nuestras:\n\n"
           "1. Entrenamos dos modelos sobre **mitades disjuntas** de los datos — ni una secuencia en común — y "
           "sus predicciones correlacionan **0.984**. Dos modelos que no compartieron un solo dato aprenden la "
           "misma función: dentro de la familia recurrente ya no hay nada más que sacar.\n"
           "2. La señal vive en un **horizonte muy corto**: suavizar el target la destruye (0.663 → 0.499 → 0.260 "
           "al promediar 20 y 100 filas).\n\n"
           "Una convolucional dilatada mira **explícitamente los últimos 683 pasos** con filtros aprendidos, en vez "
           "de comprimirlos en un estado como hace la LSTM. Es otra forma de procesar la misma información, y está "
           "mejor matcheada a una señal local.\n\n"
           f"| | |\n|---|---|\n| modelo | 128 canales × 5 capas, dilataciones 1/4/16/64/256, campo receptivo 683 filas |\n"
           f"| entrenamiento | lr **{lr}**, dropout 0.1, wd 1e-5, 10 épocas, SWA 7–10, semilla {seed} |\n"
           "| destilación | **no**, a propósito: destilar la ataría a imitar a la familia recurrente |\n"
           "| costo en inferencia | 260.352 operaciones por fila (la LSTM 128x2 usa 254.208): mismo presupuesto |\n"
           "| comparar contra | **l1 = 0.6808**, la LSTM 128x2 sin destilar |\n\n"
           "Las dos corridas cambian **solo el paso de aprendizaje**, porque el que usamos lo afinamos para la LSTM "
           "y no sabemos qué le va bien a esta arquitectura. Si falla con los dos, falla la arquitectura; si uno "
           "anda, aprendimos el paso correcto.\n\n"
           "**Qué miro después:** el score, y sobre todo **la correlación entre sus predicciones y las de la familia "
           "recurrente**. Si baja de 0.95, por primera vez tenemos un modelo que mira otra cosa, y ahí el ensamble "
           "vuelve a tener sentido aunque el modelo solo empate.\n\n"
           "Verificado en la laptop antes de gastar GPU: es estrictamente causal y procesar por trozos da "
           "**exactamente** lo mismo que la secuencia entera."),
        code("import os, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU T4 x2)'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             f"for p in {TRAIN.split()} + ['{VALID}']:\n"
             "    assert os.path.isfile(p), f'Falta input: {p}'\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
        md("## Prueba de humo (~3 min)"),
        code("# 1 época de 32 secuencias. Tiene que imprimir 'TCN 128 canales x5 capas  campo receptivo=683'. Corta si falla.\n"
             f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 "
             f"--valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --arch tcn --tcn-channels 128 --tcn-layers 5 "
             f"--dropout 0.1 --lr {lr} --seed 1 --model-seed 1 --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## Convolucional dilatada, paso {lr}"),
        code(f"!python train_official_gpu_v1.py {BASE} --lr {lr} --seed {seed} --model-seed {seed} "
             f"--run-name tcn_{name} || echo 'FALLÓ tcn_{name}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/tcn_{name}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva por epoca:', curva)\n"
             "    print('referencia: LSTM 128x2 sin destilar (l1) = 0.6808')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint14_tcn_{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
