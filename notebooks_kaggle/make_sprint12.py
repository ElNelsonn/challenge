r"""Genera los notebooks del sprint 12 (2026-09-30 noche): CAPA RECURRENTE LENTA (multi-escala).

Es la apuesta grande de la hoja de ruta (idea 2), y la única que cambia la ARQUITECTURA.

Diagnóstico que la motiva: después de 28 experimentos, todo aterriza en el mismo número (0.680 crudo / 0.684
destilado / 0.6855 ensamble) sin importar maestros, tamaño, features, receta ni función de costo. Y los errores
de todos nuestros modelos correlacionan 0.96–0.98: son el mismo modelo con distinta cara. Dos causas posibles,
y esta idea ataca las dos:

  1. El entrenamiento corta el gradiente cada 512 pasos, así que la red no puede aprender a usar información
     de hace miles de filas (la secuencia tiene 20.000).
  2. Necesitamos un modelo cuyos errores NO estén correlacionados con los actuales, para que el ensamble
     vuelva a sumar.

Cómo: una segunda red chica que corre UNA VEZ CADA k FILAS, alimentada con el promedio de los estados de la
red rápida durante esas k filas, y cuyo estado vuelve como contexto a la salida. Las filas de una ventana usan
el contexto de la ventana anterior (causal, y calculable online en el kernel).

Costo en inferencia: 1.152 MACs por fila = **0,5 % más** que la LSTM 128x2 sola. Prácticamente gratis.

  * connectome_sprint12_slow32.ipynb  — capa lenta de 64 unidades cada 32 filas (~4 h)
  * connectome_sprint12_slow256.ipynb — cada 256 filas: contexto mucho más lento, memoria más larga (~4 h)

Uso (desde challenge/):  python notebooks_kaggle/make_sprint12.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TRAIN = ("/kaggle/input/datasets/joaquinfiorina/connectome-train-part1/train_part1.parquet "
         "/kaggle/input/datasets/joaquinfiorina/connectome-train-part2/train_part2.parquet")
VALID = "/kaggle/input/datasets/joaquinfiorina/connectome-valid/valid.parquet"
MODELS = "/kaggle/input/datasets/joaquinfiorina/connectome-models"
TEACHERS = {"E": "gru128_distill_seed2029_swa.pt", "l1": "lstm128x2_l1_seed2047_swa.pt",
            "LR": "lstm128x2_studentLR_seed2072_swa.pt", "LA": "lstm128x2_studentLA_best.pt"}
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())
BASE = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
        f"--output-dir /kaggle/working/results --save-valid-preds --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
        f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --epochs 10 --swa-start 7 --slow-hidden 64 "
        f"--distill-weight 1.0 --distill-cache /kaggle/tmp/soft_S8.npz --distill-teachers {teacher_paths}")
RUNS = {"slow32": (32, 2121), "slow256": (256, 2122)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for name, (every, seed) in RUNS.items():
    cells = [
        md(f"# Connectome — sprint 12: **capa recurrente lenta** cada {every} filas\n\n"
           "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
           "**Se puede correr a la vez que el otro notebook del sprint 12.**\n\n"
           "**Por qué esto y no otra cosa:** después de 28 experimentos, todo aterriza en el mismo número sin importar "
           "maestros, tamaño, features, receta ni función de costo. Y los errores de todos nuestros modelos correlacionan "
           "0.96–0.98: son el mismo modelo con distinta cara. **Esto es lo primero que cambia la arquitectura.**\n\n"
           "**Qué hace:** además de la red rápida de siempre (una pasada por fila), corre una **segunda red chica una vez "
           f"cada {every} filas**, alimentada con el promedio de lo que vio la rápida en esas {every} filas. Su estado vuelve "
           "como contexto a la salida. Ataca dos problemas a la vez:\n\n"
           "1. **Memoria larga:** el entrenamiento corta el gradiente cada 512 pasos, así que hoy la red no puede aprender "
           "a usar información de hace miles de filas. La lenta sí llega.\n"
           "2. **Diversidad real:** un modelo con otra estructura debería equivocarse en otros lugares, y ahí el ensamble "
           "vuelve a sumar.\n\n"
           f"| | |\n|---|---|\n| modelo | LSTM 128x2 + **red lenta de 64 unidades cada {every} filas**, lr 5e-4, 10 épocas, SWA 7–10, semilla {seed} |\n"
           "| costo en inferencia | **+0,5 %** (1.152 MACs por fila): prácticamente gratis |\n"
           "| comparar contra | **la alumna LR = 0.6834** y contra los resultados del sprint 11 |\n\n"
           "Verificado en la laptop antes de gastar GPU: es estrictamente causal (cambiar el futuro no altera el pasado) "
           "y procesar por trozos da **exactamente** el mismo resultado que la secuencia entera.\n\n"
           "**Si gana ≥ 0.687**, hay que implementar la capa lenta en `solution.py` (un acumulador y una red chica cada "
           f"{every} filas) antes de poder submitear."),
        code("import os, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             f"for p in {TRAIN.split()} + ['{VALID}']:\n"
             "    assert os.path.isfile(p), f'Falta input: {p}'\n"
             f"for f in {list(TEACHERS.values())}:\n"
             f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 7'\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
        md("## Prueba de humo (~4 min)"),
        code("# 1 época de 32 secuencias con la capa lenta. Tiene que imprimir 'params' mayor que 256.258. Corta si falla.\n"
             f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
             f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 "
             f"--slow-hidden 64 --slow-every {every} --seed 1 --model-seed 1 --distill-teachers {teacher_paths} "
             f"--distill-cache '' --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## Modelo con capa lenta cada {every} filas (10 épocas)"),
        code(f"!python train_official_gpu_v1.py {BASE} --slow-every {every} --seed {seed} --model-seed {seed} "
             f"--run-name student_{name} || echo 'FALLÓ student_{name}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/student_{name}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva por epoca:', curva)\n"
             "    print('referencia: alumna LR (misma receta, sin capa lenta, 8 epocas) = 0.6834')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint12_{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
