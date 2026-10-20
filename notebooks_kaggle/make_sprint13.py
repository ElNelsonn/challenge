r"""Genera los dos notebooks del sprint 13 (2026-10-05): DIVERSIDAD POR DATOS (mitades disjuntas).

El bloqueo está medido: los errores de los seis modelos que tenemos correlacionan entre 0.975 y 0.987, sin
importar entradas, tamaño, épocas, celda ni función de costo. Por eso el ensamble está clavado en 0.6855:
no estamos combinando modelos distintos, estamos promediando seis versiones del mismo.

La causa más probable es la más simple: todos vieron **exactamente los mismos datos** con la misma receta.
Este sprint lo ataca de frente entrenando dos modelos sobre **mitades disjuntas** de las secuencias (5.303
cada uno, sin una sola secuencia en común).

Qué esperar, y por qué igual vale la pena:
  * Cada modelo va a ser PEOR solo (la mitad de los datos): algo como 0.679-0.681 en vez de 0.6834.
  * Lo que importa es la **correlación entre sus errores**. Si baja de 0.97, encontramos la palanca de la
    diversidad y a partir de ahí se pueden entrenar 3 o 4 modelos sobre particiones distintas y el ensamble
    vuelve a moverse. Si sigue en 0.98, la diversidad no viene de los datos y queda descartada esa vía.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint13.py
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
        f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --epochs 10 --swa-start 7 "
        f"--distill-weight 1.0 --distill-teachers {teacher_paths}")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for half, seed in ((1, 2131), (2, 2132)):
    cells = [
        md(f"# Connectome — sprint 13: **mitad {half} de los datos**\n\n"
           "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~1,5 h.\n\n"
           "**Correr los dos notebooks del sprint 13 a la vez** (mitad 1 y mitad 2).\n\n"
           "**El problema que ataca:** medimos que los errores de nuestros seis modelos correlacionan entre **0.975 y "
           "0.987**, sin importar entradas, tamaño, épocas, tipo de red ni función de costo. El ensamble está clavado "
           "porque no combinamos modelos distintos: promediamos seis versiones del mismo. La explicación más simple es "
           "que todos vieron **exactamente los mismos datos**.\n\n"
           f"**Qué hace:** entrena con la **mitad {half}** de las secuencias (5.303 de 10.607), sin una sola secuencia "
           "en común con el otro notebook.\n\n"
           f"| | |\n|---|---|\n| modelo | LSTM 128x2, lr 5e-4, 10 épocas, SWA 7–10, semilla {seed} |\n"
           f"| datos | mitad {half}: 5.303 secuencias (la otra mitad no la ve nunca) |\n"
           "| referencia | la alumna LR con TODOS los datos = 0.6834 |\n\n"
           "**Importante:** este modelo va a dar **peor** solo, y está bien. Con la mitad de los datos esperamos "
           "0.679–0.681. Lo que vamos a medir después, en la laptop, es la **correlación entre los errores de los dos**. "
           "Si baja de 0.97 encontramos la palanca que falta y se pueden entrenar 3 o 4 sobre particiones distintas. "
           "Si sigue en 0.98, la diversidad no viene de los datos y cerramos esa vía."),
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
        md("## Prueba de humo (~3 min)"),
        code(f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 64 "
             f"--train-half {half} --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 "
             f"--layers 2 --dropout 0.1 --lr 5e-4 --seed 1 --model-seed 1 --distill-teachers {teacher_paths} "
             f"--distill-cache '' --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## Modelo sobre la mitad {half}"),
        code(f"!python train_official_gpu_v1.py {BASE} --train-half {half} --distill-cache /kaggle/tmp/soft_half{half}.npz "
             f"--seed {seed} --model-seed {seed} --run-name student_half{half} || echo 'FALLÓ student_half{half}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/student_half{half}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva por epoca:', curva)\n"
             "    print('referencia con TODOS los datos: 0.6834. Que de menos es lo esperado.')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint13_half{half}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
