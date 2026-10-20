r"""Genera los notebooks del sprint 16 (2026-10-08).

La lección del sprint 15: **destilar y ser distinto están en conflicto.** La convolucional sin destilar tenía
correlación 0.962 con la familia recurrente (lo más distinto que logramos) pero solo 0.6739 porque no había
convergido. Al destilarla y darle 20 épocas subió a 0.6782, pero su correlación subió a 0.975: la destilación
la empujó a imitar a las LSTM, que es justo lo que no queríamos. Terminó sin aportar al ensamble (+0.0000).

Falta el cuadrante que nunca probamos: **convergida y sin destilar**.

  * connectome_sprint16_tcn40.ipynb  — convolucional, 40 épocas, SIN destilación (~5 h). Su curva subía hasta la
    época 20 (0.6776 / 0.6778 / 0.6782), así que con 40 debería llegar a 0.678-0.681 manteniendo correlación ~0.96.
  * connectome_sprint16_tcn_big.ipynb — convolucional de 192 canales, 40 épocas, sin destilar (~7 h). El
    presupuesto de inferencia aguanta hasta 256 canales; en la LSTM el tamaño no ayudaba, pero en otra
    arquitectura eso hay que medirlo, no asumirlo.

Criterio: sirve si llega a **0.680 manteniendo correlación por debajo de 0.965**. Con eso el ensamble se mueve
aunque el modelo solo empate.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint16.py
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
        f"--output-dir /kaggle/working/results --save-valid-preds --lr 5e-4 --lr-decay 0.95 --weight-decay 1e-5 "
        f"--arch tcn --tcn-layers 5 --tcn-kernel 3 --dropout 0.1 --epochs 40 --swa-start 33")
RUNS = {"tcn40": (128, 2161, "~5 h"), "tcn_big": (192, 2162, "~7 h")}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for name, (canales, seed, horas) in RUNS.items():
    macs = 112 * canales + 5 * 3 * canales ** 2 + 2 * canales
    cells = [
        md(f"# Connectome — sprint 16: convolucional de **{canales} canales**, 40 épocas, sin destilar\n\n"
           f"**Inputs:** train part1/part2, valid. GPU T4 x2. **Save & Run All.** {horas}.\n\n"
           "**La lección del sprint anterior:** destilar y ser distinto están en conflicto. La convolucional sin "
           "destilar tenía correlación **0.962** con la familia recurrente (lo más distinto que logramos en todo el "
           "proyecto) pero solo 0.6739, porque no había convergido. Al destilarla subió a 0.6782 **pero su "
           "correlación subió a 0.975**: la destilación la empujó a imitar a las LSTM, justo lo que no queríamos. "
           "Resultado: no aportó nada al ensamble.\n\n"
           "Falta el cuadrante que nunca probamos: **convergida y sin destilar**. Su curva seguía subiendo en la "
           "época 20 (0.6776 → 0.6778 → 0.6782), así que con 40 épocas debería llegar a 0.678–0.681.\n\n"
           f"| | |\n|---|---|\n| modelo | convolucional **{canales} canales** × 5 capas, campo receptivo 683 filas |\n"
           f"| entrenamiento | lr 5e-4 con decaimiento 0.95, **40 épocas**, SWA 33–40, semilla {seed} |\n"
           "| destilación | **no** |\n"
           f"| costo en inferencia | {macs:,} operaciones por fila (la LSTM 128x2 usa 254.208) |\n\n"
           "**Criterio:** sirve si llega a **0.680 manteniendo la correlación por debajo de 0.965**. Con eso el "
           "ensamble se mueve aunque el modelo solo empate. Si llega a 0.680 pero correlacionando 0.975, es otra "
           "LSTM disfrazada y la arquitectura queda cerrada."),
        code("import os, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU T4 x2)'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             f"for p in {TRAIN.split()} + ['{VALID}']:\n"
             "    assert os.path.isfile(p), f'Falta input: {p}'\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
        md("## Prueba de humo (~3 min)"),
        code(f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 "
             f"--valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --arch tcn --tcn-channels {canales} "
             f"--tcn-layers 5 --dropout 0.1 --lr 5e-4 --seed 1 --model-seed 1 --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## Convolucional {canales} canales, 40 épocas"),
        code(f"!python train_official_gpu_v1.py {BASE} --tcn-channels {canales} --seed {seed} --model-seed {seed} "
             f"--run-name {name} || echo 'FALLÓ {name}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/{name}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('ultimas 6 epocas:', [round(v, 4) for v in h[-6:]])\n"
             "    print('subio en las ultimas 5:', round(h[-1] - h[-6], 4), '(si sigue subiendo, le faltan epocas)')\n"
             "    print('referencias: convolucional destilada 20 ep 0.6782 | sin destilar 10 ep 0.6739 | LSTM 0.6834')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint16_{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, f"({macs:,} ops/fila)")
