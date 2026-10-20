r"""Genera los dos notebooks del sprint 8 (2026-09-26): ALUMNAS GRANDES.

Hipótesis (la única palanca que queda con evidencia a favor): el alumno destilado no está limitado por los
maestros sino por su propio tamaño. Todas las alumnas hasta ahora fueron LSTM 128x2 y ninguna llegó al nivel
de su ensamble maestro (maestros 0.6843 -> alumnas 0.6823 y 0.6834). Las pruebas de tamaño anteriores (160x2,
192x2, 200x1, 64x3) se hicieron SIN destilación y con lr 1e-3, o sea que nunca se probó "alumna grande".

Maestros: E + l1 + LR + LA (el ensamble de la S8, 0.6855 en valid).
Receta fija (la mejor conocida): lr 5e-4, dropout 0.1, wd 1e-5, batch 32, chunk 512, SWA sobre las últimas 4 épocas.

  * connectome_sprint8_big192.ipynb — LSTM 192x2, 9 épocas, SWA 6-9   (~4 h)
  * connectome_sprint8_big256.ipynb — LSTM 256x2, 7 épocas, SWA 5-7   (~5,5 h)

Se lanzan EN PARALELO (Kaggle permite 2 sesiones de GPU). Requieren `connectome-models` versión 6.
Guardan las predicciones de valid (best y SWA) para armar las tablas de ensamble en la laptop con
`model/submission/combine_local.py`, sin gastar más GPU.

Tiempo de inferencia (calibrado con S7 = 35m05s reales): 1x192 = 24 min, 2x192 = 43 min, 1x256 = 38 min,
1x256 + 1x128 = 47 min. O sea: las tres opciones entran.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint8.py
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
COMMON = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
          f"--output-dir /kaggle/working/results --save-valid-preds --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
          f"--cell lstm --layers 2 --dropout 0.1 --distill-weight 1.0 --distill-teachers " + teacher_paths)
RUNS = {"big192": dict(hidden=192, epochs=9, swa=6, seed=2081, horas="~4 h", cache="/kaggle/tmp/soft_S8.npz"),
        "big256": dict(hidden=256, epochs=7, swa=5, seed=2082, horas="~5,5 h", cache="/kaggle/tmp/soft_S8.npz")}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


for name, r in RUNS.items():
    cells = [
        md(f"# Connectome — sprint 8: alumna LSTM **{r['hidden']}x2** (grande)\n\n"
           f"**Inputs:** train part1/part2, valid, **`Connectome models` versión 6**. GPU T4 x2. **Save & Run All.** {r['horas']}.\n\n"
           "**Se puede correr a la vez que el otro notebook del sprint 8** (Kaggle permite 2 sesiones de GPU).\n\n"
           "**Qué se prueba:** hasta ahora todas las alumnas destiladas fueron 128x2 y ninguna alcanzó a su ensamble "
           "maestro (maestros 0.6843 → alumnas 0.6823 / 0.6834). La hipótesis es que el cuello es el **tamaño de la "
           "alumna**, no los maestros: copiar un promedio de 4 redes en una sola red chica no entra. Las pruebas de "
           "tamaño anteriores fueron sin destilar y con lr 1e-3, así que esto nunca se probó.\n\n"
           f"| | |\n|---|---|\n| alumna | LSTM **{r['hidden']}x2**, lr 5e-4, dropout 0.1, wd 1e-5, {r['epochs']} épocas, "
           f"SWA {r['swa']}–{r['epochs']}, semilla {r['seed']} |\n"
           f"| maestros | {' + '.join(TEACHERS)} (el ensamble de la S8, **0.6855** en valid) |\n"
           f"| referencia a batir | alumna 128x2 = 0.6834 sola; ensamble actual 0.6855 |\n"
           f"| inferencia | 1×{r['hidden']}x2 = {24 if r['hidden']==192 else 38} min de los 60 → entra sola, y también acompañada |\n\n"
           "Al final guarda las predicciones de valid (mejor época y SWA): las tablas de ensamble se arman después "
           "en la laptop con `combine_local.py`, sin gastar GPU."),
        code("import os, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             f"for p in {TRAIN.split()} + ['{VALID}']:\n"
             "    assert os.path.isfile(p), f'Falta input: {p}'\n"
             f"for f in {list(TEACHERS.values())}:\n"
             f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 6'\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
        writefile("utils.py", ROOT / "utils.py"),
        writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
        md("## Prueba de humo (~4 min)"),
        code("# precálculo con los 4 maestros + 1 época de 32 secuencias con la red grande. Corta si falla.\n"
             f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
             f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden {r['hidden']} --layers 2 --dropout 0.1 --lr 5e-4 "
             f"--seed 1 --model-seed 1 --distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## Alumna {name} — LSTM {r['hidden']}x2, {r['epochs']} épocas"),
        code(f"!python train_official_gpu_v1.py {COMMON} --hidden {r['hidden']} --epochs {r['epochs']} --swa-start {r['swa']} "
             f"--seed {r['seed']} --model-seed {r['seed']} --distill-cache {r['cache']} --run-name student_{name} "
             f"|| echo 'FALLÓ student_{name}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/student_{name}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva por epoca:', curva)\n"
             "    print('referencias: alumna 128x2 sola 0.6834 | mejor ensamble 0.6855')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint8_{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
