r"""Genera connectome_sprint5_lstm_students.ipynb: dos alumnos LSTM 128x2 destilados del mejor ensamble actual
(y2 LSTM + Eswa + Dswa + X + Zswa; combinaciones de 0.6843 en valid), con soft targets precalculados una vez y
compartidos entre los dos alumnos. Al final, tabla de combinaciones de hasta 4 con los alumnos nuevos.

Requiere el dataset `connectome-models` versión 5 (con gru128x2_student{X,Y,Z}_{best,swa}.pt). ~4 h en T4.
Uso (desde challenge/):  python notebooks_kaggle/make_sprint5.py
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
COMMON = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
          f"--output-dir /kaggle/working/results --save-valid-preds --epochs 8 --lr-decay 0.9 --weight-decay 1e-5 --swa-start 5")
TEACHERS = {"y2": "lstm128x2_y2_seed2045_swa.pt", "Eswa": "gru128_distill_seed2029_swa.pt", "Dswa": "gru192_seed2027_swa.pt",
            "X": "gru128x2_studentX_best.pt", "Zswa": "gru128x2_studentZ_swa.pt"}
OTHERS = {"Gswa": "gru128_gwp_seed2031_swa.pt", "Yswa": "gru128x2_studentY_swa.pt"}
STUDENTS = {"LA": 2061, "LB": 2062}   # LSTM 128x2, semillas


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())
cells = [
    md("# Connectome — sprint 5: alumnos LSTM\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 5**. GPU T4 x2. **Save & Run All.** ~4 h.\n\n"
       "Motivo: la LSTM 128x2 sola rinde como la GRU destilada (0.680) y aporta la diversidad que sube los ensambles. "
       "Un alumno LSTM que copie al mejor ensamble (y2 + Eswa + Dswa + X + Zswa, 0.6843) puede ser el mejor modelo individual.\n\n"
       "| Paso | Qué hace |\n|---|---|\n| humo | Precálculo de soft targets en GPU con batch real (corta si falla) |\n"
       f"| LA, LB | LSTM 128x2 (dropout 0.1, wd 1e-5, SWA 5–8) destiladas de {' + '.join(TEACHERS)}; semillas {list(STUDENTS.values())}. "
       "Los soft targets se calculan una vez y se comparten |\n"
       "| final | Combinaciones de hasta 4 entre maestros, Gswa, Yswa y los alumnos nuevos → `results/ensemble_eval_sprint5.json` |\n"),
    code("import os, torch\n"
         "assert torch.cuda.is_available(), 'Activá GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
         f"for p in {TRAIN.split()} + ['{VALID}']:\n"
         "    assert os.path.isfile(p), f'Falta input: {p}'\n"
         f"for f in {list(TEACHERS.values()) + list(OTHERS.values())}:\n"
         f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 5'\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
    writefile("utils.py", ROOT / "utils.py"),
    writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
    writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py"),
    md("## Prueba de humo"),
    code("# ~3 min: precálculo con los 5 maestros y batch completo + 1 época de 32 secuencias (LSTM). Corta si falla.\n"
         f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
         f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --dropout 0.1 --seed 1 --model-seed 1 "
         f"--distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
]
for name, seed in STUDENTS.items():
    cells.append(md(f"## Alumno {name} — LSTM 128x2, semilla {seed}"))
    cells.append(code(f"!python train_official_gpu_v1.py {COMMON} --cell lstm --hidden 128 --layers 2 --dropout 0.1 --seed {seed} --model-seed {seed} "
                      f"--distill-weight 1.0 --distill-cache /kaggle/tmp/soft_L.npz --distill-teachers {teacher_paths} --run-name student_{name} "
                      f"|| echo 'FALLÓ student_{name} (sigue con el siguiente)'\n"))
final_models = " ".join(f"{k}={MODELS}/{v}" for k, v in {**TEACHERS, **OTHERS}.items())
cells += [
    md("## Tabla final (maestros + Gswa + Yswa + alumnos LSTM, combinaciones de hasta 4)"),
    code("import os\n"
         f"models = '{final_models}'.split()\n"
         f"for name in {list(STUDENTS)}:\n"
         "    for s in ('best', 'swa'):\n"
         "        p = f'/kaggle/working/results/student_{name}/{s}_model.pt'\n"
         "        if os.path.isfile(p): models.append(f\"{name}{'swa' if s == 'swa' else ''}={p}\")\n"
         "print(len(models), 'modelos')\n"
         f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_sprint5.json --models {{' '.join(models)}}\n"),
    code("import json\n"
         "r = json.load(open('/kaggle/working/results/ensemble_eval_sprint5.json'))\n"
         "print(f\"{'modelos':30s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra\")\n"
         "for row in r['results'][:30]:\n"
         "    print(f\"{row['models']:30s} {row['wp']:10.4f} {row['t0']:8.4f} {row['t1']:8.4f} {row['us_per_row_est']:8.1f}  {'sí' if row['fits_60min'] else 'NO'}\")\n"
         "print('\\n-- mejores que entran, por cantidad de modelos')\n"
         "for k in (1, 2, 3, 4):\n"
         "    rows = [x for x in r['results'] if x['fits_60min'] and x['models'].count('+') == k - 1][:3]\n"
         "    for x in rows: print(f\"{k}: {x['models']:30s} {x['wp']:.4f}  {x['us_per_row_est']:.1f} us\")\n"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "connectome_sprint5_lstm_students.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", HERE / "connectome_sprint5_lstm_students.ipynb")
