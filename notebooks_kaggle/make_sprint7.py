r"""Genera connectome_sprint7_gen2.ipynb (2026-09-21 noche): alumna LSTM de "segunda generación".

Idea: la calidad del alumno sigue a la calidad de los maestros (exp 016: maestros 0.6817 -> alumno 0.6805;
exps 022/025: maestros 0.6843 -> alumnos 0.6823/0.6834). Ahora el mejor ensamble es Eswa+l1swa+LRswa+LA8 = 0.6855
(sprint 5b), así que destilamos de ese, con la receta que mejor anduvo (lr 5e-4, 8 épocas, SWA 5-8).

  * humo: precálculo con los 4 maestros nuevos + 1 época de 32 secuencias (corta si falla).
  * G2a: LSTM 128x2, seed 2073, lr 5e-4, 8 ép., SWA 5-8, destilada de Eswa+l1swa+LRswa+LA8 (~2 h + 12 min de precálculo).
  * tabla: combinaciones de hasta 4 entre las 9 mejores piezas del sprint 5b + G2a/G2aswa (561 combos, ~12 min).

Requiere el dataset `connectome-models` versión 6. ~2,5 h en T4. Uso (desde challenge/):
    python notebooks_kaggle/make_sprint7.py
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
          f"--output-dir /kaggle/working/results --save-valid-preds --lr-decay 0.9 --weight-decay 1e-5")
TEACHERS = {"Eswa": "gru128_distill_seed2029_swa.pt", "l1swa": "lstm128x2_l1_seed2047_swa.pt",
            "LRswa": "lstm128x2_studentLR_seed2072_swa.pt", "LA8": "lstm128x2_studentLA_best.pt"}      # 0.6855 en valid (S8)
OTHERS = {"y2": "lstm128x2_y2_seed2045_swa.pt", "X": "gru128x2_studentX_best.pt", "Zswa": "gru128x2_studentZ_swa.pt",
          "l3swa": "lstm160x2_l3_seed2049_swa.pt", "C2kswa": "lstm128x2_studentC2k_seed2071_swa.pt"}
STUDENTS = {"G2a": 2073}
CACHE = "/kaggle/tmp/soft_G2.npz"
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


all_files = list(TEACHERS.values()) + list(OTHERS.values())
cells = [
    md("# Connectome — sprint 7: alumna LSTM de segunda generación (G2a)\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 6**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
       "**Antes de lanzar: mirá cuántas horas de GPU te quedan esta semana. Si son menos de 3, esperá al jueves** "
       "(si Kaggle corta la sesión por cuota se pierde todo).\n\n"
       "| Paso | Qué hace |\n|---|---|\n| humo | precálculo con los 4 maestros nuevos + 1 época de 32 secuencias (corta si falla) |\n"
       f"| G2a | LSTM 128x2, seed {STUDENTS['G2a']}, **lr 5e-4**, 8 ép., SWA 5–8, destilada de **{' + '.join(TEACHERS)} (0.6855, el ensamble de la S8)** |\n"
       "| tabla | combinaciones de hasta 4 entre las 9 mejores piezas del sprint 5b + G2a → `results/ensemble_eval_sprint7.json` |\n"),
    code("import os, torch\n"
         "assert torch.cuda.is_available(), 'Activá GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
         f"for p in {TRAIN.split()} + ['{VALID}']:\n"
         "    assert os.path.isfile(p), f'Falta input: {p}'\n"
         f"for f in {all_files}:\n"
         f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 6'\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
    writefile("utils.py", ROOT / "utils.py"),
    writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
    writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py"),
    md("## Prueba de humo"),
    code("# ~3 min: precálculo con los 4 maestros y batch real + 1 época de 32 secuencias (LSTM, lr 5e-4). Corta si falla.\n"
         f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
         f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --dropout 0.1 --lr 5e-4 --seed 1 --model-seed 1 "
         f"--distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
]
for name, seed in STUDENTS.items():
    cells.append(md(f"## Alumna {name} — LSTM 128x2, lr 5e-4, semilla {seed}, maestros {' + '.join(TEACHERS)}"))
    cells.append(code(f"!python train_official_gpu_v1.py {COMMON} --epochs 8 --swa-start 5 --cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 "
                      f"--seed {seed} --model-seed {seed} --distill-weight 1.0 --distill-cache {CACHE} --distill-teachers {teacher_paths} "
                      f"--run-name student_{name} || echo 'FALLÓ student_{name}'\n"))
final_models = " ".join(f"{k}={MODELS}/{v}" for k, v in {**TEACHERS, **OTHERS}.items())
cells += [
    md("## Tabla (9 piezas del sprint 5b + G2a, combinaciones de hasta 4)"),
    code("import os\n"
         f"models = '{final_models}'.split()\n"
         f"for name in {list(STUDENTS)}:\n"
         "    for s in ('best', 'swa'):\n"
         "        p = f'/kaggle/working/results/student_{name}/{s}_model.pt'\n"
         "        if os.path.isfile(p): models.append(f\"{name}{'swa' if s == 'swa' else ''}={p}\")\n"
         "print(len(models), 'modelos')\n"
         f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_sprint7.json --models {{' '.join(models)}}\n"),
    code("import json, os\n"
         f"for k in {list(STUDENTS)}:\n"
         "    f = f'/kaggle/working/results/student_{k}/metrics.json'\n"
         "    if os.path.isfile(f):\n"
         "        m = json.load(open(f)); curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
         "        print(f\"{k}: mejor {m['best_valid_wp']:.4f}  SWA {m.get('swa_valid_wp', float('nan')):.4f}   curva: {curva}\")\n"
         "print('referencia: LR (lr 5e-4, maestros viejos) 0.6834 | mejor ensamble S8 0.6855')\n"
         "r = json.load(open('/kaggle/working/results/ensemble_eval_sprint7.json'))\n"
         "print(f\"\\n{'modelos':30s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra\")\n"
         "for row in r['results'][:20]:\n"
         "    print(f\"{row['models']:30s} {row['wp']:10.4f} {row['t0']:8.4f} {row['t1']:8.4f} {row['us_per_row_est']:8.1f}  {'sí' if row['fits_60min'] else 'NO'}\")\n"
         "print('\\n-- mejores que entran, por cantidad de modelos')\n"
         "for k in (1, 2, 3, 4):\n"
         "    rows = [x for x in r['results'] if x['fits_60min'] and x['models'].count('+') == k - 1][:3]\n"
         "    for x in rows: print(f\"{k}: {x['models']:30s} {x['wp']:.4f}  {x['us_per_row_est']:.1f} us\")\n"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "connectome_sprint7_gen2.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", HERE / "connectome_sprint7_gen2.ipynb")
