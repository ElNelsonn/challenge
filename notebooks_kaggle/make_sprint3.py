r"""Genera los notebooks de Kaggle del sprint 3 a partir de los archivos actuales del repo:

  * connectome_eval_combos.ipynb   — solo evaluación: todas las combinaciones de 1..4 modelos (10 min GPU).
  * connectome_sprint3_alumnos.ipynb — eval + elección automática de maestros + 5 alumnos destilados (~9 h GPU).

Uso (desde challenge/):  python notebooks_kaggle/make_sprint3.py
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
# candidatos (nombre -> archivo del dataset connectome-models v3)
CANDIDATES = {
    "A": "gru128_exp009_seed2026.pt",
    "B": "gru128_santiago_base.pt",
    "Cswa": "gru128_drop_seed2028_swa.pt",
    "Dswa": "gru192_seed2027_swa.pt",
    "E": "gru128_distill_seed2029_best.pt",
    "Eswa": "gru128_distill_seed2029_swa.pt",
    "F": "gru128_cosine_seed2030_best.pt",
    "Gswa": "gru128_gwp_seed2031_swa.pt",
}
SEEDS = [2032, 2033, 2034, 2035, 2036]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


check = code(
    "import os, torch\n"
    "assert torch.cuda.is_available(), 'Activá GPU en Session options -> Accelerator'\n"
    "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
    f"for p in {TRAIN.split()} + ['{VALID}']:\n"
    "    assert os.path.isfile(p), f'Falta input: {p}'\n"
    f"for f in {list(CANDIDATES.values())}:\n"
    f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 3'\n"
    "print('Inputs OK')\n")

models_arg = " ".join(f"{k}={MODELS}/{v}" for k, v in CANDIDATES.items())
eval_cmd = (f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_combos.json "
            f"--models {models_arg}")
show = code(
    "import json\n"
    "r = json.load(open('/kaggle/working/results/ensemble_eval_combos.json'))\n"
    "print(f\"{'modelos':26s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra\")\n"
    "for row in r['results'][:40]:\n"
    "    print(f\"{row['models']:26s} {row['wp']:10.4f} {row['t0']:8.4f} {row['t1']:8.4f} {row['us_per_row_est']:8.1f}  {'sí' if row['fits_60min'] else 'NO'}\")\n")

# ------------------------------------------------------------------ notebook 1: solo eval
eval_nb = nb([
    md("# Connectome — evaluación de combinaciones (1 a 4 modelos)\n\n"
       "**Inputs:** `Connectome valid` y **`Connectome models` (versión 3)**. **Accelerator:** GPU T4 x2. **Save & Run All.** ~10 min.\n\n"
       "Evalúa las 8 GRUs disponibles y todas sus combinaciones de hasta 4 (promedio simple) en las 1.873 secuencias de valid, "
       "con la estimación de µs/fila del kernel v2 (int8). Salida: `results/ensemble_eval_combos.json`."),
    code("import os, torch\nprint('GPU:', torch.cuda.get_device_name(0))\n"
         f"for f in {list(CANDIDATES.values())}:\n"
         f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 3'\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
    writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py"),
    code(eval_cmd + "\n"),
    show,
])

# ------------------------------------------------------------------ notebook 2: eval + 5 alumnos
pick = code(
    "# Maestros = la mejor combinación de hasta 4 modelos según valid (con los soft targets precalculados, el costo por\n"
    "# época no depende de cuántos maestros haya; solo el precálculo inicial, ~2 min por maestro).\n"
    "import json\n"
    "r = json.load(open('/kaggle/working/results/ensemble_eval_combos.json'))\n"
    f"files = {json.dumps(CANDIDATES)}\n"
    "best = max(r['results'], key=lambda row: row['wp'])\n"
    "teachers = [f\"" + MODELS + "/{files[n]}\" for n in best['models'].split('+')]\n"
    "print(f\"Maestros: {best['models']}  (valid {best['wp']:.4f})\")\n"
    "open('/kaggle/working/teachers.txt', 'w').write(' '.join(teachers))\n"
    "TEACHERS = ' '.join(teachers)\n")

# Maestros fijos (elegidos por connectome_eval_combos el 2026-09-18: Dswa+Eswa+F+Gswa = 0.6829, mejor combinación de <=4).
# Con None, el notebook corre la evaluación y los elige solo (10 min más).
TEACHERS_FIXED = ["Dswa", "Eswa", "F", "Gswa"]
if TEACHERS_FIXED:
    pick = code(
        f"TEACHERS = ' '.join('{MODELS}/' + f for f in {[CANDIDATES[t] for t in TEACHERS_FIXED]})\n"
        f"print('Maestros fijos: {'+'.join(TEACHERS_FIXED)}')\n"
        "open('/kaggle/working/teachers.txt', 'w').write(TEACHERS)\n")
    step0 = [md("## 0. Maestros (fijos: " + "+".join(TEACHERS_FIXED) + " = 0.6829 en valid según `connectome_eval_combos`)"), pick]
else:
    step0 = [md("## 0. Combinaciones de los modelos existentes → elección de maestros"),
             code("import os\nos.makedirs('/kaggle/working/results', exist_ok=True)\n" + eval_cmd + "\n"), show, pick]

smoke = code(
    "# Prueba de humo (~3 min): precálculo de maestros en GPU con un batch completo de 32 + 1 época sobre 32 secuencias.\n"
    "# Si esto falla, el notebook se corta acá (no lleva '|| echo') y no se pierden horas.\n"
    f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
    f"--output-dir /kaggle/tmp/smoke --epochs 1 --hidden 128 --dropout 0.1 --seed 1 --model-seed 1 "
    f"--distill-teachers $TEACHERS --distill-cache '' --run-name smoke\n"
    "import os, json\n"
    "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba antes de seguir'\n"
    "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n")

train_cells = []
for i, seed in enumerate(SEEDS, 1):
    cmd = (f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
           f"--output-dir /kaggle/working/results --save-valid-preds --epochs 8 --lr-decay 0.9 --hidden 128 --dropout 0.1 --weight-decay 1e-5 "
           f"--seed {seed} --model-seed {seed} --swa-start 5 --distill-weight 1.0 --distill-cache /kaggle/tmp/soft_targets.npz "
           f"--distill-teachers $TEACHERS --run-name student_{seed} || echo 'FALLÓ student_{seed} (sigue con el siguiente)'")
    train_cells.append(md(f"## Alumno {i}/{len(SEEDS)} — semilla {seed}"))
    train_cells.append(code(cmd + "\n"))

final_models = " ".join(f"{k}={MODELS}/{v}" for k, v in CANDIDATES.items() if k in ("B", "Cswa", "Dswa", "Eswa", "Gswa"))
final_models += " " + " ".join(f"S{s}swa=/kaggle/working/results/student_{s}/swa_model.pt" for s in SEEDS)
final_eval = code(
    "import os\n"
    f"models = '{final_models}'.split()\n"
    "models = [m for m in models if os.path.isfile(m.split('=', 1)[1])]   # por si algún alumno falló\n"
    f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_sprint3.json --models {{' '.join(models)}}\n")
show3 = code(show["source"].replace("ensemble_eval_combos.json", "ensemble_eval_sprint3.json"))

sprint_nb = nb([
    md("# Connectome — sprint 3: 5 alumnos destilados\n\n"
       "**Inputs** (Add Input → Your Work → Datasets): `Connectome train part1`, `Connectome train part2`, `Connectome valid`, "
       "**`Connectome models` (versión 3)**. **Accelerator:** GPU T4 x2. **Save Version → Save & Run All.** Duración estimada: **~9 h** "
       "(prueba de humo 3 min + precálculo de maestros ~10 min + 5 alumnos × ~1,8 h + eval final 10 min).\n\n"
       "| Paso | Qué hace | Salida |\n|---|---|---|\n"
       "| 0 | **Maestros** = Dswa+Eswa+F+Gswa (mejor combinación de ≤4 según `connectome_eval_combos`, 0.6829) | `teachers.txt` |\n"
       "| humo | Precálculo + 1 época sobre 32 secuencias en GPU; si falla, el notebook se corta acá | — |\n"
       "| 1–5 | **Alumnos**: GRU 128x2 (dropout 0.1, wd 1e-5, SWA 5–8) que imitan a los maestros, semillas 2032–2036. "
       "Los soft targets se calculan una vez (`/kaggle/tmp/soft_targets.npz`, por trozos de 2.000 pasos) y se reutilizan | `results/student_<seed>/` |\n"
       "| 6 | Tabla final de combinaciones de 1–4 con maestros + alumnos | `results/ensemble_eval_sprint3.json` |\n"),
    check,
    writefile("utils.py", ROOT / "utils.py"),
    writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
    writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py"),
    *step0,
    md("## Prueba de humo"),
    smoke,
    *train_cells,
    md("## 6. Tabla final (maestros + alumnos, combinaciones de hasta 4)"),
    final_eval,
    show3,
])

(HERE / "connectome_eval_combos.ipynb").write_text(json.dumps(eval_nb, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint3_alumnos.ipynb").write_text(json.dumps(sprint_nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", HERE / "connectome_eval_combos.ipynb", HERE / "connectome_sprint3_alumnos.ipynb")
