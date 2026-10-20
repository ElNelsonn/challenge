r"""Genera los notebooks de Kaggle del sprint 4 (idea de Joaquín: 6 maestros diversos en 2 grupos, un alumno por grupo):

  * connectome_sprint4_base_x.ipynb  — maestros x1 GRU 128x2, x2 GRU 200x1, x3 GRU 64x3          (~4,5 h)
  * connectome_sprint4_base_y.ipynb  — maestros y1 GRU 160x2, y2 LSTM 128x2, y3 GRU 128x2 drop 0.2 (~5,5 h)
  * connectome_sprint4_distill.ipynb — elige best/SWA de cada maestro, destila X (x1-3), Y (y1-3), Z (los 6 + Eswa + Dswa),
                                       tabla final de combinaciones de hasta 4                     (~6 h)

El tercero lee los maestros del dataset `connectome-models` versión 4 (subir los .pt de los dos primeros con los nombres de
BASE_FILES antes de lanzarlo). Uso (desde challenge/):  python notebooks_kaggle/make_sprint4.py
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

# maestros base: nombre -> (grupo, args del trainer, archivo en el dataset v4 sin sufijo _best/_swa)
BASE = {
    "x1": ("x", "--cell gru --hidden 128 --layers 2 --dropout 0.1 --seed 2041 --model-seed 2041", "gru128x2_x1_seed2041"),
    "x2": ("x", "--cell gru --hidden 200 --layers 1 --seed 2042 --model-seed 2042", "gru200x1_x2_seed2042"),
    "x3": ("x", "--cell gru --hidden 64 --layers 3 --dropout 0.1 --seed 2043 --model-seed 2043", "gru64x3_x3_seed2043"),
    "y1": ("y", "--cell gru --hidden 160 --layers 2 --dropout 0.1 --seed 2044 --model-seed 2044", "gru160x2_y1_seed2044"),
    "y2": ("y", "--cell lstm --hidden 128 --layers 2 --dropout 0.1 --seed 2045 --model-seed 2045", "lstm128x2_y2_seed2045"),
    "y3": ("y", "--cell gru --hidden 128 --layers 2 --dropout 0.2 --seed 2046 --model-seed 2046", "gru128x2d2_y3_seed2046"),
}
BASE_FILES = {k: v[2] for k, v in BASE.items()}
EXISTING = {"Eswa": "gru128_distill_seed2029_swa.pt", "Dswa": "gru192_seed2027_swa.pt", "Gswa": "gru128_gwp_seed2031_swa.pt",
            "Cswa": "gru128_drop_seed2028_swa.pt", "B": "gru128_santiago_base.pt"}
STUDENTS = {  # nombre -> (maestros, seed)
    "X": (["x1", "x2", "x3"], 2051),
    "Y": (["y1", "y2", "y3"], 2052),
    "Z": (["x1", "x2", "x3", "y1", "y2", "y3", "Eswa", "Dswa"], 2053),
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def check_cell(model_files=()):
    return code(
        "import os, torch\n"
        "assert torch.cuda.is_available(), 'Activá GPU en Session options -> Accelerator'\n"
        "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
        f"for p in {TRAIN.split()} + ['{VALID}']:\n"
        "    assert os.path.isfile(p), f'Falta input: {p}'\n"
        f"for f in {list(model_files)}:\n"
        f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}} en el dataset Connectome models (¿versión correcta?)'\n"
        "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


files_cells = lambda: [writefile("utils.py", ROOT / "utils.py"),
                       writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
                       writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py")]


def base_notebook(group):
    names = [k for k, v in BASE.items() if v[0] == group]
    rows = "\n".join(f"| {k} | `{BASE[k][1]}` | `results/{k}/` |" for k in names)
    cells = [md(f"# Connectome — sprint 4, maestros base grupo **{group}**\n\n"
                "**Inputs:** `Connectome train part1`, `Connectome train part2`, `Connectome valid`. **Accelerator:** GPU T4 x2. "
                "**Save & Run All.** ~5 h.\n\nReceta común: 8 épocas, lr 1e-3 decay 0.9, wd 1e-5, SWA 5–8, valid completo. "
                "Cada modelo con semilla propia (pesos, normalizador y orden de datos).\n\n| modelo | args | salida |\n|---|---|---|\n" + rows + "\n"),
             check_cell(), *files_cells()]
    for k in names:
        cells.append(md(f"## {k}: `{BASE[k][1]}`"))
        cells.append(code(f"!python train_official_gpu_v1.py {COMMON} {BASE[k][1]} --run-name {k} || echo 'FALLÓ {k} (sigue con el siguiente)'\n"))
    cells.append(md("## Resumen"))
    cells.append(code("import json, os\n"
                      f"for k in {names}:\n"
                      "    f = f'/kaggle/working/results/{k}/metrics.json'\n"
                      "    if os.path.isfile(f):\n"
                      "        m = json.load(open(f)); print(f\"{k}: mejor {m['best_valid_wp']:.4f}  SWA {m.get('swa_valid_wp', float('nan')):.4f}\")\n"
                      "    else:\n"
                      "        print(k, 'FALLÓ')\n"))
    return nb(cells)


# ------------------------------------------------------------------ notebook 3: destilación
all_files = [f"{BASE_FILES[k]}_{s}.pt" for k in BASE for s in ("best", "swa")] + list(EXISTING.values())
pick_models = " ".join(f"{k}{s}={MODELS}/{BASE_FILES[k]}_{s}.pt" for k in BASE for s in ("best", "swa"))
pick = code(
    "# 0. Para cada maestro base elegir la versión mejor (best o SWA) según valid\n"
    f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 1 --out /kaggle/working/results/eval_base_versions.json --models {pick_models}\n"
    "import json\n"
    "r = {row['models']: row['wp'] for row in json.load(open('/kaggle/working/results/eval_base_versions.json'))['results']}\n"
    f"files = {json.dumps(BASE_FILES)}\n"
    f"existing = {json.dumps(EXISTING)}\n"
    "chosen = {}\n"
    "for k, stem in files.items():\n"
    "    s = 'swa' if r[k + 'swa'] >= r[k + 'best'] else 'best'\n"
    f"    chosen[k] = f'{MODELS}/{{stem}}_{{s}}.pt'\n"
    "    print(f\"{k}: best {r[k+'best']:.4f}  swa {r[k+'swa']:.4f}  -> {s}\")\n"
    "for k, f in existing.items():\n"
    f"    chosen[k] = f'{MODELS}/{{f}}'\n"
    f"students = {json.dumps({k: v[0] for k, v in STUDENTS.items()})}\n"
    "T = {name: ' '.join(chosen[m] for m in members) for name, members in students.items()}\n"
    "json.dump({'chosen': chosen, 'teachers': T}, open('/kaggle/working/results/teachers_sprint4.json', 'w'), indent=2)\n"
    "for name in T: print(name, '<-', students[name])\n")

smoke = code(
    "# Prueba de humo (~3 min): precálculo con los 8 maestros de Z (los más pesados) con batch completo + 1 época de 32 seqs.\n"
    f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
    f"--output-dir /kaggle/tmp/smoke --epochs 1 --hidden 128 --dropout 0.1 --seed 1 --model-seed 1 "
    f"--distill-teachers {{T['Z']}} --distill-cache '' --run-name smoke\n"
    "import os, json\n"
    "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
    "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n")

student_cells = []
for name, (members, seed) in STUDENTS.items():
    student_cells.append(md(f"## Alumno {name} ← {' + '.join(members)} (GRU 128x2, semilla {seed})"))
    student_cells.append(code(
        f"!python train_official_gpu_v1.py {COMMON} --cell gru --hidden 128 --layers 2 --dropout 0.1 --seed {seed} --model-seed {seed} "
        f"--distill-weight 1.0 --distill-cache /kaggle/tmp/soft_{name}.npz --distill-teachers {{T['{name}']}} --run-name student_{name} "
        f"|| echo 'FALLÓ student_{name} (sigue con el siguiente)'\n"))

final = code(
    "import os, json\n"
    "chosen = json.load(open('/kaggle/working/results/teachers_sprint4.json'))['chosen']\n"
    "models = [f'{k}={p}' for k, p in chosen.items() if k not in ('Cswa', 'B')]\n"
    f"for name in {list(STUDENTS)}:\n"
    "    for s in ('best', 'swa'):\n"
    "        p = f'/kaggle/working/results/student_{name}/{s}_model.pt'\n"
    "        if os.path.isfile(p): models.append(f\"{name}{'swa' if s == 'swa' else ''}={p}\")\n"
    "print(len(models), 'modelos')\n"
    f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_sprint4.json --models {{' '.join(models)}}\n")
show = code(
    "import json\n"
    "r = json.load(open('/kaggle/working/results/ensemble_eval_sprint4.json'))\n"
    "print(f\"{'modelos':30s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra\")\n"
    "for row in r['results'][:40]:\n"
    "    print(f\"{row['models']:30s} {row['wp']:10.4f} {row['t0']:8.4f} {row['t1']:8.4f} {row['us_per_row_est']:8.1f}  {'sí' if row['fits_60min'] else 'NO'}\")\n"
    "print('\\n-- mejores que entran, por cantidad de modelos')\n"
    "for k in (1, 2, 3, 4):\n"
    "    rows = [x for x in r['results'] if x['fits_60min'] and x['models'].count('+') == k - 1][:3]\n"
    "    for x in rows: print(f\"{k}: {x['models']:30s} {x['wp']:.4f}  {x['us_per_row_est']:.1f} us\")\n")

distill_nb = nb([
    md("# Connectome — sprint 4: alumnos X, Y, Z\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 4** (con los 6 maestros base best+SWA). GPU T4 x2. "
       "**Save & Run All.** ~6 h.\n\n| Paso | Qué hace |\n|---|---|\n"
       "| 0 | Evalúa best vs SWA de cada maestro base y se queda con el mejor |\n"
       "| humo | Precálculo de soft targets en GPU con batch real (corta si falla) |\n"
       "| X | GRU 128x2 destilada de x1+x2+x3 |\n| Y | GRU 128x2 destilada de y1+y2+y3 |\n"
       "| Z | GRU 128x2 destilada de los 6 + Eswa + Dswa |\n"
       "| final | Combinaciones de hasta 4 entre X, Y, Z (best/SWA), Eswa, Dswa, Gswa y los 6 base → `results/ensemble_eval_sprint4.json` |\n"),
    check_cell(all_files), *files_cells(),
    md("## 0. Versión de cada maestro y listas de maestros"), pick,
    md("## Prueba de humo"), smoke,
    *student_cells,
    md("## Tabla final"), final, show,
])

# ------------------------------------------------------------------ notebook 4: más LSTM (la LSTM 128x2 dio 0.6803 sola, la mejor base)
LSTM = {
    "l1": ("l", "--cell lstm --hidden 128 --layers 2 --dropout 0.1 --seed 2047 --model-seed 2047", "lstm128x2_l1_seed2047"),
    "l2": ("l", "--cell lstm --hidden 128 --layers 2 --dropout 0.1 --seed 2048 --model-seed 2048", "lstm128x2_l2_seed2048"),
    "l3": ("l", "--cell lstm --hidden 160 --layers 2 --dropout 0.1 --seed 2049 --model-seed 2049", "lstm160x2_l3_seed2049"),
}


def lstm_notebook():
    names = list(LSTM)
    rows = "\n".join(f"| {k} | `{LSTM[k][1]}` | `results/{k}/` |" for k in names)
    cells = [md("# Connectome — sprint 4, más LSTM\n\n"
                "**Inputs:** `Connectome train part1`, `Connectome train part2`, `Connectome valid`. **Accelerator:** GPU T4 x2. "
                "**Save & Run All.** ~5,5 h.\n\nMotivo: la LSTM 128x2 (y2) dio **0.6803 sola**, +0.004 sobre cualquier GRU normal e igual que la "
                "destilada. Tres LSTM más con otras semillas/tamaños: maestros fuertes y, con un kernel LSTM, candidatas directas al ensamble.\n\n"
                "| modelo | args | salida |\n|---|---|---|\n" + rows + "\n"),
             check_cell(), *files_cells()]
    for k in names:
        cells.append(md(f"## {k}: `{LSTM[k][1]}`"))
        cells.append(code(f"!python train_official_gpu_v1.py {COMMON} {LSTM[k][1]} --run-name {k} || echo 'FALLÓ {k} (sigue con el siguiente)'\n"))
    cells.append(md("## Resumen"))
    cells.append(code("import json, os\n"
                      f"for k in {names}:\n"
                      "    f = f'/kaggle/working/results/{k}/metrics.json'\n"
                      "    if os.path.isfile(f):\n"
                      "        m = json.load(open(f)); print(f\"{k}: mejor {m['best_valid_wp']:.4f}  SWA {m.get('swa_valid_wp', float('nan')):.4f}\")\n"
                      "    else:\n"
                      "        print(k, 'FALLÓ')\n"))
    return nb(cells)


(HERE / "connectome_sprint4_lstm.ipynb").write_text(json.dumps(lstm_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint4_base_x.ipynb").write_text(json.dumps(base_notebook("x"), ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint4_base_y.ipynb").write_text(json.dumps(base_notebook("y"), ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint4_distill.ipynb").write_text(json.dumps(distill_nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok: base_x, base_y, distill")
print("archivos esperados en connectome-models v4:", [f"{v}_{s}.pt" for v in BASE_FILES.values() for s in ("best", "swa")])
