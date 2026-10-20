r"""Genera tres notebooks (2026-09-19 noche; ampliado 2026-09-21):

  * connectome_sprint5b_lite.ipynb    — retoma la alumna LSTM LA (ép. 8 -> 12, SWA 9-12) y tabla final con TODAS las LSTM
    (l1/l2/l3, LA8, LA12, LR = lr 5e-4, C2k = chunk 2000) + las GRU. ~2 h (para la cuota que queda esta semana).
  * connectome_sprint5b_LA12_LB.ipynb — lo mismo más LB (alumna LSTM seed 2062, 12 ép.). ~4,5 h.
  * connectome_sprint6_hparams.ipynb  — plan B sobre la mejor receta (alumna LSTM): chunk 2000 vs 512, lr 5e-4 vs 1e-3. ~5 h (ya corrido).

Requieren el dataset `connectome-models` versión 6 (LA best/swa/checkpoint, l1-l3, LR, C2k). Uso (desde challenge/):
    python notebooks_kaggle/make_sprint6.py
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
TEACHERS = {"y2": "lstm128x2_y2_seed2045_swa.pt", "Eswa": "gru128_distill_seed2029_swa.pt", "Dswa": "gru192_seed2027_swa.pt",
            "X": "gru128x2_studentX_best.pt", "Zswa": "gru128x2_studentZ_swa.pt"}
LA_FILES = {"checkpoint.pt": "lstm128x2_studentLA_checkpoint.pt", "best_model.pt": "lstm128x2_studentLA_best.pt",
            "swa_model.pt": "lstm128x2_studentLA_swa.pt", "metrics.json": "lstm128x2_studentLA_metrics.json",
            "run_config.json": "lstm128x2_studentLA_run_config.json"}
LSTM_BASE = {"l1": "lstm128x2_l1_seed2047", "l2": "lstm128x2_l2_seed2048", "l3": "lstm160x2_l3_seed2049",
             "LR": "lstm128x2_studentLR_seed2072", "C2k": "lstm128x2_studentC2k_seed2071"}   # LR/C2k: alumnos del sprint 6
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())
CACHE = "/kaggle/tmp/soft_L.npz"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def check(files):
    return code("import os, torch\n"
                "assert torch.cuda.is_available(), 'Activá GPU'\n"
                "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
                f"for p in {TRAIN.split()} + ['{VALID}']:\n"
                "    assert os.path.isfile(p), f'Falta input: {p}'\n"
                f"for f in {files}:\n"
                f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 6'\n"
                "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


files = lambda: [writefile("utils.py", ROOT / "utils.py"),
                 writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
                 writefile("eval_ensemble.py", CH / "model" / "submission" / "eval_ensemble.py")]

smoke = code("# Prueba de humo (~3 min): precálculo con los 5 maestros y batch completo + 1 época de 32 secuencias (LSTM). Corta si falla.\n"
             f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
             f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --dropout 0.1 --seed 1 --model-seed 1 "
             f"--distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n")


def show(json_path):
    return code("import json\n"
                f"r = json.load(open('{json_path}'))\n"
                "print(f\"{'modelos':30s} {'GLOBAL WP':>10s} {'t0':>8s} {'t1':>8s} {'us/fila':>8s}  entra\")\n"
                "for row in r['results'][:30]:\n"
                "    print(f\"{row['models']:30s} {row['wp']:10.4f} {row['t0']:8.4f} {row['t1']:8.4f} {row['us_per_row_est']:8.1f}  {'sí' if row['fits_60min'] else 'NO'}\")\n"
                "print('\\n-- mejores que entran, por cantidad de modelos')\n"
                "for k in (1, 2, 3, 4):\n"
                "    rows = [x for x in r['results'] if x['fits_60min'] and x['models'].count('+') == k - 1][:3]\n"
                "    for x in rows: print(f\"{k}: {x['models']:30s} {x['wp']:.4f}  {x['us_per_row_est']:.1f} us\")\n")


# ------------------------------------------------------------------ A: sprint5b
v6_files = list(TEACHERS.values()) + list(LA_FILES.values()) + [f"{v}_{s}.pt" for v in LSTM_BASE.values() for s in ("best", "swa")]
copy_la = code("# Copiar el run de LA (época 8) a una carpeta escribible para retomarlo\n"
               "import shutil, os\n"
               "dst = '/kaggle/working/results/student_LA'\n"
               "os.makedirs(dst, exist_ok=True)\n"
               f"for local, remote in {json.dumps(LA_FILES)}.items():\n"
               f"    shutil.copy(f'{MODELS}/{{remote}}', f'{{dst}}/{{local}}')\n"
               "print(sorted(os.listdir(dst)))\n")
resume_la = code(f"!python train_official_gpu_v1.py --resume /kaggle/working/results/student_LA --resume-epochs 12 --resume-swa-start 9 "
                 f"--resume-cache {CACHE} || echo 'FALLÓ el resume de LA'\n")
train_lb = code(f"!python train_official_gpu_v1.py {COMMON} --epochs 12 --swa-start 9 --cell lstm --hidden 128 --layers 2 --dropout 0.1 "
                f"--seed 2062 --model-seed 2062 --distill-weight 1.0 --distill-cache {CACHE} --distill-teachers {teacher_paths} "
                f"--run-name student_LB || echo 'FALLÓ student_LB'\n")
final_a_models = " ".join([f"{k}={MODELS}/{v}" for k, v in TEACHERS.items()] +
                          [f"{k}swa={MODELS}/{v}_swa.pt" for k, v in LSTM_BASE.items()])   # solo SWA: en las 5 LSTM el SWA >= mejor época; 13 modelos -> 1.092 combos
final_a = code("import os\n"
               f"models = '{final_a_models} LA8={MODELS}/{LA_FILES['best_model.pt']}'.split()\n"
               "for name in ['LA', 'LB']:\n"
               "    for s in ('best', 'swa'):\n"
               "        p = f'/kaggle/working/results/student_{name}/{s}_model.pt'\n"
               "        if os.path.isfile(p): models.append(f\"{name}{'swa' if s == 'swa' else ''}={p}\")\n"
               "print(len(models), 'modelos')\n"
               f"!python eval_ensemble.py --valid {VALID} --n all --max-combo 4 --out /kaggle/working/results/ensemble_eval_sprint5b.json --models {{' '.join(models)}}\n")
def build_a(with_lb):
    return nb([
        md(f"# Connectome — sprint 5b{'' if with_lb else ' lite'}: LA hasta 12 épocas{', LB' if with_lb else ''}, tabla final\n\n"
           f"**Inputs:** train part1/part2, valid, **`Connectome models` versión 6**. GPU T4 x2. **Save & Run All.** ~{'4,5' if with_lb else '2'} h.\n\n"
           "| Paso | Qué hace |\n|---|---|\n| humo | precálculo en GPU con batch real (corta si falla) |\n"
           "| LA 9–12 | retoma la alumna LSTM LA desde el checkpoint de la época 8 (0.6823, seguía subiendo) hasta la 12, SWA 9–12 (~1,2 h) |\n"
           + ("| LB | alumna LSTM seed 2062, 12 épocas, SWA 9–12 (la vez anterior murió por el bug de recarga del cache, ya corregido) |\n" if with_lb else "")
           + "| final | combinaciones de hasta 4 entre maestros, l1/l2/l3, LR (lr 5e-4), C2k (chunk 2000), LA8, LA12"
           + (", LB" if with_lb else "") + " → `results/ensemble_eval_sprint5b.json` (~30 min) |\n"),
        check(v6_files), *files(),
        md("## Prueba de humo"), smoke,
        md("## LA: épocas 9–12"), copy_la, resume_la,
        *([md("## LB: 12 épocas"), train_lb] if with_lb else []),
        md("## Tabla final"), final_a, show("/kaggle/working/results/ensemble_eval_sprint5b.json"),
    ])


nb_a = build_a(True)
nb_lite = build_a(False)

# ------------------------------------------------------------------ B: sprint6 hparams
RUNS = {
    "chunk2000": ("--chunk 2000", 2071),
    "lr5e-4": ("--lr 5e-4", 2072),
}
cells_b = [
    md("# Connectome — sprint 6: hiperparámetros nunca tocados, sobre la mejor receta (alumna LSTM)\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models` (v5 o v6). GPU T4 x2. **Save & Run All.** ~5 h.\n\n"
       "Referencia: LA (chunk 512, lr 1e-3, seed 2061): 0.6823 en la ép. 8 / (ver LA-12 en el sprint 5b). Cada corrida cambia UNA cosa:\n\n"
       "| corrida | cambio | pregunta |\n|---|---|---|\n"
       "| chunk2000 | trozos de BPTT de 2.000 pasos en vez de 512 | ¿memoria más larga ayuda (sobre todo a t1)? |\n"
       "| lr5e-4 | velocidad de aprendizaje 5e-4 en vez de 1e-3 | ¿aprende más fino? |\n\n"
       "Las dos con 10 épocas, SWA 7–10, seeds distintas (ruido entre seeds ≈ ±0.001: solo cuentan diferencias > 0.002)."),
    check(list(TEACHERS.values())), *files(),
    md("## Prueba de humo"), smoke,
]
for name, (extra, seed) in RUNS.items():
    cells_b.append(md(f"## {name}: `{extra}`"))
    cells_b.append(code(f"!python train_official_gpu_v1.py {COMMON} --epochs 10 --swa-start 7 --cell lstm --hidden 128 --layers 2 --dropout 0.1 "
                        f"--seed {seed} --model-seed {seed} --distill-weight 1.0 --distill-cache {CACHE} --distill-teachers {teacher_paths} "
                        f"{extra} --run-name student_{name} || echo 'FALLÓ {name}'\n"))
cells_b.append(md("## Resumen"))
cells_b.append(code("import json, os\n"
                    "print('referencia LA (chunk 512, lr 1e-3): 0.6823 ép. 8')\n"
                    f"for k in {list(RUNS)}:\n"
                    "    f = f'/kaggle/working/results/student_{k}/metrics.json'\n"
                    "    if os.path.isfile(f):\n"
                    "        m = json.load(open(f)); curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
                    "        print(f\"{k:10s} mejor {m['best_valid_wp']:.4f}  SWA {m.get('swa_valid_wp', float('nan')):.4f}   curva: {curva}\")\n"
                    "    else:\n"
                    "        print(k, 'FALLÓ')\n"))
nb_b = nb(cells_b)

(HERE / "connectome_sprint5b_LA12_LB.ipynb").write_text(json.dumps(nb_a, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint5b_lite.ipynb").write_text(json.dumps(nb_lite, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "connectome_sprint6_hparams.ipynb").write_text(json.dumps(nb_b, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok: sprint5b, sprint5b_lite, sprint6")
print("archivos esperados en v6:", v6_files)
