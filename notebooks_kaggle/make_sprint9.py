r"""Genera los dos notebooks del sprint 9 (2026-09-27), después de que el sprint 8 refutara la hipótesis del tamaño.

  * connectome_sprint9_cross.ipynb  — alumna LSTM 128x2 con 20 FEATURES CRUZADAS nuevas (~2,5 h).
    Es el experimento importante: el único eje de entrada nunca probado. Los intentos anteriores de features
    fueron lags y deltas EN EL TIEMPO (la red ya los aprende). Estas son NO LINEALES dentro de la misma fila
    (desequilibrios de volumen como cocientes, desvíos y máximos entre niveles, productos): la primera capa de
    la red es una combinación lineal de las 112 entradas, así que sumas y restas no aportan nada, pero cocientes
    y máximos sí. Comparación limpia: misma receta exacta que la alumna LR (0.6834), solo cambian las entradas.

  * connectome_sprint9_r256.ipynb   — retoma la alumna LSTM 256x2 de la época 7 a la 12, SWA 9-12 (~2,5 h).
    Su curva seguía subiendo (0.6834 / 0.6838 en las dos últimas) y es nuestro mejor modelo individual (SWA 0.6843).

Los dos en paralelo. Requieren `connectome-models` versión 7. Uso (desde challenge/):
    python notebooks_kaggle/make_sprint9.py
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
B256 = {"checkpoint.pt": "lstm256x2_big256_seed2082_checkpoint.pt", "best_model.pt": "lstm256x2_big256_seed2082_best.pt",
        "swa_model.pt": "lstm256x2_big256_seed2082_swa.pt", "metrics.json": "lstm256x2_big256_seed2082_metrics.json",
        "run_config.json": "lstm256x2_big256_seed2082_run_config.json"}
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())
CACHE = "/kaggle/tmp/soft_S8.npz"
COMMON = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
          f"--output-dir /kaggle/working/results --save-valid-preds --lr-decay 0.9 --weight-decay 1e-5 "
          f"--cell lstm --layers 2 --dropout 0.1 --distill-weight 1.0 --distill-teachers " + teacher_paths)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


def check(files, ver=7):
    return code("import os, torch\n"
                "assert torch.cuda.is_available(), 'Activá GPU'\n"
                "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
                f"for p in {TRAIN.split()} + ['{VALID}']:\n"
                "    assert os.path.isfile(p), f'Falta input: {p}'\n"
                f"for f in {files}:\n"
                f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN {ver}'\n"
                "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


def files():
    return [writefile("utils.py", ROOT / "utils.py"),
            writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py")]


def resumen(name, ref):
    return code("import json, os\n"
                f"d = '/kaggle/working/results/{name}'\n"
                "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
                "if m:\n"
                "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
                "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
                "    print('curva por epoca:', curva)\n"
                f"    print('{ref}')\n"
                "    print(sorted(os.listdir(d)))\n"
                "else:\n"
                "    print('FALLO: revisar el error de arriba')\n")


# ------------------------------------------------------------------ A: features cruzadas
nb_a = {"cells": [
    md("# Connectome — sprint 9A: alumna con **features cruzadas**\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
       "**Qué se prueba (esto es lo importante del sprint):** darle a la red 20 entradas nuevas calculadas "
       "**dentro de cada fila, combinando columnas**: desequilibrios de volumen (compra vs venta, como cociente), "
       "desvíos y máximos entre los 11 niveles del libro, concentración del volumen, y comparaciones entre los dos "
       "instrumentos.\n\n"
       "Por qué puede funcionar cuando los intentos anteriores de features fracasaron: aquellos eran **lags y deltas en "
       "el tiempo**, que una red recurrente ya aprende sola. Y ojo: sumas y restas entre columnas **tampoco** sirven, "
       "porque la primera capa de la red ya es una suma pesada de las 112 entradas. Las 20 de acá son todas **no "
       "lineales** (cocientes, desvíos, máximos, productos): eso una red con 112 entradas correlacionadas aprende mal.\n\n"
       "| | |\n|---|---|\n| alumna | LSTM 128x2, lr 5e-4, 8 épocas, SWA 5–8, semilla 2091, **`--cross-features`** |\n"
       f"| maestros | {' + '.join(TEACHERS)} (los mismos de siempre) |\n"
       "| comparación | **LR = 0.6834**: misma receta exacta, sin las features. Cualquier diferencia son las entradas nuevas |\n"
       "| qué esperar | si da ≥ 0.687 hay información nueva de verdad y vale implementarlas en el kernel; si da 0.683–0.684, puerta cerrada |\n"),
    check(list(TEACHERS.values())), *files(),
    md("## Prueba de humo (~4 min)"),
    code("# precálculo con los 4 maestros + 1 época de 32 secuencias CON las features nuevas (132 entradas). Corta si falla.\n"
         f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
         f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --cross-features "
         f"--seed 1 --model-seed 1 --distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
    md("## Alumna con features cruzadas (132 entradas)"),
    code(f"!python train_official_gpu_v1.py {COMMON} --hidden 128 --epochs 8 --swa-start 5 --lr 5e-4 --cross-features "
         f"--seed 2091 --model-seed 2091 --distill-cache {CACHE} --run-name student_cross || echo 'FALLÓ student_cross'\n"),
    md("## Resumen"),
    resumen("student_cross", "comparar contra LR (misma receta sin features cruzadas) = 0.6834 | mejor ensamble 0.6855"),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

# ------------------------------------------------------------------ B: retomar la 256
nb_b = {"cells": [
    md("# Connectome — sprint 9B: retomar la alumna 256x2 (épocas 8–12)\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
       "La alumna LSTM 256x2 del sprint 8 terminó la época 7 **todavía subiendo** (0.6819 / 0.6834 / 0.6838 en las "
       "últimas tres) y con SWA 0.6843 es nuestro mejor modelo individual. Acá se retoma desde su checkpoint hasta la "
       "época 12, con SWA sobre las épocas 9–12.\n\n"
       "Es la corrida barata y segura del sprint; la importante es la de features cruzadas."),
    check(list(TEACHERS.values()) + list(B256.values())), *files(),
    md("## Copiar el run al disco de trabajo"),
    code("import shutil, os\n"
         "dst = '/kaggle/working/results/student_big256'\n"
         "os.makedirs(dst, exist_ok=True)\n"
         f"for local, remote in {json.dumps(B256)}.items():\n"
         f"    shutil.copy(f'{MODELS}/{{remote}}', f'{{dst}}/{{local}}')\n"
         "print(sorted(os.listdir(dst)))\n"),
    md("## Épocas 8–12 (SWA 9–12)"),
    code("!python train_official_gpu_v1.py --resume /kaggle/working/results/student_big256 --resume-epochs 12 "
         f"--resume-swa-start 9 --resume-cache {CACHE} || echo 'FALLÓ el resume'\n"),
    md("## Resumen"),
    resumen("student_big256", "referencia: la misma red en la época 7 daba 0.6838 (SWA 5-7: 0.6843)"),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

for name, nb in (("connectome_sprint9_cross.ipynb", nb_a), ("connectome_sprint9_r256.ipynb", nb_b)):
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", name)
