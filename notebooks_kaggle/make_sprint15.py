r"""Genera los dos notebooks del sprint 15 (2026-10-07):

  A) connectome_sprint15_mask.ipynb — entrenar PESANDO las filas que la métrica va a puntuar (~2 h).
     La métrica solo mira el 9% de las filas (is_scored) y esa columna no viene en train. Ajustamos un modelo
     que la adivina desde las 112 features: **AUC 0.794**, y el decil más probable tiene 34.9% de filas
     puntuadas contra 0.5% del decil más bajo. Hasta ahora entrenábamos parejo sobre las 20.000 filas de cada
     secuencia, gastando la mayor parte del esfuerzo en filas que no cuentan.
     Comparación exacta contra la alumna LR = **0.6834** (misma receta sin pesar).

  B) connectome_sprint15_tcn20.ipynb — la convolucional, pero bien entrenada (~4 h).
     En el sprint 14 dio 0.6739 contra 0.6808 de la LSTM, pero **su curva seguía subiendo en la época 10** y
     corrió **sin destilación**. Esta corrida le da las dos cosas que le faltaban: 20 épocas y destilación.
     Es el desempate justo: si con eso sigue abajo, la arquitectura queda cerrada con evidencia.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint15.py
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
COMUN = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
         f"--output-dir /kaggle/working/results --save-valid-preds --lr-decay 0.9 --weight-decay 1e-5 --dropout 0.1")
COEF = json.loads((CH / "model" / "official_gpu" / "mask_coef.json").read_text(encoding="utf-8"))


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def check(con_modelos):
    extra = (f"for f in {list(TEACHERS.values())}:\n"
             f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models'\n") if con_modelos else ""
    return code("import os, torch\n"
                "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU T4 x2)'\n"
                "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
                f"for p in {TRAIN.split()} + ['{VALID}']:\n"
                "    assert os.path.isfile(p), f'Falta input: {p}'\n" + extra +
                "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


def archivos():
    return [code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
            code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8"))]


def resumen(nombre, ref):
    return code("import json, os\n"
                f"d = '/kaggle/working/results/{nombre}'\n"
                "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
                "if m:\n"
                "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
                "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
                "    print('curva por epoca:', curva)\n"
                f"    print('{ref}')\n"
                "    print(sorted(os.listdir(d)))\n"
                "else:\n"
                "    print('FALLO: revisar el error de arriba')\n")


# ---------------------------------------------------------------- A: pesado por la máscara
nb_a = {"cells": [
    md("# Connectome — sprint 15A: **entrenar donde la métrica mira**\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models`. GPU T4 x2. **Save & Run All.** ~2 h.\n\n"
       "**El problema:** la métrica solo puntúa el **9 %** de las filas (las marcadas con `is_scored`), vienen en "
       "rachas cortas, y esa columna **no existe en los datos de entrenamiento**. Veníamos entrenando parejo sobre "
       "las 20.000 filas de cada secuencia, o sea gastando la mayor parte del esfuerzo en filas que no cuentan.\n\n"
       "**La solución:** ajustamos un modelo que adivina qué filas van a ser puntuadas, usando solo las 112 "
       "features. Da **AUC 0.794**, y la separación es fuerte: el decil con mayor probabilidad tiene **34,9 %** de "
       "filas puntuadas contra **0,5 %** del decil más bajo. Con eso pesamos cada fila al entrenar.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, lr 5e-4, 10 épocas, SWA 7–10, destilada, semilla 2151 |\n"
       "| cambio | peso de cada fila = \\|y\\| × (probabilidad de ser puntuada / promedio) |\n"
       "| comparar contra | **la alumna LR = 0.6834**: misma receta exacta sin pesar |\n\n"
       "Es lo último barato que quedaba sin probar, y es el único cambio que alinea lo que optimizamos con lo que "
       "nos puntúan."),
    check(True), *archivos(),
    md("## Coeficientes del modelo de la máscara"),
    code("import json\n"
         f"coef = {json.dumps(COEF)}\n"
         "json.dump(coef, open('mask_coef.json', 'w'))\n"
         "print('AUC del modelo de la máscara:', coef['auc'], '| tasa base de filas puntuadas:', round(coef['tasa_base'], 3))\n"),
    md("## Prueba de humo (~3 min)"),
    code(f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 "
         f"--valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --layers 2 "
         f"--dropout 0.1 --lr 5e-4 --mask-coef mask_coef.json --mask-power 1.0 --seed 1 --model-seed 1 "
         f"--distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
    md("## Alumna pesada hacia las filas puntuadas"),
    code(f"!python train_official_gpu_v1.py {COMUN} --cell lstm --hidden 128 --layers 2 --lr 5e-4 --epochs 10 "
         f"--swa-start 7 --mask-coef mask_coef.json --mask-power 1.0 --distill-weight 1.0 "
         f"--distill-cache /kaggle/tmp/soft_S8.npz --distill-teachers {teacher_paths} --seed 2151 --model-seed 2151 "
         f"--run-name student_mask || echo 'FALLÓ student_mask'\n"),
    md("## Resumen"), resumen("student_mask", "referencia EXACTA: alumna LR, misma receta sin pesar = 0.6834"),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

# ---------------------------------------------------------------- B: la convolucional, bien entrenada
nb_b = {"cells": [
    md("# Connectome — sprint 15B: la convolucional, **bien entrenada**\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models`. GPU T4 x2. **Save & Run All.** ~4 h.\n\n"
       "**Por qué repetirla.** En el sprint 14 dio 0.6739 contra 0.6808 de la LSTM, pero esa comparación era "
       "injusta por dos motivos: **su curva seguía subiendo en la última época** (0.6684 → 0.6722, no había "
       "convergido) y corrió **sin destilación**, que en la LSTM vale +0.003.\n\n"
       "Esta corrida le da las dos cosas: **20 épocas** y **destilación**. Es el desempate justo.\n\n"
       "| | |\n|---|---|\n| modelo | convolucional 128 canales × 5 capas, campo receptivo 683 filas |\n"
       "| entrenamiento | lr 5e-4, 20 épocas, SWA 15–20, destilada de los 4 maestros, semilla 2152 |\n"
       "| comparar contra | la alumna LR destilada = **0.6834**, y contra el 0.6739 de la versión sin convergir |\n\n"
       "**Qué decide:** si llega a ~0.680 y mantiene su correlación baja con la familia recurrente (0.962, la más "
       "baja que medimos), entra al ensamble y se abre la rama de arquitecturas. Si se queda en 0.675, la "
       "convolucional queda cerrada **con evidencia** y no se vuelve."),
    check(True), *archivos(),
    md("## Prueba de humo (~4 min)"),
    code(f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 "
         f"--valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --arch tcn --tcn-channels 128 --tcn-layers 5 "
         f"--dropout 0.1 --lr 5e-4 --seed 1 --model-seed 1 --distill-teachers {teacher_paths} "
         f"--distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
    md("## Convolucional destilada, 20 épocas"),
    code(f"!python train_official_gpu_v1.py {COMUN} --arch tcn --tcn-channels 128 --tcn-layers 5 --tcn-kernel 3 "
         f"--lr 5e-4 --epochs 20 --swa-start 15 --distill-weight 1.0 --distill-cache /kaggle/tmp/soft_S8.npz "
         f"--distill-teachers {teacher_paths} --seed 2152 --model-seed 2152 --run-name tcn_full "
         f"|| echo 'FALLÓ tcn_full'\n"),
    md("## Resumen"), resumen("tcn_full", "referencias: LSTM destilada 0.6834 | convolucional sin convergir 0.6739"),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

for nombre, nb in (("connectome_sprint15_mask.ipynb", nb_a), ("connectome_sprint15_tcn20.ipynb", nb_b)):
    (HERE / nombre).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", nombre)
