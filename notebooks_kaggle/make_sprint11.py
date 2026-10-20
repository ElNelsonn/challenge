r"""Genera connectome_sprint11_ema.ipynb: MEMORIA LARGA (idea 1 de la hoja de ruta).

El entrenamiento corta el gradiente cada 512 pasos (BPTT truncado), así que el modelo **no puede aprender**
a usar información de hace miles de filas aunque la secuencia tenga 20.000 y el estado la arrastre.
`--ema-features` se la da servida: para 12 columnas clave, el **desvío respecto del promedio móvil** de
100, 1.000 y 10.000 filas (36 entradas) + una **volatilidad lenta** (4). Total 40 entradas nuevas → 152.

Por qué esto no es "otro intento de features" de los que ya fallaron: los cuatro anteriores eran de memoria
corta (lags 1–50, deltas, combinaciones dentro de la misma fila), justamente lo que una red recurrente ya
construye sola. Esto ataca un límite estructural del entrenamiento.

Costo: 13 ms por secuencia en el cargador de datos (en paralelo con la GPU, no agrega tiempo de pared).
En inferencia serían ~1,2 µs/fila extra: barato, pero **hay que implementarlo en `solution.py` si gana**.

Comparación limpia: misma receta EXACTA que la alumna LR (destilada, lr 5e-4, WMSE) = **0.6834**.
Requiere `connectome-models` versión 7. Uso (desde challenge/):  python notebooks_kaggle/make_sprint11.py
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
CMD = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
       f"--output-dir /kaggle/working/results --save-valid-preds --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
       f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --epochs 8 --swa-start 5 --ema-features "
       f"--distill-weight 1.0 --distill-cache /kaggle/tmp/soft_S8.npz --distill-teachers {teacher_paths} "
       f"--seed 2111 --model-seed 2111 --run-name student_ema")


def check():
    return code("import os, torch\n"
                "assert torch.cuda.is_available(), 'Activá GPU'\n"
                "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
                "import scipy; print('scipy', scipy.__version__)\n"
                f"for p in {TRAIN.split()} + ['{VALID}']:\n"
                "    assert os.path.isfile(p), f'Falta input: {p}'\n"
                f"for f in {list(TEACHERS.values())}:\n"
                f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 7'\n"
                "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# Connectome — sprint 11: **memoria larga** (EMA)\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
       "**Qué se prueba:** el entrenamiento corta el gradiente cada 512 pasos, así que el modelo **no puede aprender** a "
       "usar información de hace miles de filas, aunque la secuencia tenga 20.000. Estas entradas se la dan servida: "
       "para 12 columnas clave, el **desvío respecto del promedio de las últimas 100, 1.000 y 10.000 filas**, más una "
       "medida de **volatilidad lenta**. Son 40 entradas nuevas (152 en total).\n\n"
       "**No es otro intento de los que ya fallaron:** los cuatro anteriores eran de memoria corta (lags 1–50, deltas, "
       "combinaciones dentro de la misma fila), justo lo que una red recurrente ya construye sola. Esto ataca un límite "
       "estructural del entrenamiento, no una comodidad.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, lr 5e-4, dropout 0.1, wd 1e-5, 8 épocas, SWA 5–8, semilla 2111 |\n"
       "| entradas | 112 + **40 de memoria larga** |\n"
       "| comparar contra | **la alumna LR = 0.6834**: misma receta exacta sin estas entradas |\n"
       "| si gana | hay que implementar las 40 cuentas en `solution.py` (una por fila, ~1,2 µs extra) |\n\n"
       "**Criterio:** ≥ 0.687 = hay información nueva y vale el trabajo del kernel. 0.683–0.684 = puerta cerrada."),
    check(),
    code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
    code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
    md("## Prueba de humo (~3 min)"),
    code("# 1 época de 32 secuencias con las entradas nuevas. Tiene que decir 'entradas=152'. Corta si falla.\n"
         f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
         f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --ema-features "
         f"--seed 1 --model-seed 1 --distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
    md("## Alumna con memoria larga"),
    code(f"!python train_official_gpu_v1.py {CMD} || echo 'FALLÓ student_ema'\n"),
    md("## Etapa 2: seguir hasta la época 20 (SWA 15–20)\n\n"
       "Retoma el mismo modelo desde su checkpoint. Comparar contra la etapa 1 aísla el efecto de entrenar el doble, "
       "ya con la memoria larga puesta. Son 12 épocas más, ~2,5 h."),
    code("!python train_official_gpu_v1.py --resume /kaggle/working/results/student_ema --resume-epochs 20 "
         f"--resume-swa-start 15 --resume-cache /kaggle/tmp/soft_S8.npz || echo 'FALLÓ la etapa 2'\n"),
    md("## Resumen"),
    code("import json, os\n"
         "d = '/kaggle/working/results/student_ema'\n"
         "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
         "if m:\n"
         "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
         "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
         "    print('curva por epoca:', curva)\n"
         "    print('referencia EXACTA: alumna LR, misma receta sin memoria larga, 8 épocas = 0.6834')\n"
         "    print(sorted(os.listdir(d)))\n"
         "else:\n"
         "    print('FALLO: revisar el error de arriba')\n"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "connectome_sprint11_ema.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok: connectome_sprint11_ema.ipynb")

# ------------------------------------------------------------------ B: entrenamiento largo, sin memoria larga
CMD_B = CMD.replace(" --ema-features", "").replace("--epochs 8 --swa-start 5", "--epochs 20 --swa-start 15") \
           .replace("--seed 2111 --model-seed 2111", "--seed 2112 --model-seed 2112") \
           .replace("student_ema", "student_long")
cells_b = [
    md("# Connectome — sprint 11B: **entrenamiento largo** (20 épocas)\n\n"
       "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~5 h.\n\n"
       "**Se puede correr a la vez que el notebook 11A.**\n\n"
       "**Qué se prueba:** todos nuestros modelos entrenaron 8 o 9 épocas y se plantaron alrededor de la 7. Nunca "
       "probamos entrenar **dos veces y media más**, con el paso de aprendizaje bajando siempre (llega a 7e-5 en la última). "
       "Si el techo de 0.684 fuera por falta de entrenamiento y no por falta de información, esto lo mueve.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, lr 5e-4, dropout 0.1, wd 1e-5, **20 épocas**, SWA 15–20, semilla 2112 |\n"
       "| entradas | las 112 de siempre (sin memoria larga: eso lo prueba el otro notebook) |\n"
       "| comparar contra | **la alumna LR = 0.6834**: misma receta con 8 épocas |\n\n"
       "**Criterio:** ≥ 0.687 = estábamos entrenando de menos. 0.683–0.685 = el techo no es por entrenamiento."),
    check(),
    code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
    code("%%writefile train_official_gpu_v1.py\n" + (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
    md("## Prueba de humo (~3 min)"),
    code("!python train_official_gpu_v1.py --train-parquet " + TRAIN + " --valid-parquet " + VALID +
         " --train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 "
         "--layers 2 --dropout 0.1 --lr 5e-4 --seed 1 --model-seed 1 --distill-teachers " + teacher_paths +
         " --distill-cache '' --run-name smoke\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
         "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
    md("## 20 épocas"),
    code(f"!python train_official_gpu_v1.py {CMD_B} || echo 'FALLÓ student_long'\n"),
    md("## Resumen"),
    code("import json, os\n"
         "d = '/kaggle/working/results/student_long'\n"
         "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
         "if m:\n"
         "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
         "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
         "    print('curva por epoca:', curva)\n"
         "    print('referencia EXACTA: alumna LR, misma receta con 8 épocas = 0.6834')\n"
         "    print(sorted(os.listdir(d)))\n"
         "else:\n"
         "    print('FALLO: revisar el error de arriba')\n"),
]
nb_b = {"cells": cells_b, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "connectome_sprint11_long.ipynb").write_text(json.dumps(nb_b, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok: connectome_sprint11_long.ipynb")
