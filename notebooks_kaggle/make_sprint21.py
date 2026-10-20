r"""Genera los notebooks del sprint 21 (2026-10-19): el modelo final, entrenado con train Y valid.

POR QUÉ AHORA. Todas las demás ramas están cerradas con evidencia:

  arquitecturas (convolucional)        4 corridas  -> +0.000
  más redes al ensamble                 r = 0.985  -> +0.000 (hace falta igualar al ensamble entero)
  árboles LightGBM                 r = 0.9271     -> -0.0014
  árboles XGBoost                  r = 0.9279     -> -0.0015 (y correlaciona 0.993 con LightGBM)
  corrector del error del ensamble      r = -0.14  -> +0.0003 (el error es impredecible)
  features de entrada (centinela, memoria larga x2, cruzadas)  -> +0.000
  reparametrizar el target                        -> -0.0031

Queda una sola cosa que no probamos y que tiene un mecanismo real detrás: **más datos**. Valid son 1.873
secuencias contra las 10.607 de train, o sea **+17,7 %**.

CÓMO SE HACE SIN QUEDARSE CIEGO. Si valid entra al entrenamiento perdemos la única forma de darnos cuenta de
que una corrida salió rota. Por eso se dejan afuera **las mismas 200 secuencias** que venimos usando para
medir los árboles, y se sigue evaluando ahí. Se pierde el 11 % de los datos nuevos y queda un termómetro
honesto. Ese número **no sirve para elegir nada** (el modelo final usa épocas fijas y SWA): sirve para saber
que la corrida anduvo.

QUÉ ESPERAR. Entre +0.002 y +0.004. La relación entre datos y puntaje es plana en este problema: a los árboles
multiplicarles los datos por 3,3 les dio +0.0137, así que un +17,7 % da bastante menos. **No es la solución, es
lo último que queda con fundamento.**

Y OJO CON ESTO: una vez entrenado con valid, **valid deja de poder compararse con los modelos viejos**. El
único juez real pasa a ser el leaderboard, que es justamente lo que veníamos necesitando: llevamos cuatro
submissions donde valid subía +0.0020 y el leaderboard se movía -0.0002.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint21.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
MODELS = "/kaggle/input/datasets/joaquinfiorina/connectome-models"
TEACHERS = ["gru128_distill_seed2029_swa.pt", "lstm128x2_l1_seed2047_swa.pt",
            "lstm128x2_studentLR_seed2072_swa.pt", "lstm128x2_studentLA_best.pt"]
N_TRAIN, N_VALID = 10607, 1873

# las 200 secuencias de valid que quedan FUERA del entrenamiento: las mismas que usa el entrenador para
# evaluar, o sea select_groups(1873, 200, seed+1) con seed=2026
EVAL = sorted(np.random.default_rng(2027).choice(N_VALID, 200, replace=False).tolist())
EXCLUIR = [N_TRAIN + g for g in EVAL]
SEEDS = {"final_a": 2211, "final_b": 2212, "final_c": 2213, "final_d": 2214}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for nombre, seed in SEEDS.items():
    cells = [
        md(f"# Connectome — sprint 21 ({nombre}): el modelo final, con **train + valid**\n\n"
           "**Inputs:** train part1/part2, valid, `Connectome models`. GPU. **Save & Run All.** ~3 h.\n\n"
           "---\n\n"
           "## Por qué esto y no otra cosa\n\n"
           "Todas las demás ramas están cerradas **con evidencia**:\n\n"
           "| lo que probamos | correlación con el ensamble | aporte |\n|---|---|---|\n"
           "| arquitectura convolucional (4 corridas) | 0.957–0.978 | +0.0000 |\n"
           "| otra red más al ensamble | 0.985 | +0.0000 |\n"
           "| árboles LightGBM | 0.9271 | −0.0014 |\n"
           "| árboles XGBoost | 0.9279 | −0.0015 |\n"
           "| corrector del error del ensamble | **−0.14** | +0.0003 |\n"
           "| features de entrada (centinela, memoria larga ×2) | — | +0.0000 |\n"
           "| reparametrizar el target | — | −0.0031 |\n\n"
           "Queda **una sola** cosa sin probar con un mecanismo real detrás: **más datos**. Valid son 1.873 "
           "secuencias contra las 10.607 de train, o sea **+17,7 %**.\n\n"
           "## Cómo se hace sin quedarse ciego\n\n"
           "Si valid entra entero al entrenamiento perdemos la única forma de saber si una corrida salió rota. "
           "Por eso se dejan afuera **las mismas 200 secuencias** que venimos usando para medir los árboles, y "
           "se sigue evaluando ahí.\n\n"
           "| | |\n|---|---|\n"
           f"| entrenamiento | 10.607 de train + **1.673 de valid** = {N_TRAIN + N_VALID - 200:,} secuencias |\n"
           "| evaluación | las 200 de valid que quedaron afuera |\n"
           f"| receta | **idéntica** a la alumna LR (0.6834): LSTM 128x2 destilada, lr 5e-4, 10 épocas, SWA 7–10, semilla {seed} |\n"
           "| cómo se elige el modelo | **no se elige**: épocas fijas y SWA. El número de valid es un termómetro, no un selector |\n\n"
           "## Qué esperar\n\n"
           "**Entre +0.002 y +0.004.** La relación entre datos y puntaje es plana acá: a los árboles "
           "multiplicarles los datos por 3,3 les dio +0.0137, así que un +17,7 % da bastante menos. No es la "
           "solución, es lo último que queda con fundamento.\n\n"
           "Y una advertencia importante: una vez entrenado con valid, **el número de valid deja de ser "
           "comparable con los modelos viejos**. El único juez pasa a ser el leaderboard — que es justo lo que "
           "necesitábamos, porque llevamos cuatro submissions donde valid subió +0.0020 y el leaderboard se "
           "movió −0.0002."),

        code("import os, time, glob, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             "\n"
             "def buscar(nombre, espera=600):\n"
             "    \"\"\"Busca el archivo por nombre debajo de /kaggle/input: los datasets grandes a veces tardan\n"
             "    en montarse o aparecen en otra ruta.\"\"\"\n"
             "    t0 = time.time()\n"
             "    while True:\n"
             "        hits = glob.glob(f'/kaggle/input/**/{nombre}', recursive=True)\n"
             "        if hits:\n"
             "            return hits[0]\n"
             "        if time.time() - t0 > espera:\n"
             "            print('NO aparecio:', nombre, '| esto es lo que hay montado:')\n"
             "            for d in sorted(glob.glob('/kaggle/input/*')) + sorted(glob.glob('/kaggle/input/*/*')):\n"
             "                print('   ', d)\n"
             "            raise FileNotFoundError(nombre)\n"
             "        time.sleep(15)\n"
             "\n"
             "P1, P2 = buscar('train_part1.parquet'), buscar('train_part2.parquet')\n"
             "VALID = buscar('valid.parquet')\n"
             "TRAIN = f'{P1} {P2} {VALID}'   # valid va ULTIMO: sus grupos quedan en [10607, 12480)\n"
             f"TEACHERS = ' '.join(buscar(n) for n in {TEACHERS})\n"
             "print('entrenamiento:', TRAIN)\nprint('evaluacion:', VALID)\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),

        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" +
             (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),

        md("## Las 200 secuencias que quedan afuera\n\n"
           "Son exactamente las mismas que el entrenador va a usar para evaluar, así que ninguna fila de "
           "evaluación se ve durante el entrenamiento."),
        code("import numpy as np\n"
             f"EVAL = sorted(np.random.default_rng(2027).choice({N_VALID}, 200, replace=False).tolist())\n"
             f"EXCLUIR = ' '.join(str({N_TRAIN} + g) for g in EVAL)\n"
             "print('200 secuencias de valid afuera del entrenamiento; primeras 5:', EVAL[:5])\n"
             f"print('entrenan {N_TRAIN} + {N_VALID} - 200 =', {N_TRAIN} + {N_VALID} - 200, 'secuencias')\n"),

        md("## Prueba de humo (~5 min)\n\n"
           "Verifica lo único delicado: que las 200 secuencias de evaluación **no** estén entre las de "
           "entrenamiento. Lo comprueba comparando las dos listas que el entrenador deja guardadas."),
        code("!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             "--train-groups all --valid-groups 200 --exclude-train-groups $EXCLUIR "
             "--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 32 --layers 1 --dropout 0.1 "
             "--lr 5e-4 --batch 8 --seed 2026 --model-seed 1 --run-name smoke 2>&1 | tail -20\n"
             "import json, os\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLO'\n"
             "c = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['config']\n"
             "tr, va = set(c['train_ids']), set(c['valid_ids'])\n"
             f"glob_va = {{{N_TRAIN} + g for g in va}}\n"
             "print('entrenamiento:', len(tr), 'secuencias | evaluacion:', len(va))\n"
             "assert not (tr & glob_va), f'SE FILTRARON {len(tr & glob_va)} secuencias de evaluacion al entrenamiento'\n"
             f"assert len(tr) == {N_TRAIN + N_VALID - 200}, f'esperaba {N_TRAIN + N_VALID - 200} secuencias, hay {{len(tr)}}'\n"
             "print('OK: ninguna de las 200 de evaluacion entra al entrenamiento')\n"),

        md(f"## El modelo final (semilla {seed})"),
        code("!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             "--train-groups all --valid-groups 200 --exclude-train-groups $EXCLUIR "
             "--output-dir /kaggle/working/results --save-valid-preds --cell lstm --hidden 128 --layers 2 "
             "--dropout 0.1 --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 --epochs 10 --swa-start 7 "
             "--distill-weight 1.0 --distill-teachers $TEACHERS --distill-cache /kaggle/tmp/soft_S21.npz "
             f"--seed 2026 --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"),

        md("## Resumen"),
        code("import json, os\n"
            f"d = '/kaggle/working/results/{nombre}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva:', [round(v, 4) for v in h])\n"
             "    print('secuencias de entrenamiento:', len(m['config']['train_ids']))\n"
             "    print()\n"
             "    print('REFERENCIA: la misma receta con solo train, medida en TODO valid, dio 0.6834.')\n"
             "    print('Este numero esta medido en 200 secuencias, asi que no es comparable directo:')\n"
             "    print('el ensamble de 4 redes da 0.6862 en todo valid y 0.6755 en estas 200.')\n"
             "    print('Sirve para ver que la corrida no salio rota. El juez real es el leaderboard.')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint21_{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, f"(semilla {seed})")
