r"""Genera los 4 notebooks del sprint 24 (2026-10-21): un ensamble DIVERSO de verdad.

EL DIAGNOSTICO, MEDIDO HOY SOBRE TODO VALID:

    destilados (los 4 del ensamble)   corr entre si 0.9854   promediar vale +0.0021
    sin destilar (sprint 4)           corr entre si 0.9635   promediar vale +0.0051

Promediar vale 2,4 veces mas cuando los modelos no comparten maestros. Nuestro ensamble gana +0.0021 sobre el
mejor modelo solo; en una competencia normal promediar 4 modelos vale entre +0.005 y +0.015.

LA CAUSA: nunca variamos NADA entre los modelos del ensamble salvo la semilla. Los cuatro vieron los MISMOS
datos, fueron destilados de los MISMOS 4 maestros y usan la MISMA receta.

QUE CAMBIA ACA. Las tres palancas estandar de diversidad, las tres sin usar hasta hoy:

  1. DATOS DISTINTOS (bagging): cada modelo ve un 70 % de las secuencias, distinto para cada uno. Comparten
     el 70 % de a pares y entre los cuatro cubren el 99 % de train.
  2. MAESTROS DISTINTOS: cada alumno se destila de un PAR distinto de los 4 maestros, en vez de los 4 juntos.
     Conserva el +0.003 que da destilar sin que todos converjan al mismo objetivo.
  3. RECETAS DISTINTAS: tasas de aprendizaje y dropout distintos, y uno de los cuatro es GRU en vez de LSTM.

POR QUE IMPORTA AHORA. El leaderboard es densisimo: de #400 (donde estamos) a #100 hay 0.0077. Con la brecha
medida valid -> leaderboard de 0.0323, +0.003 en valid son ~116 puestos y +0.005 son ~200.

Si la correlacion baja de 0.985 a ~0.95 manteniendo el puntaje individual cerca de 0.681, el ensamble se va a
~0.690 -> leaderboard ~0.658 -> puesto ~235.

Cada corrida usa 70 % de los datos y 12 epocas: ~1,7 h. Kaggle deja 2 sesiones de GPU a la vez.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint24.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

T_E = "gru128_distill_seed2029_swa.pt"
T_L1 = "lstm128x2_l1_seed2047_swa.pt"
T_LR = "lstm128x2_studentLR_seed2072_swa.pt"
T_LA = "lstm128x2_studentLA_best.pt"

# cada alumno: (celda, hidden, lr, dropout, maestros, semilla)
CORRIDAS = {
    "div_a": ("lstm", 128, "5e-4", 0.10, [T_E, T_L1], 2301),
    "div_b": ("lstm", 128, "4e-4", 0.15, [T_LR, T_LA], 2302),
    "div_c": ("lstm", 128, "6e-4", 0.10, [T_E, T_LA], 2303),
    "div_d": ("gru", 128, "6e-4", 0.10, [T_L1, T_LR], 2304),
}
TODOS = [T_E, T_L1, T_LR, T_LA]
N_TRAIN, FRAC = 10607, 7424


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def buscador(maestros):
    return ("import os, time, glob, torch\n"
            "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
            "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
            "\n"
            "def buscar(nombre, espera=600):\n"
            "    \"\"\"Busca el archivo por nombre debajo de /kaggle/input: los datasets grandes a veces tardan\n"
            "    en montarse o aparecen en otra ruta.\"\"\"\n"
            "    t0 = time.time()\n"
            "    while True:\n"
            "        hits = sorted(glob.glob(f'/kaggle/input/**/{nombre}', recursive=True))\n"
            "        if hits:\n"
            "            # Kaggle monta el MISMO dataset en dos rutas y una es ~70x mas lenta para leer.\n"
            "            # Medido: /kaggle/input/datasets/<usuario>/... da 0.06 s por secuencia y la otra 4.5 s.\n"
            "            rapidas = [h for h in hits if '/datasets/' in h]\n"
            "            elegida = (rapidas or hits)[0]\n"
            "            print('   ', nombre, '->', elegida, '(rapida)' if rapidas else '(UNICA: puede ser lenta)')\n"
            "            return elegida\n"
            "        if time.time() - t0 > espera:\n"
            "            print('NO aparecio:', nombre, '| esto es lo que hay montado:')\n"
            "            for d in sorted(glob.glob('/kaggle/input/*')) + sorted(glob.glob('/kaggle/input/*/*')):\n"
            "                print('   ', d)\n"
            "            raise FileNotFoundError(nombre)\n"
            "        time.sleep(15)\n"
            "\n"
            "TRAIN = ' '.join(buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet'))\n"
            "VALID = buscar('valid.parquet')\n"
            f"TEACHERS = ' '.join(buscar(n) for n in {maestros})\n"
            f"print('maestros de este alumno:', {[m.split('_')[0] + '/' + m.split('_')[1] for m in maestros]})\n"
            "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


for nombre, (cell, hid, lr, drop, maestros, seed) in CORRIDAS.items():
    cortos = ", ".join(m.replace("_swa.pt", "").replace("_best.pt", "") for m in maestros)
    otros = ", ".join(m.replace("_swa.pt", "").replace("_best.pt", "") for m in TODOS if m not in maestros)
    cells = [
        md(f"# Connectome — sprint 24 ({nombre}): ensamble **diverso de verdad**\n\n"
           "**Inputs:** train part1/part2, valid, `Connectome models`. GPU. **Save & Run All.** ~1,7 h.\n\n"
           "---\n\n"
           "## El diagnóstico\n\n"
           "Medido hoy sobre todo valid:\n\n"
           "| familia | correlación entre ellos | lo que gana promediar 4 |\n|---|---|---|\n"
           "| los 4 de nuestro ensamble (destilados de los mismos maestros) | **0.9854** | **+0.0021** |\n"
           "| 4 modelos sin destilar | **0.9635** | **+0.0051** |\n\n"
           "**Promediar vale 2,4 veces más cuando los modelos no comparten maestros.** En una competencia "
           "normal promediar 4 modelos vale entre +0.005 y +0.015; a nosotros nos vale +0.002.\n\n"
           "**La causa:** nunca variamos **nada** entre los modelos del ensamble salvo la semilla. Los cuatro "
           "vieron los mismos datos, se destilaron de los mismos 4 maestros y usan la misma receta.\n\n"
           "## Qué cambia en esta corrida\n\n"
           "Las tres palancas estándar de diversidad, las tres sin usar hasta hoy:\n\n"
           f"| | este alumno (`{nombre}`) |\n|---|---|\n"
           f"| **datos** | un **70 %** de las secuencias ({FRAC:,} de {N_TRAIN:,}), elegido con la semilla {seed}. Comparte el 70 % con cada uno de los otros tres |\n"
           f"| **maestros** | **{cortos}** — y NO {otros} |\n"
           f"| **receta** | {cell.upper()} {hid}x2, lr {lr}, dropout {drop} |\n\n"
           "## Por qué importa ahora\n\n"
           "El leaderboard es densísimo: de **#400** (donde estamos) a **#100** hay **0.0077**. Con la brecha "
           "medida valid → leaderboard de 0.0323:\n\n"
           "| mejora en valid | puesto |\n|---|---|\n"
           "| +0.003 | ~294 |\n| +0.005 | ~214 |\n| +0.008 | ~118 |\n\n"
           "Si la correlación baja de 0.985 a ~0.95 manteniendo el puntaje individual cerca de 0.681, el "
           "ensamble se va a ~0.690 → leaderboard ~0.658 → **puesto ~235**.\n\n"
           "**Qué mirar al final:** el puntaje individual de este alumno va a ser **más bajo** que el de "
           "nuestros modelos actuales (ve 70 % de los datos y solo 2 maestros). Eso es esperado y no es malo. "
           "Lo que decide es la correlación entre los cuatro, que se mide en la laptop cuando estén los cuatro."),

        code(buscador(maestros)),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" +
             (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),

        md("## Prueba de humo (~3 min)"),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell {cell} "
             f"--hidden {hid} --layers 2 --dropout {drop} --lr {lr} --seed 1 --model-seed 1 "
             f"--distill-teachers $TEACHERS --distill-cache '' --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLO'\n"
             "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
             f"assert len(m['config']['distill_teachers']) == 2, 'esperaba 2 maestros, no los 4'\n"
             f"assert m['config']['cell'] == '{cell}', 'celda equivocada'\n"
             "print('Prueba de humo OK | maestros:', len(m['config']['distill_teachers']),\n"
             "      '| celda:', m['config']['cell'], '| valid WP:', round(m['best_valid_wp'], 4))\n"),

        md(f"## El alumno {nombre}: 70 % de los datos, 2 maestros, receta propia"),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups {FRAC} --valid-groups all --output-dir /kaggle/working/results "
             f"--save-valid-preds --cell {cell} --hidden {hid} --layers 2 --dropout {drop} --lr {lr} "
             f"--lr-decay 0.9 --weight-decay 1e-5 --epochs 12 --swa-start 9 --distill-weight 1.0 "
             f"--distill-teachers $TEACHERS --distill-cache /kaggle/tmp/soft_{nombre}.npz "
             f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"),

        md("## Resumen"),
        code("import json, os\n"
            f"d = '/kaggle/working/results/{nombre}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva:', [round(v, 4) for v in h])\n"
             "    print('secuencias de entrenamiento:', len(m['config']['train_ids']),\n"
             "          '| maestros:', len(m['config']['distill_teachers']))\n"
             "    print()\n"
             "    print('OJO: este numero va a ser MAS BAJO que nuestros modelos actuales (0.683).')\n"
             "    print('Ve 70% de los datos y 2 maestros en vez de 4. Eso es lo buscado.')\n"
             "    print('Lo que decide es la correlacion entre los 4, que se mide en la laptop.')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint24_{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ok: {out.name}  ({cell} {hid}, lr {lr}, maestros {cortos}, semilla {seed})")
