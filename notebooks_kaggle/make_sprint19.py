r"""Genera el notebook del sprint 19 (2026-10-16): memoria larga elegida por medición, no por intuición.

DE DÓNDE SALE. Midiendo la correlación entre el ERROR del ensamble y los cambios de cada columna a varios
retrasos (ruido ±0.003), quedan correlaciones reales que las redes todavía no usan:

    i1_v20 (retraso 200) -0.0343   <- la más alta, y es del instrumento correlacionado
    i0_p9  (retraso  50) +0.0269
    i0_p15 (retraso  50) -0.0236
    i0_p2  (retraso  20) +0.0233
    i0_v4  (retraso 100) -0.0228
    i1_v4  (retraso  50) -0.0224

Ninguna de esas columnas tiene hoy features de memoria larga: la lista actual (12 columnas) se eligió por
intuición en septiembre. `--ema-plus` agrega las 11 que aparecen en esa medición: 73 features de memoria larga
en vez de 40, 185 entradas en vez de 152.

EL TAMAÑO, DICHO DE FRENTE. Dos avisos que hay que leer antes de entusiasmarse:

  1. **El intento anterior de esto dio exactamente cero.** `student_ema` (las 40 features actuales, 20 épocas)
     dio 0.6838, y `student_long` (misma receta, 20 épocas, SIN esas features) dio 0.6838 también.
  2. **La cuenta pone el techo bajo.** La correlación más alta (0.034) contra un error que se lleva el 88 % de
     la varianza vale como +0.0007 en la métrica. Todas juntas, descontando lo que se repite: +0.001 a +0.002.

Y lo que da +0.002 en valid viene sin trasladarse al leaderboard: las últimas cuatro submissions subieron
+0.0020 en valid y movieron −0.0002 en el leaderboard.

**Lo único que cambia respecto del intento anterior es que ahora las columnas se eligen midiendo contra el
error del ensamble, en vez de a ojo.** Si con eso tampoco se mueve, la rama de features queda cerrada.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint19.py
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
TEACHERS = ["gru128_distill_seed2029_swa.pt", "lstm128x2_l1_seed2047_swa.pt",
            "lstm128x2_studentLR_seed2072_swa.pt", "lstm128x2_studentLA_best.pt"]
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS)
RECETA = ("--train-parquet $TRAIN --valid-parquet $VALID --train-groups all --valid-groups all "
          "--output-dir /kaggle/working/results --save-valid-preds --cell lstm --hidden 128 --layers 2 "
          "--dropout 0.1 --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 --epochs 20 --swa-start 15 "
          "--distill-weight 1.0 --distill-teachers $TEACHERS")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# Connectome — sprint 19: **memoria larga elegida midiendo**\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models`. GPU T4 x2. **Save & Run All.** ~4 h.\n\n"
       "---\n\n"
       "## De dónde sale\n\n"
       "Se midió la correlación entre el **error del ensamble** (lo que las redes todavía no explican) y los "
       "cambios de cada una de las 112 columnas a retrasos de 1 a 200 filas. El ruido a ese tamaño de muestra "
       "es ±0.003, así que lo que pasa de 0.01 es real:\n\n"
       "| columna | retraso | correlación con el error |\n|---|---|---|\n"
       "| **i1_v20** | 200 | **−0.0343** |\n| i0_p9 | 50 | +0.0269 |\n| i0_p15 | 50 | −0.0236 |\n"
       "| i0_p2 | 20 | +0.0233 |\n| i0_v4 | 100 | −0.0228 |\n| i1_v4 | 50 | −0.0224 |\n\n"
       "**Ninguna de esas columnas tiene hoy features de memoria larga.** La lista actual son 12 columnas "
       "elegidas por intuición en septiembre. `--ema-plus` agrega las 11 que aparecen midiendo: **73 features "
       "de memoria larga en vez de 40**, 185 entradas en vez de 152.\n\n"
       "Una LSTM en teoría puede recordar 200 filas, pero en la práctica se olvida mucho antes; darle el "
       "promedio móvil ya calculado le evita tener que acordarse.\n\n"
       "## El tamaño, dicho de frente\n\n"
       "Dos avisos, antes de entusiasmarse:\n\n"
       "1. **El intento anterior de esta misma idea dio exactamente cero.** `student_ema` (las 40 features "
       "actuales, 20 épocas) dio **0.6838**, y `student_long` (misma receta, 20 épocas, **sin** esas features) "
       "dio **0.6838** también.\n"
       "2. **La cuenta pone el techo bajo.** La correlación más alta (0.034) contra un error que se lleva el "
       "88 % de la varianza vale como **+0.0007**. Todas juntas, descontando lo que se repite entre ellas: "
       "**+0.001 a +0.002**.\n\n"
       "Y lo que da +0.002 en valid viene sin trasladarse: las últimas cuatro submissions subieron +0.0020 en "
       "valid y movieron **−0.0002** en el leaderboard.\n\n"
       "**Lo único que cambia respecto del intento anterior es que ahora las columnas se eligen midiendo "
       "contra el error del ensamble, en vez de a ojo.** Si con eso tampoco se mueve, la rama de features de "
       "entrada queda cerrada con evidencia.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, destilada de los 4 maestros, semilla 2191 |\n"
       "| receta | **idéntica** a `student_ema` y `student_long`: lr 5e-4, 20 épocas, SWA 15–20 |\n"
       "| único cambio | `--ema-features --ema-plus` |\n"
       "| comparar contra | **0.6838**, que es lo que dieron las dos |"),

    code("import os, time, glob, torch\n"
         "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
         "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
         "\n"
         "def buscar(nombre, espera=600):\n"
         "    \"\"\"Busca el archivo por nombre debajo de /kaggle/input. Los datasets grandes a veces tardan\n"
         "    en montarse o aparecen en otra ruta, asi que se espera y se reintenta.\"\"\"\n"
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
         "TRAIN = ' '.join(buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet'))\n"
         "VALID = buscar('valid.parquet')\n"
         f"TEACHERS = ' '.join(buscar(n) for n in {TEACHERS})\n"
         "print('train:', TRAIN)\nprint('valid:', VALID)\nprint('maestros:', TEACHERS.count('.pt'), 'archivos')\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),

    code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
    code("%%writefile train_official_gpu_v1.py\n" +
         (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),

    md("## Prueba de humo (~3 min)\n\n"
       "Recorre el mismo camino que la corrida larga y verifica que las 73 features de memoria larga estén "
       "realmente ahí, incluida `i1_v20`, que es la que motivó todo esto."),
    code("!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
         "--train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm "
         "--hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --save-valid-preds --ema-features --ema-plus "
         "--seed 1 --model-seed 1 --distill-teachers $TEACHERS --distill-cache '' --run-name smoke\n"
         "import os, json, torch\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLO'\n"
         "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
         "print('Prueba de humo OK, valid WP:', round(m['best_valid_wp'], 4))\n"
         "pt = [f for f in os.listdir('/kaggle/tmp/smoke/smoke') if f.endswith('.pt') and 'checkpoint' not in f][0]\n"
         "ck = torch.load(f'/kaggle/tmp/smoke/smoke/{pt}', map_location='cpu', weights_only=False)\n"
         "cols = ck['pipeline']['names']\n"
         "mem = [c for c in cols if 'ema' in c or 'vol_lenta' in c]\n"
         "print('entradas:', ck['n_inputs'], '| de memoria larga:', len(mem), '(antes eran 40)')\n"
         "assert ck['n_inputs'] == 185, f\"esperaba 185 entradas, hay {ck['n_inputs']}\"\n"
         "assert len(mem) == 73, f'esperaba 73 de memoria larga, hay {len(mem)}'\n"
         "assert any('i1_v20' in c for c in mem), 'falta i1_v20, que es la que motivo el experimento'\n"
         "print('columnas nuevas:', [c for c in mem if 'i1_v20' in c or 'i0_p9' in c])\n"),

    md("## La alumna con memoria larga extendida"),
    code(f"!python train_official_gpu_v1.py {RECETA} --ema-features --ema-plus "
         f"--distill-cache /kaggle/tmp/soft_S19.npz --seed 2191 --model-seed 2191 "
         f"--run-name student_emaplus || echo 'FALLO student_emaplus'\n"),

    md("## Resumen"),
    code("import json, os\n"
         "d = '/kaggle/working/results/student_emaplus'\n"
         "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
         "if m:\n"
         "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
         "    mejor = max(m['best_valid_wp'], m.get('swa_valid_wp') or 0)\n"
         "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
         "    print('curva:', [round(v, 4) for v in h])\n"
         "    print()\n"
         "    print('referencia EXACTA: student_ema (40 features de memoria larga) = 0.6838')\n"
         "    print('                   student_long (misma receta, sin esas features) = 0.6838')\n"
         "    print('o sea que la version anterior de esta idea valio +0.0000')\n"
         "    print()\n"
         "    print('veredicto:', 'SIRVE' if mejor >= 0.6865 else ('ruido' if mejor >= 0.6815 else 'PEOR'))\n"
         "    print(sorted(os.listdir(d)))\n"
         "else:\n"
         "    print('FALLO: revisar el error de arriba')\n"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "connectome_sprint19_emaplus.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", out.name, f"({len(cells)} celdas)")
