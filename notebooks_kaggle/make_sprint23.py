r"""Genera los notebooks del sprint 23 (2026-10-20): normalizar las entradas DENTRO de cada secuencia.

EL PROBLEMA, MEDIDO HOY SOBRE 150 SECUENCIAS DE VALID:

    97 de las 112 features cambian de escala MAS DE 2x entre secuencias
    77 cambian mas de 3x
    la mediana es 3,87x, y las peores (i1_p9, i0_p9) llegan a 10x

Y nosotros las normalizamos con **una media y un desvio globales**, calculados una sola vez sobre una muestra
de train, desde el experimento 007 en septiembre. O sea que a la red le llegan numeros hasta 4 veces mas
grandes segun en que secuencia este, y tiene que gastar capacidad en darse cuenta sola y compensarlo.

Verificado en la laptop, mirando el desvio de cada feature por secuencia tal como lo ve la red:

    normalizacion global (hoy)    desvio tipico 0.74   variacion entre secuencias 4.07x
    normalizacion adaptativa      desvio tipico 0.98   variacion entre secuencias 1.09x

POR QUE PUEDE IMPORTAR MAS QUE TODO LO DEMAS. En septiembre medimos que si pudieramos multiplicar las
predicciones de cada secuencia por el factor correcto ganariamos **+0.047**. Nunca pudimos estimar ese factor
desde las features. Esto ataca el mismo problema por el otro lado: en vez de corregir la salida, pone la
entrada de todas las secuencias en la misma escala.

ES CAUSAL Y LEGAL: los promedios moviles solo miran el pasado, y arrancan en los valores globales, asi que las
primeras filas quedan normalizadas como hoy y la normalizacion se adapta a medida que hay historia.

  * connectome_sprint23_adapt2000.ipynb — se adapta rapido (2.000 filas de memoria)
  * connectome_sprint23_adapt8000.ipynb — se adapta lento (8.000), conserva mas dinamica interna como senal

Cuesta ~18 min por epoca de normalizacion extra, asi que la corrida pasa de ~2 h a ~5 h.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint23.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TEACHERS = ["gru128_distill_seed2029_swa.pt", "lstm128x2_l1_seed2047_swa.pt",
            "lstm128x2_studentLR_seed2072_swa.pt", "lstm128x2_studentLA_best.pt"]
CORRIDAS = {"adapt2000": (2000, 2231), "adapt8000": (8000, 2232)}

BUSCAR = ("import os, time, glob, torch\n"
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
          "            # Preferimos la ruta /datasets/ porque suele ser la cacheada, PERO el verdadero\n"
          "            # factor es que maquina te toca: medido, dos corridas tomaron esta misma ruta y una\n"
          "            # igual leyo a 6,6 s por secuencia. Lo que protege la cuota es la guarda de abajo.\n"
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
          f"TEACHERS = ' '.join(buscar(n) for n in {TEACHERS})\n"
          "print('train:', TRAIN)\nprint('valid:', VALID)\n"
          "os.makedirs('/kaggle/working/results', exist_ok=True)\n"
          "\n"
          "# CONTROL DE VELOCIDAD DE LECTURA. Kaggle a veces monta el dataset sin cache local en la maquina\n"
          "# que te toco, y leer pasa de 0.06 a 6 segundos por secuencia. Cuando eso pasa, la corrida se come\n"
          "# el limite de 12 h SOLO precalculando los targets del maestro, sin entrenar una sola epoca.\n"
          "# Ya nos costo tres corridas y 35 h de cuota. Medirlo cuesta 20 segundos.\n"
          "import pyarrow.parquet as pq\n"
          "pf = pq.ParquetFile(TRAIN.split()[0])\n"
          "t0 = time.time()\n"
          "for g in range(15):\n"
          "    pf.read_row_group(g, use_threads=False)\n"
          "seg = (time.time() - t0) / 15\n"
          "print(f'velocidad de lectura: {seg:.2f} s por secuencia '\n"
          "      f'(rapida < 0.5 | esta corrida necesita ~{seg*10607/3600:.1f} h solo para leer train una vez)')\n"
          "assert seg < 1.0, (f'LECTURA LENTA: {seg:.1f} s por secuencia. Kaggle monto el dataset sin cache en '\n"
          "                   f'esta maquina y la corrida moriria en el limite de 12 h sin entrenar nada. '\n"
          "                   f'Abortado a proposito. Volver a lanzar: es suerte de maquina, no un error del codigo.')\n"
          "print('Inputs OK')\n")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


for nombre, (span, seed) in CORRIDAS.items():
    cells = [
        md(f"# Connectome — sprint 23: normalizar **dentro de cada secuencia** (memoria {span:,})\n\n"
           "**Inputs:** train part1/part2, valid, `Connectome models`. GPU. **Save & Run All.** ~5 h.\n\n"
           "---\n\n"
           "## El problema, medido\n\n"
           "La escala de las features cambia muchísimo de una secuencia a otra. Medido sobre 150 secuencias "
           "de valid:\n\n"
           "| | |\n|---|---|\n"
           "| features que cambian de escala **más de 2x** entre secuencias | **97 de 112** |\n"
           "| más de 3x | 77 |\n"
           "| mediana | **3,87x** |\n"
           "| las peores (`i1_p9`, `i0_p9`) | **10x** |\n\n"
           "Y desde septiembre las normalizamos con **una media y un desvío globales**, calculados una sola "
           "vez sobre una muestra de train. O sea que a la red le llegan números hasta 4 veces más grandes "
           "según en qué secuencia esté, y tiene que gastar capacidad en darse cuenta sola y compensarlo.\n\n"
           "## La solución, verificada en la laptop\n\n"
           "Normalizar con una media y un desvío **móviles dentro de la secuencia**, que solo miran el pasado. "
           "Mirando el desvío de cada feature por secuencia, tal como lo ve la red:\n\n"
           "| | desvío típico | variación entre secuencias |\n|---|---|---|\n"
           "| normalización global (hoy) | 0.74 | **4,07x** |\n"
           f"| adaptativa (memoria {span:,}) | 0.98 | **1,09x** |\n\n"
           "**Es causal y legal:** los promedios móviles solo miran el pasado, y **arrancan en los valores "
           "globales**, así que las primeras filas quedan normalizadas como hoy y la normalización se adapta "
           "a medida que hay historia. No hay salto ni problema de arranque.\n\n"
           "## Por qué esto puede importar más que todo lo demás\n\n"
           "En septiembre medimos que si pudiéramos multiplicar las predicciones de cada secuencia por el "
           "factor correcto ganaríamos **+0.047**, y nunca pudimos estimar ese factor desde las features. "
           "**Esto ataca el mismo problema por el otro lado:** en vez de corregir la salida, pone la entrada "
           "de todas las secuencias en la misma escala.\n\n"
           "| | |\n|---|---|\n| modelo | LSTM 128x2, destilada de los 4 maestros, semilla " + str(seed) + " |\n"
           "| receta | **idéntica** a la alumna LR: lr 5e-4, decaimiento 0,9, 10 épocas, SWA 7–10 |\n"
           f"| único cambio | `--adapt-span {span}` |\n"
           "| comparar contra | **la alumna LR = 0.6834** |\n\n"
           f"**Criterio:** sirve si pasa de **0.6855**. {'Memoria corta: se adapta rápido pero puede borrar la volatilidad, que también es señal.' if span == 2000 else 'Memoria larga: conserva más dinámica interna como señal, pero tarda más en adaptarse.'}\n\n"
           "**Tarda ~5 h** porque la normalización agrega ~18 min por época."),

        code(BUSCAR),
        code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
        code("%%writefile train_official_gpu_v1.py\n" +
             (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),

        md("## Prueba de humo (~4 min)\n\n"
           "Además de entrenar una época, verifica **lo único que importa de este cambio**: que el desvío de "
           "las features que ve la red sea parecido en todas las secuencias, en vez de variar 4 veces."),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm "
             f"--hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --adapt-span {span} --seed 1 --model-seed 1 "
             f"--distill-teachers $TEACHERS --distill-cache '' --run-name smoke\n"
             "import os, json, torch, numpy as np\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLO'\n"
             "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
             f"assert m['config']['adapt_span'] == {span}, 'no se aplico la normalizacion adaptativa'\n"
             "print('Prueba de humo OK | memoria:', m['config']['adapt_span'], '| valid WP:', round(m['best_valid_wp'], 4))\n"
             "\n"
             "# la verificacion de fondo: comparar lo que ve la red con una y con otra normalizacion\n"
             "import train_official_gpu_v1 as T\n"
             "from pathlib import Path\n"
             "st = torch.load('/kaggle/tmp/smoke/smoke/checkpoint.pt', map_location='cpu', weights_only=False)['pipeline']\n"
             "g = T.ParquetGroups([VALID])\n"
             "glob_p = T.FeaturePipeline(0, False, Path('x'), False, False, False, 0)\n"
             f"adap_p = T.FeaturePipeline(0, False, Path('x'), False, False, False, {span})\n"
             "for q in (glob_p, adap_p): q.mean, q.std = st['mean'], st['std']\n"
             "sg, sa = [], []\n"
             "for i in range(25):\n"
             "    x = g.read(i)[2]\n"
             "    sg.append(np.nanstd(glob_p.transform(x), axis=0))\n"
             "    sa.append(np.nanstd(adap_p.transform(x), axis=0))\n"
             "def var_entre_seqs(M):\n"
             "    M = np.array(M); rel = M / np.median(M, axis=0)\n"
             "    p10, p90 = np.percentile(rel, [10, 90], axis=0)\n"
             "    return float(np.median(p90 / np.maximum(p10, 1e-9)))\n"
             "vg, va = var_entre_seqs(sg), var_entre_seqs(sa)\n"
             "print(f'variacion de escala entre secuencias: global {vg:.2f}x -> adaptativa {va:.2f}x')\n"
             "assert va < vg / 2, f'la normalizacion adaptativa no redujo la variacion ({vg:.2f} -> {va:.2f})'\n"
             "print('OK: la red ahora ve todas las secuencias en la misma escala')\n"),

        md(f"## La alumna con normalización adaptativa (memoria {span:,})"),
        code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
             f"--train-groups all --valid-groups all --output-dir /kaggle/working/results --save-valid-preds "
             f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
             f"--epochs 10 --swa-start 7 --adapt-span {span} --distill-weight 1.0 "
             f"--distill-teachers $TEACHERS --distill-cache /kaggle/tmp/soft_S23.npz "
             f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"),

        md("## Resumen"),
        code("import json, os\n"
            f"d = '/kaggle/working/results/{nombre}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    h = [x['wp'] for x in m['history'] if x['tag'] != 'swa']\n"
             "    mejor = max(m['best_valid_wp'], m.get('swa_valid_wp') or 0)\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva:', [round(v, 4) for v in h])\n"
             f"    print('memoria de la normalizacion:', m['config']['adapt_span'])\n"
             "    print()\n"
             "    print('referencia EXACTA: alumna LR, misma receta con normalizacion global = 0.6834')\n"
             "    print('veredicto:', 'SIRVE' if mejor >= 0.6855 else ('ruido' if mejor >= 0.6815 else 'PEOR'))\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint23_{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, f"(memoria {span}, semilla {seed})")
