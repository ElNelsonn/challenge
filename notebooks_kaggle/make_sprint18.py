r"""Genera el notebook del sprint 18 (2026-10-13): volver a los árboles, pero con la cuenta hecha.

POR QUÉ. En septiembre probamos LightGBM (la misma familia que XGBoost) y lo descartamos por tiempo de
inferencia. Nunca medimos lo único que importaba. Medido ahora, sobre las predicciones que quedaron guardadas:

    los árboles correlacionan 0.900-0.930 con cada una de nuestras redes,
    y 0.918 con el promedio del ensamble.

Para comparar: nuestras redes correlacionan 0.956-0.987 entre ellas, y la convolucional del sprint 16, que era
otra arquitectura, daba 0.957-0.978. Los árboles son, por lejos, lo más distinto que tenemos.

LA CUENTA. Para un ensamble con puntaje S y un modelo nuevo con puntaje s y correlación r con el ensamble, la
mezcla mejora si y solo si  s > S * r. Con S = 0.6755 (nuestras 4 redes en las 200 secuencias de este sprint):

    r = 0.918 (árboles)        ->  necesitan pasar de 0.620
    r = 0.978 (convolucional)  ->  necesitaba pasar de 0.661, dio 0.654  -> por eso aportó 0.0000
    r = 0.985 (otra LSTM)      ->  necesita pasar de 0.665, o sea igualar al ensamble entero

Los árboles de septiembre dieron 0.6253, apenas 0.005 arriba de su umbral: aportan +0.0001, nada. Pero la
pendiente es muy empinada justo ahí, porque su correlación es baja:

    si los árboles dieran 0.640  ->  ensamble +0.0019
                          0.650  ->  ensamble +0.0042
                          0.660  ->  ensamble +0.0075

O sea que no hace falta que los árboles se acerquen a las redes. Con que suban 0.025 (de 0.625 a 0.650) ya es
la mejora más grande en un mes.

QUÉ CAMBIA RESPECTO DE SEPTIEMBRE. Tres cosas, y ninguna es un ajuste fino:

  1. **3,3 veces más datos.** Septiembre usó 300 filas por secuencia = 3,2M filas, el 1,5 % de train. Ahora
     1.000 por secuencia = 10,6M filas. El límite es la memoria, no el tiempo.
  2. **Las 40 features de memoria larga** (desvíos respecto de promedios móviles de 100, 1.000 y 10.000 filas,
     más volatilidad lenta). Un árbol no tiene estado: no puede calcularse esto solo. Nuestras redes lo
     consiguen gratis por ser recurrentes, y es su ventaja principal. En la prueba chica de la laptop, cuatro
     de estas features aparecieron entre las más importantes de entrada.
  3. **Sacar 410 columnas muertas.** De las 560 de septiembre, las 150 mejores concentran el 90,3 % de la
     ganancia. Sacar el resto es lo que libera la memoria para el punto 1.

Y CORRE EN **CPU**, no en GPU: no gasta nada de las 30 h semanales, así que puede correr en paralelo con lo que
haya en la GPU.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint18.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TRAIN = ("/kaggle/input/datasets/joaquinfiorina/connectome-train-part1/train_part1.parquet "
         "/kaggle/input/datasets/joaquinfiorina/connectome-train-part2/train_part2.parquet")
VALID = "/kaggle/input/datasets/joaquinfiorina/connectome-valid/valid.parquet"

IMP = CH / "model" / "lgbm_deltas" / "results" / "lgbm_deltas_all" / "feature_importance.json"
importancia = json.loads(IMP.read_text(encoding="utf-8"))

COMUN = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --lags 1 5 20 100 --ema-features "
         f"--top-features 150 --importance importancia.json --learning-rate 0.05 --num-leaves 63 "
         f"--min-data-in-leaf 200 --feature-fraction 0.5 --bagging-fraction 0.8 --lambda-l2 1.0 --threads -1")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# Connectome — sprint 18: **los árboles, con la cuenta hecha**\n\n"
       "**Inputs:** train part1/part2, valid. **CPU** (no hace falta GPU: Session options → Accelerator → None). "
       "**Save & Run All.** ~2,5 h.\n\n"
       "---\n\n"
       "## Por qué volvemos a esto\n\n"
       "En septiembre probamos LightGBM (la misma familia que XGBoost) y lo descartamos por tiempo de "
       "inferencia. Nunca medimos lo único que importaba. Medido ahora, sobre las predicciones que habían "
       "quedado guardadas:\n\n"
       "| | correlación con el promedio del ensamble |\n|---|---|\n"
       "| otra LSTM como las nuestras | 0.985 |\n"
       "| la convolucional del sprint 16 | 0.978 |\n"
       "| **los árboles** | **0.918** |\n\n"
       "Son, por lejos, lo más distinto que conseguimos en todo el proyecto.\n\n"
       "## La cuenta\n\n"
       "Para un ensamble con puntaje **S** y un modelo nuevo con puntaje **s** y correlación **r** con ese "
       "ensamble, mezclarlos mejora **si y solo si  s > S × r**. Con S = **0.6755** (nuestras 4 redes medidas "
       "en las mismas 200 secuencias que usa este notebook):\n\n"
       "| modelo nuevo | r | tiene que pasar de | dio | resultado |\n|---|---|---|---|---|\n"
       "| otra LSTM | 0.985 | 0.665 | ~0.684 | casi igualar al ensamble entero: imposible en la práctica |\n"
       "| la convolucional | 0.978 | 0.661 | 0.654 | **0.0000**, le faltaron 0.007 |\n"
       "| **los árboles de septiembre** | **0.918** | **0.620** | 0.625 | **+0.0001**: justo en el límite |\n\n"
       "Y como su correlación es baja, la pendiente ahí es empinada:\n\n"
       "| si los árboles dieran | el ensamble ganaría |\n|---|---|\n"
       "| 0.640 | +0.0019 |\n| 0.650 | **+0.0042** |\n| 0.660 | **+0.0075** |\n| 0.670 | **+0.0117** |\n\n"
       "**No hace falta que los árboles se acerquen a las redes.** Con que suban 0.025 (de 0.625 a 0.650) es la "
       "mejora más grande en un mes.\n\n"
       "## Qué cambia respecto de septiembre\n\n"
       "1. **3,3 veces más datos.** Septiembre usó 300 filas por secuencia = 3,2M filas, el **1,5 %** de train. "
       "Acá van 1.000 por secuencia = **10,6M filas**. El límite es la memoria, no el tiempo.\n"
       "2. **Las 40 features de memoria larga**: desvíos respecto de promedios móviles de 100, 1.000 y 10.000 "
       "filas, más volatilidad lenta. **Un árbol no tiene estado**, no puede calcularse esto solo; nuestras "
       "redes lo consiguen gratis por ser recurrentes y es su ventaja principal. En la prueba chica de la "
       "laptop, cuatro de estas features aparecieron entre las más importantes de entrada.\n"
       "3. **Sacar 410 columnas muertas.** De las 560 de septiembre, las 150 mejores tienen el **90,3 %** de la "
       "ganancia. Sacarlas es lo que libera memoria para el punto 1.\n\n"
       "## Lo que hay que mirar al final\n\n"
       "El número clave es **el puntaje de los árboles solos**, contra el **0.6253** de septiembre:\n\n"
       "- **≥ 0.650** → entra al ensamble y es la mejora grande que buscamos\n"
       "- 0.635–0.650 → aporta poco pero real; hay que ver si paga el tiempo de inferencia\n"
       "- ≤ 0.630 → los árboles quedan cerrados con evidencia\n\n"
       "El tiempo de inferencia es el otro problema y está medido aparte: 600 árboles cuestan **58 µs por fila "
       "por target**, y nos quedan 30 µs libres. La salida: correrlos solo en las filas que la métrica puntúa. "
       "Con el modelo de la máscara, corriéndolos en el 25 % de las filas se atrapa el **68,7 %** del peso de "
       "la métrica y entran **623 árboles**. Eso se resuelve después de saber si los árboles mejoraron."),

    code("import os, sys, json, numpy as np\n"
         "print('python', sys.version.split()[0])\n"
         "import lightgbm as lgb; print('lightgbm', lgb.__version__)\n"
         f"for p in {TRAIN.split()} + ['{VALID}']:\n"
         "    assert os.path.isfile(p), f'Falta input: {p}'\n"
         "import psutil; print('RAM disponible: %.1f GB' % (psutil.virtual_memory().total/1e9))\n"
         "assert psutil.virtual_memory().total > 20e9, 'Hace falta la sesion de CPU (30 GB), no la de GPU (13 GB)'\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),

    code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
    code("%%writefile train_lgbm_deltas_v1.py\n" +
         (CH / "model" / "lgbm_deltas" / "train_lgbm_deltas_v1.py").read_text(encoding="utf-8")),

    md("## Importancia de las features de septiembre\n\n"
       "Se usa para elegir las 150 columnas que valen la pena."),
    code("import json\n"
         f"importancia = {json.dumps(importancia)}\n"
         "json.dump(importancia, open('importancia.json', 'w'))\n"
         "g = [v for _, v in importancia]\n"
         "tot = sum(g)\n"
         "print(f'{len(g)} features | las 150 mejores tienen el {100*sum(g[:150])/tot:.1f} % de la ganancia')\n"
         "print('top 10:', [k for k, _ in importancia[:10]])\n"),

    md("## Prueba de humo (~3 min)\n\n"
       "Recorre exactamente el mismo camino que la corrida larga (features de memoria larga, selección de "
       "columnas, entrenamiento y evaluación), con 8 secuencias y 30 árboles."),
    code(f"!python train_lgbm_deltas_v1.py {COMUN} --train-groups 8 --rows-per-group 200 --valid-groups 12 "
         f"--n-estimators 30 --output-dir /kaggle/tmp/smoke --run-name humo\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/humo/metrics.json'), 'La prueba de humo FALLO'\n"
         "m = json.load(open('/kaggle/tmp/smoke/humo/metrics.json'))\n"
         "print('Prueba de humo OK | WP:', round(m['lgbm_global']['weighted_pearson'], 4))\n"
         "esp = json.load(open('/kaggle/tmp/smoke/humo/feature_spec.json'))\n"
         "assert esp['ema'] is True, 'no se usaron las features de memoria larga'\n"
         "assert len(esp['names']) == 190, f\"esperaba 190 columnas, hay {len(esp['names'])}\"\n"
         "ema = [n for n in esp['names'] if 'ema' in n or 'vol_lenta' in n]\n"
         "print(f'columnas: {len(esp[\"names\"])} (de las cuales {len(ema)} son de memoria larga)')\n"),

    md("## Los árboles, con todo\n\n"
       "10.607 secuencias × 1.000 filas = **10,6M filas**, 190 columnas, 600 árboles por target.\n\n"
       "Evaluación en **200 secuencias** de valid (septiembre usó 10, demasiado pocas para distinguir 0.002)."),
    code(f"!python train_lgbm_deltas_v1.py {COMUN} --train-groups all --rows-per-group 1000 "
         f"--valid-groups 200 --n-estimators 600 --seed 2026 "
         f"--output-dir /kaggle/working/results --run-name arboles_v2 || echo 'FALLO arboles_v2'\n"),

    md("## Resumen"),
    code("import json, os\n"
         "d = '/kaggle/working/results/arboles_v2'\n"
         "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
         "if m:\n"
         "    s = m['lgbm_global']['weighted_pearson']\n"
         "    print(f\"ARBOLES v2: {s:.4f}   (t0 {m['lgbm_global']['t0']:.4f}, t1 {m['lgbm_global']['t1']:.4f})\")\n"
         "    print(f'septiembre:  0.6253')\n"
         "    print(f'diferencia:  {s - 0.6253:+.4f}')\n"
         "    print()\n"
         "    print('umbral para aportar al ensamble: 0.6755 x 0.918 = 0.6200')\n"
         "    print('ensamble de 4 redes en estas mismas 200 secuencias: 0.6755')\n"
         "    print()\n"
         "    if s >= 0.650:\n"
         "        print('VEREDICTO: SIRVE. Ganancia esperada en el ensamble: +0.004 o mas')\n"
         "    elif s >= 0.635:\n"
         "        print('VEREDICTO: aporta poco pero real (+0.001 a +0.002). Hay que ver si paga el tiempo')\n"
         "    elif s > 0.6253:\n"
         "        print('VEREDICTO: mejoro pero no alcanza')\n"
         "    else:\n"
         "        print('VEREDICTO: los arboles quedan cerrados con evidencia')\n"
         "    imp = json.load(open(d + '/feature_importance.json'))\n"
         "    tot = sum(v for _, v in imp)\n"
         "    ema = sum(v for k, v in imp if 'ema' in k or 'vol_lenta' in k)\n"
         "    print()\n"
         "    print(f'las 40 features de memoria larga se llevaron el {100*ema/tot:.1f} % de la ganancia')\n"
         "    print('top 15:', [k for k, _ in imp[:15]])\n"
         "    print()\n"
         "    print(sorted(os.listdir(d)))\n"
         "else:\n"
         "    print('FALLO: revisar el error de arriba')\n"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "connectome_sprint18_arboles.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", out.name, f"({len(cells)} celdas, {out.stat().st_size/1e6:.2f} MB)")
