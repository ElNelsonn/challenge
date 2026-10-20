r"""Genera el notebook del sprint 20 (2026-10-17): XGBoost, con todo lo demás idéntico a LightGBM.

Joaquín lo pidió tres veces y hasta ahora yo solo había argumentado por qué no esperaba mucho. Se corre.

QUÉ CAMBIA Y QUÉ NO. Mismas 10,6M filas, mismas 190 columnas, mismos 600 árboles, mismas 200 secuencias de
valid, misma semilla. **Lo único que cambia es la librería que arma los árboles.**

La diferencia de fondo entre las dos: LightGBM hace crecer el árbol **por hojas** (parte siempre la hoja que
más promete, y el árbol queda desbalanceado y profundo); XGBoost crece **por niveles** (completa cada nivel
antes de bajar). Con datos tan ruidosos como estos, crecer por niveles con la profundidad limitada a veces
generaliza mejor. `--max-depth 6` da hasta 64 hojas, que es lo más parecido a las 63 de LightGBM.

LAS DOS COSAS QUE SE MIDEN:

  1. **El puntaje**, contra el 0.6390 de LightGBM.
  2. **La correlación**, que para nosotros vale tanto como el puntaje. Nuestra cuenta: un modelo aporta al
     ensamble si `s > S·r`. Con el ensamble en 0.6755 y LightGBM en r=0.9271, los árboles necesitaban pasar de
     0.6263 y dieron 0.6390: aportaron −0.0014 medido honestamente. **Bajar la correlación de 0.927 a 0.90
     vale lo mismo que subir el puntaje de 0.639 a 0.655.** Si XGBoost arma árboles lo bastante distintos, el
     número cambia aunque el puntaje sea igual.

Corre en **CPU**, no gasta nada de las 30 h semanales de GPU.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint20.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

IMP = CH / "model" / "lgbm_deltas" / "results" / "lgbm_deltas_all" / "feature_importance.json"
importancia = json.loads(IMP.read_text(encoding="utf-8"))

COMUN = ("--train-parquet $TRAIN --valid-parquet $VALID --lags 1 5 20 100 --ema-features "
         "--top-features 150 --importance importancia.json --learning-rate 0.05 "
         "--min-data-in-leaf 200 --feature-fraction 0.5 --bagging-fraction 0.8 --lambda-l2 1.0 "
         "--threads -1 --lib xgboost --max-depth 6")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# Connectome — sprint 20: **XGBoost**\n\n"
       "**Inputs:** train part1/part2, valid. **CPU** (Accelerator → None). **Save & Run All.** ~3 h.\n\n"
       "---\n\n"
       "## Qué cambia y qué no\n\n"
       "Mismas **10,6M filas**, mismas **190 columnas**, mismos **600 árboles**, mismas **200 secuencias** de "
       "valid, misma semilla. **Lo único que cambia es la librería que arma los árboles**, así que la "
       "comparación contra el 0.6390 de LightGBM es limpia.\n\n"
       "La diferencia de fondo entre las dos:\n\n"
       "| | cómo crece el árbol |\n|---|---|\n"
       "| LightGBM | **por hojas**: parte siempre la hoja que más promete. Árboles desbalanceados y profundos |\n"
       "| XGBoost | **por niveles**: completa cada nivel antes de bajar al siguiente |\n\n"
       "Con datos tan ruidosos como estos, crecer por niveles con la profundidad limitada a veces generaliza "
       "mejor. `--max-depth 6` da hasta 64 hojas, lo más parecido a las 63 de LightGBM.\n\n"
       "## Las dos cosas que se miden\n\n"
       "**1. El puntaje**, contra el **0.6390** de LightGBM.\n\n"
       "**2. La correlación con nuestro ensamble**, que para nosotros vale tanto como el puntaje. La cuenta "
       "del proyecto: un modelo aporta si `s > S × r`. Con el ensamble en **0.6755** y LightGBM en **r = "
       "0.9271**, los árboles tenían que pasar de 0.6263; dieron 0.6390 y aportaron **−0.0014** medido "
       "honestamente.\n\n"
       "| para ganar +0.004 el modelo de árboles necesita… | |\n|---|---|\n"
       "| subir el puntaje | de 0.639 a **0.655** |\n| **o** bajar la correlación | de 0.927 a **0.90** |\n\n"
       "O sea que **si XGBoost arma árboles lo bastante distintos, el número cambia aunque el puntaje sea "
       "igual**. Por eso se guardan las predicciones: la correlación se mide después, en la laptop, contra "
       "las de LightGBM y las de las 4 redes."),

    code("import os, sys, time, glob, json\n"
         "import xgboost as xgb; print('xgboost', xgb.__version__)\n"
         "import lightgbm as lgb; print('lightgbm', lgb.__version__)\n"
         "import psutil; print('RAM: %.1f GB' % (psutil.virtual_memory().total/1e9))\n"
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
         "TRAIN = ' '.join(buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet'))\n"
         "VALID = buscar('valid.parquet')\n"
         "print('train:', TRAIN)\nprint('valid:', VALID)\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),

    code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
    code("%%writefile train_lgbm_deltas_v1.py\n" +
         (CH / "model" / "lgbm_deltas" / "train_lgbm_deltas_v1.py").read_text(encoding="utf-8")),

    md("## Importancia de las features (para elegir las 150 columnas)"),
    code("import json\n"
         f"importancia = {json.dumps(importancia)}\n"
         "json.dump(importancia, open('importancia.json', 'w'))\n"
         "print(len(importancia), 'features; las 150 mejores tienen el 90.3 % de la ganancia')\n"),

    md("## Prueba de humo (~3 min)\n\n"
       "Recorre el mismo camino que la corrida larga con 8 secuencias y 30 árboles. Ya se probó en la laptop "
       "con xgboost 3.4.1; esto verifica que también ande con la versión de Kaggle."),
    code(f"!python train_lgbm_deltas_v1.py {COMUN} --train-groups 8 --rows-per-group 200 --valid-groups 12 "
         f"--n-estimators 30 --output-dir /kaggle/tmp/smoke --run-name humo\n"
         "import os, json\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/humo/metrics.json'), 'La prueba de humo FALLO'\n"
         "m = json.load(open('/kaggle/tmp/smoke/humo/metrics.json'))\n"
         "print('Prueba de humo OK | WP:', round(m['lgbm_global']['weighted_pearson'], 4))\n"
         "assert m['config']['lib'] == 'xgboost', 'no corrio con xgboost'\n"
         "assert os.path.isfile('/kaggle/tmp/smoke/humo/xgb_t0.json'), 'no guardo el modelo de xgboost'\n"
         "esp = json.load(open('/kaggle/tmp/smoke/humo/feature_spec.json'))\n"
         "assert len(esp['names']) == 190, f\"esperaba 190 columnas, hay {len(esp['names'])}\"\n"
         "print('columnas:', len(esp['names']), '| libreria:', m['config']['lib'])\n"),

    md("## XGBoost, con todo\n\n"
       "10.607 secuencias × 1.000 filas = **10,6M filas**, 190 columnas, 600 árboles por target, "
       "evaluado en **las mismas 200 secuencias** que usó LightGBM (misma semilla)."),
    code(f"!python train_lgbm_deltas_v1.py {COMUN} --train-groups all --rows-per-group 1000 "
         f"--valid-groups 200 --n-estimators 600 --seed 2026 "
         f"--output-dir /kaggle/working/results --run-name xgb_v1 || echo 'FALLO xgb_v1'\n"),

    md("## Resumen"),
    code("import json, os\n"
         "d = '/kaggle/working/results/xgb_v1'\n"
         "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
         "if m:\n"
         "    s = m['lgbm_global']['weighted_pearson']\n"
         "    print(f\"XGBoost:  {s:.4f}   (t0 {m['lgbm_global']['t0']:.4f}, t1 {m['lgbm_global']['t1']:.4f})\")\n"
         "    print(f'LightGBM: 0.6390   (mismas filas, mismas columnas, mismos arboles)')\n"
         "    print(f'diferencia: {s - 0.6390:+.4f}')\n"
         "    print()\n"
         "    print('ensamble de 4 redes en estas mismas 200 secuencias: 0.6755')\n"
         "    print('umbral si la correlacion fuera la de LightGBM (0.9271): 0.6263')\n"
         "    print()\n"
         "    print('OJO: el puntaje es solo la mitad. La correlacion con el ensamble se mide en la laptop')\n"
         "    print('con valid_predictions.npz, y es la que decide si entra o no.')\n"
         "    imp = json.load(open(d + '/feature_importance.json'))\n"
         "    print()\n"
         "    print('top 15:', [k for k, _ in imp[:15]])\n"
         "    print(sorted(os.listdir(d)))\n"
         "else:\n"
         "    print('FALLO: revisar el error de arriba')\n"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "connectome_sprint20_xgboost.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", out.name, f"({len(cells)} celdas)")
