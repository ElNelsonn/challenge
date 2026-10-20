r"""Genera los dos notebooks del sprint 17 (2026-10-09): las dos últimas cosas sin probar, ambas de entrada.

Las dos corridas son la MISMA receta exacta de la alumna LR (LSTM 128x2, destilada de 4 maestros, lr 5e-4,
10 épocas, SWA 7-10) que dio **0.6834**. Lo único que cambia en cada una es una línea. Así la comparación es
limpia y lo que salga se puede atribuir al cambio y a nada más.

  A) connectome_sprint17_sentinel.ipynb — `--sentinel-flags` (~2 h)
     Tres de las 112 columnas (a5, a6, a7) usan el valor **-5.1993** como marca de "este dato no existe", y
     nosotros se lo pasamos a la red como si fuera un número de verdad. Medido en valid:

        a7  tiene el centinela en el 44,0 % de las filas -> su desvío pasa de 0,583 a 2,934 (inflado 5,0x)
        a6  en el 17,5 %  -> desvío 0,808 -> 2,154 (2,7x)
        a5  en el 10,9 %  -> desvío 0,816 -> 1,903 (2,3x)

     Al estandarizar dividimos por ese desvío inflado, así que la variación real de a7 le llega a la red
     comprimida a una quinta parte. Y encima, sobre las filas puntuadas, a7 **sin** los centinelas correlaciona
     **0,21** con t1 (la correlación más alta de una sola feature en todo el proyecto); tal como se la pasamos
     hoy, correlaciona **0,04**. La bandera pone el centinela en 0 y agrega una columna que avisa.

  B) connectome_sprint17_cd.ipynb — `--target-cd` (~2 h)
     t0 y t1 correlacionan entre -0,58 y -0,87: son casi la misma cosa con el signo cambiado. Entrenando sobre
     t0 y t1 por separado, la red reparte su esfuerzo entre dos objetivos redundantes. Con `--target-cd` aprende
     la parte común (t0+t1)/2 y la diferencia (t0-t1)/2, que es donde está casi toda la información, cada una
     con su propio peso. Al guardar, los pesos de la última capa se transforman de vuelta a t0/t1, así que el
     zip de submission y el kernel de inferencia no cambian en nada.

Criterio para las dos: sirve si pasa de **0.6855** (0.6834 + 0.002, dos veces el ruido entre semillas que
medimos). Si quedan en 0.683-0.684 son ruido y se cierran.

Uso (desde challenge/):  python notebooks_kaggle/make_sprint17.py
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
# receta EXACTA de la alumna LR (0.6834), leída de su metrics.json
RECETA = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
          f"--output-dir /kaggle/working/results --save-valid-preds --cell lstm --hidden 128 --layers 2 "
          f"--dropout 0.1 --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 --epochs 10 --swa-start 7 "
          f"--distill-weight 1.0 --distill-teachers {teacher_paths}")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def check():
    return code("import os, torch\n"
                "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU T4 x2)'\n"
                "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
                f"for p in {TRAIN.split()} + ['{VALID}']:\n"
                "    assert os.path.isfile(p), f'Falta input: {p}'\n"
                f"for f in {list(TEACHERS.values())}:\n"
                f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models'\n"
                "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n")


def archivos():
    return [code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
            code("%%writefile train_official_gpu_v1.py\n" +
                 (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8"))]


def humo(flag, extra_check=""):
    """Prueba de humo que recorre EXACTAMENTE el código nuevo que va a usar la corrida larga."""
    return code(f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} "
                f"--train-groups 32 --valid-groups 4 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm "
                f"--hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 --save-valid-preds {flag} --seed 1 --model-seed 1 "
                f"--distill-teachers {teacher_paths} --distill-cache '' --run-name smoke\n"
                "import os, json, torch, numpy as np\n"
                "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ'\n"
                "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
                "print('Prueba de humo OK, valid WP:', round(m['best_valid_wp'], 4))\n"
                "pt = [f for f in os.listdir('/kaggle/tmp/smoke/smoke') if f.endswith('.pt') and 'checkpoint' not in f][0]\n"
                "ck = torch.load(f'/kaggle/tmp/smoke/smoke/{pt}', map_location='cpu', weights_only=False)\n"
                "sd = ck['state_dict']\n"
                "assert sd['head.weight'].shape[0] == 2, f\"la cabeza guardada tiene {sd['head.weight'].shape[0]} salidas, deberia tener 2\"\n"
                "pr = np.load('/kaggle/tmp/smoke/smoke/valid_preds_best.npz')['preds']\n"
                "print('predicciones guardadas:', pr.shape, '| rango', round(float(pr.min()), 3), round(float(pr.max()), 3))\n"
                "assert abs(float(pr.min())) < 50 and abs(float(pr.max())) < 50, 'predicciones fuera de escala'\n"
                + extra_check)


def resumen(nombre, ref):
    return code("import json, os\n"
                f"d = '/kaggle/working/results/{nombre}'\n"
                "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
                "if m:\n"
                "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
                "    mejor = max(m['best_valid_wp'], m.get('swa_valid_wp') or 0)\n"
                "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
                "    print('curva por epoca:', curva)\n"
                f"    print('{ref}')\n"
                "    print('veredicto:', 'SIRVE' if mejor >= 0.6855 else ('ruido' if mejor >= 0.6815 else 'PEOR'))\n"
                "    print(sorted(os.listdir(d)))\n"
                "else:\n"
                "    print('FALLO: revisar el error de arriba')\n")


REF = "referencia EXACTA: alumna LR, misma receta sin el cambio = 0.6834 | criterio para seguir: 0.6855"

# ---------------------------------------------------------------- A: columnas centinela
nb_a = {"cells": [
    md("# Connectome — sprint 17A: **las columnas centinela**\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models`. GPU T4 x2. **Save & Run All.** ~2 h.\n\n"
       "**El problema, medido hoy en valid.** Tres de las 112 columnas usan el número **−5,1993** como marca de "
       "\"este dato no existe\". Nosotros se lo venimos pasando a la red como si fuera un valor real, en **todas** "
       "las corridas del proyecto:\n\n"
       "| columna | filas con el centinela | desvío real | desvío que ve la red | inflado |\n"
       "|---|---|---|---|---|\n"
       "| a7 | **44,0 %** | 0,583 | 2,934 | **5,0x** |\n"
       "| a6 | 17,5 % | 0,808 | 2,154 | 2,7x |\n"
       "| a5 | 10,9 % | 0,816 | 1,903 | 2,3x |\n\n"
       "Al estandarizar dividimos por el desvío inflado, así que **la variación real de a7 le llega a la red "
       "comprimida a una quinta parte**, y encima la red tiene que gastar capacidad en aprender sola la regla "
       "\"si a7 vale −5,2, ignorála\".\n\n"
       "**Y a7 importa.** Sobre las filas que la métrica puntúa, a7 sin los centinelas correlaciona **0,21** con "
       "t1 — la correlación más alta de una sola feature en todo el proyecto. Tal como se la pasamos hoy, "
       "correlaciona **0,04**.\n\n"
       "**El cambio:** `--sentinel-flags` pone el centinela en 0 y agrega 3 columnas que avisan cuál fila lo "
       "tenía (115 entradas en vez de 112). La bandera está programada desde agosto y **nunca se usó en una "
       "corrida en serio**.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, destilada de los 4 maestros, semilla 2171 |\n"
       "| receta | **idéntica** a la alumna LR: lr 5e-4, decaimiento 0,9, 10 épocas, SWA 7–10 |\n"
       "| único cambio | `--sentinel-flags` |\n"
       "| comparar contra | **la alumna LR = 0.6834** |\n\n"
       "**Criterio:** sirve si pasa de **0.6855** (0.6834 más dos veces el ruido entre semillas). Un aviso "
       "honesto: probamos esta codificación con un modelo lineal y no cambió nada, pero el modelo lineal no "
       "puede sufrir este problema (absorbe el desvío inflado en su coeficiente). Esta corrida es la prueba que "
       "corresponde."),
    check(), *archivos(),
    md("## Prueba de humo (~3 min)\n\n"
       "Recorre el mismo camino que la corrida larga: arma el pipeline con 115 columnas, entrena una época, "
       "guarda el modelo y las predicciones, y verifica que la cabeza guardada tenga 2 salidas."),
    humo("--sentinel-flags",
         "assert ck['pipeline']['sentinel_flags'] is True, 'el pipeline guardado NO tiene las banderas'\n"
         "cols = ck['pipeline']['names']\n"
         "print('columnas de entrada:', ck['n_inputs'], '| en el pipeline:', len(cols))\n"
         "assert ck['n_inputs'] == 115 and len(cols) == 115, f'esperaba 115 columnas, hay {ck[\"n_inputs\"]}'\n"
         "assert cols[-3:] == ['a5_is_sentinel', 'a6_is_sentinel', 'a7_is_sentinel'], cols[-3:]\n"
         "print('ultimas 6 columnas:', cols[-6:])\n"),
    md("## Alumna con las columnas centinela arregladas"),
    code(f"!python train_official_gpu_v1.py {RECETA} --sentinel-flags "
         f"--distill-cache /kaggle/tmp/soft_S17.npz --seed 2171 --model-seed 2171 "
         f"--run-name student_sent || echo 'FALLÓ student_sent'\n"),
    md("## Resumen"), resumen("student_sent", REF),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

# ---------------------------------------------------------------- B: suma y diferencia
nb_b = {"cells": [
    md("# Connectome — sprint 17B: aprender **suma y diferencia** en vez de t0 y t1\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models`. GPU T4 x2. **Save & Run All.** ~2 h.\n\n"
       "**La observación:** t0 y t1 correlacionan entre **−0,58 y −0,87**. Son casi la misma cosa con el signo "
       "cambiado. Entrenando sobre los dos por separado, la red reparte su esfuerzo entre dos objetivos que se "
       "repiten, y el peso de cada fila (\\|t0\\| para uno, \\|t1\\| para el otro) tira para lados distintos.\n\n"
       "**El cambio:** con `--target-cd` la red aprende dos cosas que no se pisan:\n\n"
       "- la **parte común**, (t0+t1)/2 — chica, porque se cancelan\n"
       "- la **diferencia**, (t0−t1)/2 — ahí está casi toda la información\n\n"
       "Cada una con su propio peso. Al guardar el modelo, los pesos de la última capa se transforman de vuelta "
       "a t0 y t1 (t0 = común + diferencia, t1 = común − diferencia), así que **el zip de submission y el kernel "
       "de inferencia no cambian en nada**: el modelo guardado sigue devolviendo t0 y t1 como siempre.\n\n"
       "| | |\n|---|---|\n| modelo | LSTM 128x2, destilada de los 4 maestros, semilla 2172 |\n"
       "| receta | **idéntica** a la alumna LR: lr 5e-4, decaimiento 0,9, 10 épocas, SWA 7–10 |\n"
       "| único cambio | `--target-cd` (los soft targets del maestro también se transforman) |\n"
       "| comparar contra | **la alumna LR = 0.6834** |\n\n"
       "**Criterio:** sirve si pasa de **0.6855**. Estaba anotado como pendiente desde septiembre y nunca se "
       "había implementado. Vale aclarar que puede salir peor: la métrica nos puntúa t0 y t1 por separado, y "
       "esta reparametrización optimiza algo parecido pero no igual."),
    check(), *archivos(),
    md("## Prueba de humo (~3 min)\n\n"
       "Además de entrenar una época, verifica lo único delicado de este cambio: que el modelo **guardado** "
       "devuelva t0 y t1 (no la suma y la diferencia). Lo comprueba comparando la salida del modelo guardado "
       "contra las predicciones que el entrenador dejó en el npz, que ya están en t0/t1."),
    humo("--target-cd",
         "# el .pt del modelo guarda un config reducido; el completo esta en checkpoint.pt\n"
         "full = torch.load('/kaggle/tmp/smoke/smoke/checkpoint.pt', map_location='cpu', weights_only=False)\n"
         "assert full['config'].get('target_cd') is True, 'el checkpoint no registro target_cd'\n"
         "print('cabeza guardada:', tuple(sd['head.weight'].shape),\n"
         "      '| sesgo:', [round(float(v), 4) for v in sd['head.bias']])\n"
         "# prueba de verdad: correr el modelo guardado y comparar contra el npz, que ya esta en t0/t1\n"
         "import numpy as np, train_official_gpu_v1 as T\n"
         "net = T.RNNModel(ck['n_inputs'], full['config']['hidden'], full['config']['layers'],\n"
         "                 full['config']['cell'], 0.0)\n"
         "net.load_state_dict(sd); net.eval()\n"
         "pipe = T.FeaturePipeline(full['config']['top_k'], full['config']['sentinel_flags'],\n"
         "                         __import__('pathlib').Path(full['config']['ranking']),\n"
         "                         full['config'].get('cross_features', False),\n"
         "                         full['config'].get('ema_features', False))\n"
         "pipe.mean, pipe.std = full['pipeline']['mean'], full['pipeline']['std']\n"
         "g = T.ParquetGroups(['%s'])\n"
         "ids = [int(v) for v in np.load('/kaggle/tmp/smoke/smoke/valid_preds_best.npz')['seq_ix']]\n"
         "dif = None\n"
         "for gi in range(len(g)):\n"
         "    d = g.read(gi)\n"
         "    if int(d[0]) != ids[0]:\n"
         "        continue\n"
         "    with torch.no_grad():\n"
         "        q = net(torch.from_numpy(pipe.transform(d[2])).unsqueeze(0))[0][0].numpy()\n"
         "    dif = float(np.abs(q - pr[0].astype(np.float64)).max())\n"
         "    break\n"
         "print('diferencia maxima entre el modelo guardado y el npz:', dif)\n"
         "assert dif is not None and dif < 2e-3, 'la cabeza NO quedo en el espacio t0/t1'\n"
         "print('OK: el modelo guardado devuelve t0 y t1, el kernel de inferencia no cambia')\n" % VALID),
    md("## Alumna que aprende suma y diferencia"),
    code(f"!python train_official_gpu_v1.py {RECETA} --target-cd "
         f"--distill-cache /kaggle/tmp/soft_S17.npz --seed 2172 --model-seed 2172 "
         f"--run-name student_cd || echo 'FALLÓ student_cd'\n"),
    md("## Resumen"), resumen("student_cd", REF),
], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

for nombre, nb in (("sentinel", nb_a), ("cd", nb_b)):
    out = HERE / f"connectome_sprint17_{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, f"({len(nb['cells'])} celdas)")
