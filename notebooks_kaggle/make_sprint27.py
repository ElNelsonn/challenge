import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

RECETA = ("--cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 1e-3 --lr-decay 0.7 --weight-decay 1e-4 --clip 1.0 "
          "--epochs 6 --swa-start 5 --unscored-weight 0.1 --train-groups all --valid-groups all --workers 4 "
          "--cv-folds 5 --save-valid-preds")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


BUSCADOR = (
    "import os, time, glob, torch\n"
    "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
    "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
    "\n"
    "def buscar(nombre, espera=600):\n"
    "    t0 = time.time()\n"
    "    while True:\n"
    "        hits = sorted(glob.glob(f'/kaggle/input/**/{nombre}', recursive=True))\n"
    "        if hits:\n"
    "            rapidas = [h for h in hits if '/datasets/' in h]\n"
    "            elegida = (rapidas or hits)[0]\n"
    "            print('   ', nombre, '->', elegida)\n"
    "            return elegida\n"
    "        if time.time() - t0 > espera:\n"
    "            for d in sorted(glob.glob('/kaggle/input/*')) + sorted(glob.glob('/kaggle/input/*/*')):\n"
    "                print('   ', d)\n"
    "            raise FileNotFoundError(nombre)\n"
    "        time.sleep(15)\n"
    "\n"
    "VALID = buscar('valid.parquet')\n"
    "INIT = buscar('ftv_init_LR_swa.pt')\n"
    "os.makedirs('/kaggle/working/results', exist_ok=True)\n"
    "\n"
    "import pyarrow.parquet as pq\n"
    "pf = pq.ParquetFile(VALID)\n"
    "t0 = time.time()\n"
    "for g in range(15):\n"
    "    pf.read_row_group(g, use_threads=False)\n"
    "seg = (time.time() - t0) / 15\n"
    "print(f'velocidad de lectura: {seg:.2f} s por secuencia')\n"
    "assert seg < 1.5, f'LECTURA LENTA: {seg:.1f} s por secuencia. Volver a lanzar (suerte de maquina).'\n"
    "print('Inputs OK')\n")


def humo(k):
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--train-groups 24 --valid-groups 60 --cv-folds 5 --cv-fold 0 --output-dir /kaggle/tmp/smoke "
            f"--epochs 1 --cell lstm --hidden 128 --layers 2 --lr 1e-3 --seed 1 --model-seed 1 "
            f"--init-from $INIT --unscored-weight 0.1 --lookahead {k} --run-name smoke\n"
            "import json\n"
            "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
            f"assert m['config']['lookahead'] == {k}, 'sin lookahead'\n"
            "assert not set(m['config']['train_ids']) & set(m['config']['valid_ids']), 'train y valid se pisan'\n"
            "print('Prueba de humo OK | WP', round(m['best_valid_wp'], 4))\n")


def correr(k, fold):
    nombre = f"la{k}_f{fold}"
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT --lookahead {k} {RECETA} --cv-fold {fold} "
            f"--seed {2700 + fold} --model-seed {2700 + fold} --run-name {nombre} || echo 'FALLO {nombre}'\n"
            f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt /kaggle/working/results/{nombre}/valid_preds_best.npz\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| valid', len(m['config']['valid_ids']), '| SWA',\n"
    "          round(m.get('swa_valid_wp', float('nan')), 4), '| curva', h)\n")

for k in [int(a) for a in sys.argv[1:]] or [25, 50]:
    nombre = f"connectome_sprint27_la{k}"
    titulo = (f"# Connectome — sprint 27: maestro que mira {k} filas adelante\n\n"
              "**Inputs:** `Connectome valid` y `Connectome ftv`. GPU. **Save & Run All.**\n\n"
              f"La LSTM LR, ajustada en valid para predecir el target de la fila t−{k} desde la fila t: ve {k} filas "
              "del futuro. Cinco pliegues: cada secuencia de valid recibe la predicción de un maestro que no la vio. "
              "Esas predicciones son las etiquetas blandas del sprint 28. Solo sirve para entrenar; no va a la entrega.")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo(k))]
    for fold in range(5):
        cells += [md(f"## la{k}_f{fold}"), code(correr(k, fold))]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
