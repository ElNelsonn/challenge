import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

RECETA = ("--cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 1e-3 --lr-decay 0.7 --weight-decay 1e-4 --clip 1.0 "
          "--epochs 3 --swa-start 2 --train-groups 2000 --valid-groups all --workers 4 --save-valid-preds")


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
    "TRAIN = ' '.join(buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet'))\n"
    "VALID = buscar('valid.parquet')\n"
    "INIT = buscar('ftv_init_LR_swa.pt')\n"
    "os.makedirs('/kaggle/working/results', exist_ok=True)\n"
    "\n"
    "import pyarrow.parquet as pq\n"
    "pf = pq.ParquetFile(TRAIN.split()[0])\n"
    "t0 = time.time()\n"
    "for g in range(15):\n"
    "    pf.read_row_group(g, use_threads=False)\n"
    "seg = (time.time() - t0) / 15\n"
    "print(f'velocidad de lectura: {seg:.2f} s por secuencia')\n"
    "assert seg < 1.0, f'LECTURA LENTA: {seg:.1f} s por secuencia. Volver a lanzar (suerte de maquina).'\n"
    "print('Inputs OK')\n")


def humo(k):
    return (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
            f"--train-groups 24 --valid-groups 30 --output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 "
            f"--layers 2 --lr 1e-3 --seed 1 --model-seed 1 --init-from $INIT --lookahead {k} --run-name smoke\n"
            "import json\n"
            "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
            f"assert m['config']['lookahead'] == {k}, 'sin lookahead'\n"
            "assert 'valid' not in ' '.join(m['config']['train_parquet']), 'el maestro no puede ver valid'\n"
            "print('Prueba de humo OK | WP', round(m['best_valid_wp'], 4))\n")


def correr(k):
    nombre = f"latr{k}"
    return (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT --lookahead {k} {RECETA} "
            f"--seed {3300 + k} --model-seed {3300 + k} --run-name {nombre} || echo 'FALLO {nombre}'\n"
            f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt /kaggle/working/results/{nombre}/valid_preds_best.npz\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4), '| curva', h)\n")

for k in [int(a) for a in sys.argv[1:]] or [25, 10]:
    nombre = f"connectome_sprint33_latr{k}"
    titulo = (f"# Connectome — sprint 33: maestro que mira {k} filas adelante, entrenado SOLO con train\n\n"
              "**Inputs:** train part1/part2, `Connectome valid`, `Connectome ftv`. GPU. **Save & Run All.**\n\n"
              "Los maestros del sprint 27 se entrenaron con valid y sus etiquetas hicieron daño en el LB (S14: 0.65307): "
              "memorizaron el futuro de esas mismas secuencias. Este maestro parte de la LR y aprende a mirar adelante "
              "con 2000 secuencias de train; nunca ve valid, así que sus predicciones sobre valid son etiquetas honestas "
              "y la prueba de 5 pliegues que las use es limpia.")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo(k)), md(f"## latr{k}"), code(correr(k)),
             md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
