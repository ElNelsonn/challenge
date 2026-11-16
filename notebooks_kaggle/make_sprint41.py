import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

DESDE_S17 = [int(c) for c in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
CELDA = {"E_swa": "gru", "X_best": "gru", "Z_swa": "gru", "Y_best": "gru", "LR_swa": "lstm", "long_best": "lstm"}
PARTES = {"a": ["E_swa", "X_best", "LR_swa"], "b": ["Z_swa", "Y_best", "long_best"]}
CLASES = [0, 1, 2, 3]
SEMILLAS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "41"
RECETA = ("--dropout 0.1 --lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 "
          "--distill-weight 4 --unscored-weight 0.1 --train-groups all --workers 4 --layers 2 --hidden 128 "
          "--soft-npz $SOFT --cv-folds 5 --cv-fold -1 --valid-groups 60")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def buscador(miembros):
    return (
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
        "SOFT = buscar('soft_s10_valid.npz')\n"
        + "".join(f"INIT_{m} = buscar('ftv_init_{m}.pt')\nINIT17_{m} = buscar('ftv17_{m}.pt')\n" for m in miembros) +
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


def humo(m):
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--train-groups 60 --valid-groups 30 --cv-folds 5 --cv-fold -1 --output-dir /kaggle/tmp/smoke "
            f"--epochs 1 --cell {CELDA[m]} --hidden 128 --layers 2 --lr 2e-4 --seed 1 --model-seed 1 "
            f"--init-from $INIT17_{m} --soft-npz $SOFT --distill-weight 4 --unscored-weight 0.1 "
            f"--train-classes 3 --valid-classes 3 --run-name smoke\n"
            "import json\n"
            "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
            "assert m['config']['init_from'], 'no inicializó'\n"
            "assert m['config']['soft_npz'], 'sin etiquetas blandas'\n"
            "assert m['config']['train_classes'] == [3], 'sin filtro de instrumento'\n"
            "print('Prueba de humo OK | train', len(m['config']['train_ids']), '| WP', round(m['best_valid_wp'], 4))\n")


def correr(nombre, m, c, seed):
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT{'17' if c in DESDE_S17 else ''}_{m} "
            f"--cell {CELDA[m]} {RECETA} "
            f"--train-classes {c} --valid-classes {c} --seed {seed} --model-seed {seed} --run-name {nombre} "
            f"|| echo 'FALLO {nombre}'\n"
            f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n")

for parte, miembros in PARTES.items():
    titulo = (f"# Connectome — sprint {TAG}{parte.upper()}: especialistas por instrumento para la entrega\n\n"
              "**Inputs:** `Connectome valid`, `Connectome ftv` y `Connectome ftv17`. GPU. **Save & Run All.**\n\n"
              f"Receta de S13 con todo valid, pero cada corrida solo con las secuencias de un instrumento. "
              f"Miembros: {', '.join(miembros)}; 4 instrumentos × semillas {SEMILLAS}. Los instrumentos {DESDE_S17} parten "
              "de los miembros ya ajustados de S17 (como 40B); el resto, de los modelos base (como 40A). "
              "Es la combinación que ganó en la prueba limpia: +0.0009.")
    cells = [md(titulo), code(buscador(miembros)),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo(miembros[0]))]
    corridas = [(f"ftv41_{m}_c{c}_s{k}", m, c, 4100 + 10 * c + k) for m in miembros for c in CLASES for k in SEMILLAS]
    for run, m, c, seed in corridas:
        cells += [md(f"## {run}"), code(correr(run, m, c, seed))]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint{TAG}{parte}_spec_full.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", len(corridas), "corridas | desde S17:", DESDE_S17)
