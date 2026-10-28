import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

SOFT = sys.argv[1] if len(sys.argv) > 1 else "soft_s10_valid.npz"
TAG = sys.argv[2] if len(sys.argv) > 2 else "26"
PRE = "ftv_" if TAG == "26" else f"ftv{TAG}_"
LR = sys.argv[3] if len(sys.argv) > 3 else "2e-4"
DW = sys.argv[4] if len(sys.argv) > 4 else "4"

RECETA = (f"--dropout 0.1 --lr {LR} --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 "
          f"--distill-weight {DW} --unscored-weight 0.1 --train-groups all --workers 4 --layers 2 --hidden 128")
CELDA = {"LR_swa": "lstm", "mask_swa": "lstm", "LA_best": "lstm", "long_best": "lstm", "E_swa": "gru", "X_best": "gru",
         "Z_swa": "gru", "Y_best": "gru"}
MIEMBROS_B = sys.argv[5].split(",") if len(sys.argv) > 5 else None


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
    f"SOFT = buscar('{SOFT}')\n"
    "for m in " + repr(list(CELDA)) + ":\n"
    "    globals()['INIT_' + m] = buscar(f'ftv_init_{m}.pt')\n"
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
            f"--train-groups 24 --valid-groups 60 --cv-folds 5 --cv-fold 0 --output-dir /kaggle/tmp/smoke "
            f"--epochs 1 --cell {CELDA[m]} --hidden 128 --layers 2 --lr 2e-4 --seed 1 --model-seed 1 "
            f"--init-from $INIT_{m} --soft-npz $SOFT --distill-weight 4 --unscored-weight 0.1 --run-name smoke\n"
            "import json\n"
            "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
            "assert m['config']['init_from'], 'no inicializó'\n"
            "assert m['config']['soft_npz'], 'sin etiquetas blandas'\n"
            "assert m['config']['unscored_weight'] == 0.1, 'sin peso de filas puntuadas'\n"
            "assert not set(m['config']['train_ids']) & set(m['config']['valid_ids']), 'train y valid se pisan'\n"
            "print('Prueba de humo OK | train', len(m['config']['train_ids']), '| valid', len(m['config']['valid_ids']),\n"
            "      '| WP', round(m['best_valid_wp'], 4))\n")


def correr(nombre, m, extra, seed):
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT_{m} --soft-npz $SOFT --cell {CELDA[m]} "
            f"{RECETA} {extra} --seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"
            f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| valid', len(m['config']['valid_ids']), '| SWA(=última)',\n"
    "          round(m.get('swa_valid_wp', float('nan')), 4), '| curva', h)\n")

INTRO = ("**Inputs:** `Connectome valid` y `Connectome ftv` (etiquetas blandas + checkpoints de partida). GPU. "
         "**Save & Run All.**\n\n"
         "## Qué se prueba\n\n"
         "Ajuste fino de los miembros del ensamble **con las secuencias de valid**, pesando 1 las filas que puntúa "
         "la métrica y 0,1 el resto, y con target 0,2 × real + 0,8 × etiqueta blanda (`" + SOFT + "`, de modelos que "
         "nunca vieron esa secuencia). 6 épocas, lr 2e-4, weight decay 1e-4. Un competidor documentó +0.007 en el leaderboard con esta "
         "receta; train y valid son regímenes de mercado distintos y el test se parece a valid.\n\n")

LR_FOLDS = [(f"{PRE}LR_f{k}", "LR_swa", f"--cv-folds 5 --cv-fold {k} --valid-groups all --save-valid-preds", 2600 + k)
            for k in range(5)]
LR_FULL = [(f"{PRE}LR_all_s{s}", "LR_swa", "--cv-folds 5 --cv-fold -1 --valid-groups 60", 2610 + s) for s in (1, 2)]
RESTO = ([(f"{PRE}{m}_all_s1", m, "--cv-folds 5 --cv-fold -1 --valid-groups 60", 2621)
          for m in ("mask_swa", "LA_best", "long_best", "E_swa", "X_best")]
         + [(f"{PRE}{m}_all_s2", m, "--cv-folds 5 --cv-fold -1 --valid-groups 60", 2622)
            for m in ("mask_swa", "LA_best", "long_best")])
if MIEMBROS_B:
    RESTO = [(f"{PRE}{m}_all_s{k}", m, "--cv-folds 5 --cv-fold -1 --valid-groups 60", 2620 + k)
             for k in (1, 2) for m in MIEMBROS_B]

NBS = {
    f"connectome_sprint{TAG}a_ftv_lr": (
        f"# Connectome — sprint {TAG}A: ajuste fino en valid, validación cruzada de la LR\n\n" + INTRO +
        "Cinco corridas dejando afuera un quinto de valid cada vez: cada secuencia se mide con un modelo que no la "
        "vio. Después, dos ajustes con todo valid (semillas distintas) para la entrega.",
        "LR_swa", LR_FOLDS + LR_FULL),
    f"connectome_sprint{TAG}b_ftv_resto": (
        f"# Connectome — sprint {TAG}B: ajuste fino en valid del resto de los miembros\n\n" + INTRO +
        "mask, LA y long (LSTM de S10) y E, X (GRU de S7), con todo valid.",
        "E_swa", RESTO),
}

for nombre, (titulo, humo_m, corridas) in NBS.items():
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo(humo_m))]
    for run, m, extra, seed in corridas:
        cells += [md(f"## {run}"), code(correr(run, m, extra, seed))]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", ", ".join(r for r, _, _, _ in corridas))
