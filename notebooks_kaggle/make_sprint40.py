import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

RECETA = ("--dropout 0.1 --lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 "
          "--distill-weight 4 --unscored-weight 0.1 --train-groups all --workers 4 --layers 2 --hidden 128 "
          "--cell lstm --soft-npz $SOFT")
CLASES = [0, 1, 2, 3]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def buscador(desde_pliegue):
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
        + ("".join(f"INIT_{f} = buscar('ftv26_LR_f{f}.pt')\n" for f in range(5)) if desde_pliegue else
           "".join(f"INIT_{f} = buscar('ftv_init_LR_swa.pt')\n" for f in range(5))) +
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


HUMO = ("!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
        "--train-groups 60 --valid-groups all --cv-folds 5 --cv-fold 0 --output-dir /kaggle/tmp/smoke "
        "--epochs 1 --cell lstm --hidden 128 --layers 2 --lr 2e-4 --seed 1 --model-seed 1 "
        "--init-from $INIT_0 --soft-npz $SOFT --distill-weight 4 --unscored-weight 0.1 "
        "--train-classes 1 --valid-classes 1 --run-name smoke\n"
        "import json\n"
        "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
        "assert m['config']['init_from'], 'no inicializó'\n"
        "assert m['config']['soft_npz'], 'sin etiquetas blandas'\n"
        "assert m['config']['train_classes'] == [1], 'sin filtro de instrumento'\n"
        "assert not set(m['config']['train_ids']) & set(m['config']['valid_ids']), 'train y valid se pisan'\n"
        "print('Prueba de humo OK | train', len(m['config']['train_ids']), '| valid', len(m['config']['valid_ids']),\n"
        "      '| WP', round(m['best_valid_wp'], 4))\n")


def correr(nombre, c, f, seed):
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT_{f} {RECETA} "
            f"--cv-folds 5 --cv-fold {f} --valid-groups all --train-classes {c} --valid-classes {c} "
            f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"
            f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| valid', len(m['config']['valid_ids']),\n"
    "          '| curva', h)\n")

INTRO = ("**Inputs:** `Connectome valid`, `Connectome ftv`" + "{extra}" + ". GPU. **Save & Run All.**\n\n"
         "## Qué se prueba\n\n"
         "Especialistas por instrumento con la receta del ajuste en valid (S13): la LR se ajusta solo con las "
         "secuencias de un instrumento, 5 pliegues de valid (cada secuencia se mide con un modelo que no la vio). "
         "Se compara contra la LR ajustada con todos los instrumentos (26A) en las mismas secuencias. "
         "Sin destilar, el especialista ganaba +0.0011 / +0.0022 en los instrumentos 0 y 1 (sprint 25). "
         "En la entrega no cuesta tiempo: cada secuencia usa las redes de su instrumento.\n\n")

NBS = {
    "connectome_sprint40a_spec": (
        "# Connectome — sprint 40A: especialistas por instrumento desde la LR base\n\n"
        + INTRO.format(extra="") + "**Variante A:** parte de la LR base (la misma de la que parte el ajuste 26A).",
        False, "ftv40_LR", 4000),
    "connectome_sprint40b_spec": (
        "# Connectome — sprint 40B: especialistas por instrumento desde la LR ya ajustada\n\n"
        + INTRO.format(extra=" y `Connectome ftv folds`") +
        "**Variante B:** parte de la LR ajustada con todo valid del pliegue (26A) y la especializa con 6 épocas más.",
        True, "ftv40b_LR", 4100),
}

for nombre, (titulo, desde_pliegue, pre, base_seed) in NBS.items():
    cells = [md(titulo), code(buscador(desde_pliegue)),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(HUMO)]
    corridas = [(f"{pre}_c{c}_f{f}", c, f, base_seed + 10 * c + f) for c in CLASES for f in range(5)]
    for run, c, f, seed in corridas:
        cells += [md(f"## {run}"), code(correr(run, c, f, seed))]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", len(corridas), "corridas")
