import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

RECETA = ("--dropout 0.1 --lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 "
          "--unscored-weight 0.1 --train-groups all --workers 4 --layers 2 --hidden 128 --cell lstm --soft-npz $SOFT")
PARTES = {
    "a": [("ftv42_LR_c3_dw9", 3, "base", 9), ("ftv42_LR_c3_dw99", 3, "base", 99)],
    "b": [("ftv42b_LR_c0_dw1", 0, "pliegue", 1), ("ftv42b_LR_c1_dw1", 1, "pliegue", 1),
          ("ftv42b_LR_c2_dw1", 2, "pliegue", 1)],
}


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
    "SOFT = buscar('soft_s10_valid.npz')\n"
    "INIT_base = buscar('ftv_init_LR_swa.pt')\n"
    + "".join(f"INIT_f{f} = buscar('ftv26_LR_f{f}.pt')\n" for f in range(5)) +
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
        "--init-from $INIT_f0 --soft-npz $SOFT --distill-weight 9 --unscored-weight 0.1 "
        "--train-classes 3 --valid-classes 3 --run-name smoke\n"
        "import json\n"
        "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
        "assert m['config']['distill_weight'] == 9, 'peso de destilación'\n"
        "assert m['config']['train_classes'] == [3], 'sin filtro de instrumento'\n"
        "assert not set(m['config']['train_ids']) & set(m['config']['valid_ids']), 'train y valid se pisan'\n"
        "print('Prueba de humo OK | train', len(m['config']['train_ids']), '| WP', round(m['best_valid_wp'], 4))\n")


def correr(nombre, c, origen, dw, f, seed):
    init = "$INIT_base" if origen == "base" else f"$INIT_f{f}"
    return (f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from {init} {RECETA} --distill-weight {dw} "
            f"--cv-folds 5 --cv-fold {f} --valid-groups all --train-classes {c} --valid-classes {c} "
            f"--seed {seed} --model-seed {seed} --run-name {nombre}_f{f} || echo 'FALLO {nombre}_f{f}'\n"
            f"!rm -f /kaggle/working/results/{nombre}_f{f}/checkpoint.pt\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n")

for parte, grupos in PARTES.items():
    titulo = (f"# Connectome — sprint 42{parte.upper()}: cuánto target real quiere cada instrumento\n\n"
              "**Inputs:** `Connectome valid`, `Connectome ftv` y `Connectome ftv folds`. GPU. **Save & Run All.**\n\n"
              "Especialistas con la receta de S13 cambiando solo la mezcla del target (`--distill-weight`: 9 = 10 % "
              "real, 99 = 1 %, 1 = 50 %; S13 usa 4 = 20 %). Prueba limpia de 5 pliegues. Medido hasta ahora: más "
              "target real hunde el instrumento 3 y mejora un poco el 0 y el 1. "
              + ("Instrumento 3, partiendo de la LR base (como 40A)." if parte == "a" else
                 "Instrumentos 0, 1 y 2, partiendo de la LR ajustada del pliegue (como 40B)."))
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(HUMO)]
    n = 0
    for gi, (nombre, c, origen, dw) in enumerate(grupos):
        for f in range(5):
            cells += [md(f"## {nombre}_f{f}"), code(correr(nombre, c, origen, dw, f, 4200 + 100 * gi + 10 * c + f))]
            n += 1
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint42{parte}_mezcla.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", n, "corridas")
