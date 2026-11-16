import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

MAESTROS = ["mask_swa", "LR_swa", "LA_best", "long_best"]
RECETA = ("--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64 --lr 5e-4 --lr-decay 0.9 "
          "--weight-decay 1e-5 --clip 1.0 --epochs 10 --swa-start 7 --batch 32 --chunk 512 --distill-weight 1.0 "
          "--distill-cache '' --train-groups all --valid-groups all --workers 4 --save-valid-preds")
SEMILLAS = {"a": 4401, "b": 4402}


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
    "MAESTROS = ' '.join(buscar(f'ftv_init_{m}.pt') for m in " + repr(MAESTROS) + ")\n"
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

HUMO = ("!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID --train-groups 16 --valid-groups 8 "
        "--epochs 1 --arch fusion --fusion-features --hidden 128 --lr 5e-4 --seed 1 --model-seed 1 "
        "--distill-teachers $MAESTROS --distill-cache '' --output-dir /kaggle/tmp/smoke --run-name smoke\n"
        "import json\n"
        "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
        "assert m['config']['arch'] == 'fusion' and m['config']['fusion_features'], 'no es la red fusión'\n"
        "assert len(m['config']['distill_teachers']) == 4, 'faltan maestros'\n"
        "print('Prueba de humo OK | WP', round(m['best_valid_wp'], 4))\n")

RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4), '| curva', h)\n")

for parte, semilla in SEMILLAS.items():
    nombre = f"fusion_s{semilla}"
    titulo = (f"# Connectome — sprint 44{parte.upper()}: la red fusión, entrenamiento base\n\n"
              "**Inputs:** `Connectome train part1/part2`, `Connectome valid`, `Connectome ftv`. GPU. **Save & Run All.**\n\n"
              "Otra estructura de red para el ensamble: las 112 columnas más su diferencia con la fila anterior y su "
              "desvío respecto de promedios móviles de 15, 127 y 1023 filas (560 entradas) → lineal a 64 → GRU de una "
              "capa de 128 → la salida mira el estado de la GRU y la proyección juntos (192 → 64 → 2). 122.882 "
              "parámetros, ~2/3 del costo de una GRU 128×2. Receta de la alumna LR: 10 épocas, SWA 7–10, destilada "
              "de los 4 modelos de S10 (entrenados solo con train). Al competidor de 0.6645 este tipo de miembro le dio "
              f"+0.0017 en el LB. Semilla {semilla}.")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(HUMO),
             md(f"## {nombre}"),
             code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
                  f"--output-dir /kaggle/working/results {RECETA} --distill-teachers $MAESTROS "
                  f"--seed {semilla} --model-seed {semilla} --run-name {nombre} || echo 'FALLO {nombre}'\n"
                  f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt\n"),
             md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint44{parte}_fusion.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
