import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

MAESTROS = ["mask_swa", "LR_swa", "LA_best", "long_best"]
PADRE = ("--epochs 3 --batch 64 --chunk 1024 --lr-schedule cosine --lr 1.5e-3 --lr-min 1e-4 --weight-decay 1e-5 --clip 1.0 "
         "--real-weight 0 --distill-weight 1 --weight-floor 0.2 --distill-teachers $MAESTROS --distill-cache '' "
         "--train-groups all --valid-groups all --workers 4")
AJUSTE = ("--lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 --distill-weight 4 "
          "--unscored-weight 0.1 --train-groups all --workers 4 --soft-npz $SOFT")
ARQ = {"a": ("fusion", "--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64", 4701),
       "b": ("lstm", "--cell lstm --layers 2 --hidden 128 --dropout 0.1", 4702)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


BUSCADOR = (
    "import os, time, glob, sys, subprocess, torch\n"
    "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
    "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
    "try:\n"
    "    import numba\n"
    "except ImportError:\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numba'], check=True)\n"
    "    import numba\n"
    "print('numba', numba.__version__)\n"
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
    "SOFT = buscar('soft_s10_valid.npz')\n"
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

RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n")

for parte, (nombre, arq, semilla) in ARQ.items():
    padre = f"corto_{nombre}"
    humo = (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {arq} {PADRE.replace('--train-groups all --valid-groups all', '--train-groups 16 --valid-groups 8').replace('--epochs 3', '--epochs 1')} "
            "--seed 1 --model-seed 1 --output-dir /kaggle/tmp/smoke --run-name padre\n"
            f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID {arq} {AJUSTE.replace('--train-groups all', '--train-groups 30').replace('--epochs 6', '--epochs 1').replace('--swa-start 6', '--swa-start 0')} "
            "--cv-folds 5 --cv-fold 0 --valid-groups 8 --seed 1 --model-seed 1 --init-from /kaggle/tmp/smoke/padre/last_model.pt "
            "--output-dir /kaggle/tmp/smoke --run-name ajuste\n"
            "import json\n"
            "p = json.load(open('/kaggle/tmp/smoke/padre/metrics.json'))\n"
            "a = json.load(open('/kaggle/tmp/smoke/ajuste/metrics.json'))\n"
            "assert p['config']['real_weight'] == 0 and len(p['config']['distill_teachers']) == 4, 'el padre no usa solo etiquetas blandas'\n"
            "assert a['config']['init_from'] and a['config']['soft_npz'], 'el ajuste no parte del padre'\n"
            + ("!python -c \"import train_official_gpu_v1 as T; assert T._fusion_norm is not None; print('entradas rapidas OK')\"\n" if nombre == "fusion" else "")
            + "print('Prueba de humo OK | WP', round(p['best_valid_wp'], 4), round(a['best_valid_wp'], 4))\n")
    titulo = (f"# Connectome — sprint 47{parte.upper()}: padre corto ({nombre}) con etiquetas puras\n\n"
              "**Inputs:** `connectome-train-part1/part2`, `connectome-valid`, `connectome-soft-s10`, `connectome-fusion`. "
              "GPU. **Save & Run All.**\n\n"
              "La receta de padres del competidor de 0.6645: 3 épocas, coseno, lotes de 64, trozos de 1.024 filas y solo "
              "etiquetas de maestros, sin target real. Después, la prueba limpia de 5 pliegues con la receta de S13.")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo),
             md(f"## {padre}"),
             code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {arq} {PADRE} "
                  f"--seed {semilla} --model-seed {semilla} --output-dir /kaggle/working/results --run-name {padre} || echo 'FALLO {padre}'\n"
                  f"!rm -f /kaggle/working/results/{padre}/checkpoint.pt\n")]
    for f in range(5):
        corrida = f"{padre}_f{f}"
        cells += [md(f"## {corrida}"),
                  code(f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID {arq} {AJUSTE} "
                       f"--cv-folds 5 --cv-fold {f} --valid-groups all --seed {semilla + 10 + f} --model-seed {semilla + 10 + f} "
                       f"--init-from /kaggle/working/results/{padre}/last_model.pt --output-dir /kaggle/working/results "
                       f"--run-name {corrida} || echo 'FALLO {corrida}'\n"
                       f"!rm -f /kaggle/working/results/{corrida}/checkpoint.pt\n")]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint47{parte}_padre_corto.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
