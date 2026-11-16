import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
OG = CH / "model" / "official_gpu"

CAUSALES = ["mask_swa", "LR_swa", "LA_best", "long_best"]
RECETA = ("--epochs 3 --batch 64 --chunk 1024 --lr-schedule cosine --lr 1.5e-3 --lr-min 1e-4 --weight-decay 1e-5 --clip 1.0 "
          "--weight-floor 0.2 --train-groups all --valid-groups all --workers 4")
ALUMNOS = {"lstm": ("--cell lstm --layers 2 --hidden 128 --dropout 0.1", 4702),
           "fusion": ("--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64", 4701)}
PARTES = {"a": (25, "1 2 4 8 9", 4801, ["lstm"]),
          "b": (50, "1 2 4 8 16 18", 4802, ["lstm", "fusion"])}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


BUSCADOR = (
    "import os, time, glob, sys, shutil, subprocess, torch\n"
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
    "PARTES = [buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet')]\n"
    "VALID = buscar('valid.parquet')\n"
    "CAUSALES = ' '.join(buscar(f'ftv_init_{m}.pt') for m in " + repr(CAUSALES) + ")\n"
    "os.makedirs('/kaggle/working/results', exist_ok=True)\n"
    "LOCAL = '/kaggle/tmp/datos'\n"
    "os.makedirs(LOCAL, exist_ok=True)\n"
    "tam = sum(os.path.getsize(p) for p in PARTES + [VALID]) / 1e9\n"
    "libre = shutil.disk_usage(LOCAL).free / 1e9\n"
    "print(f'datos {tam:.1f} GB, disco local libre {libre:.1f} GB')\n"
    "assert libre > tam + 3, 'no entra en el disco local'\n"
    "t0 = time.time()\n"
    "locales = []\n"
    "for p in PARTES + [VALID]:\n"
    "    d = os.path.join(LOCAL, os.path.basename(p))\n"
    "    shutil.copyfile(p, d)\n"
    "    locales.append(d)\n"
    "    print(f'    copiado {os.path.basename(p)} ({(time.time() - t0) / 60:.1f} min)', flush=True)\n"
    "TRAIN = ' '.join(locales[:2])\n"
    "VALID = locales[2]\n"
    "print('Inputs OK en disco local')\n")

RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n")

for parte, (K, dil, semilla, alumnos) in PARTES.items():
    maestro = f"maestro_la{K}"
    arq_m = f"--arch lookahead --hidden 128 --layers 2 --la-canales 64 --la-dil {dil}"
    humo_rec = RECETA.replace("--train-groups all --valid-groups all", "--train-groups 16 --valid-groups 8").replace("--epochs 3", "--epochs 1")
    humo = (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {arq_m} {humo_rec} "
            "--seed 1 --model-seed 1 --output-dir /kaggle/tmp/smoke --run-name maestro\n"
            "!python etiquetas_la.py --parquet $TRAIN --maestros-la /kaggle/tmp/smoke/maestro/last_model.pt "
            "--maestros-causales $CAUSALES --grupos 16 --seed 1 --batch 16 --salida /kaggle/tmp/smoke/soft.npz\n")
    for a in alumnos:
        humo += (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {ALUMNOS[a][0]} {humo_rec} "
                 "--real-weight 0 --distill-weight 1 --soft-npz /kaggle/tmp/smoke/soft.npz "
                 f"--seed 1 --model-seed 1 --output-dir /kaggle/tmp/smoke --run-name alumno_{a}\n")
    humo += ("import json\n"
             "m = json.load(open('/kaggle/tmp/smoke/maestro/metrics.json'))\n"
             f"assert m['config']['arch'] == 'lookahead' and m['config']['la_dil'] == [{dil.replace(' ', ', ')}], 'el maestro no es el esperado'\n"
             f"a = json.load(open('/kaggle/tmp/smoke/alumno_{alumnos[0]}/metrics.json'))\n"
             "assert a['config']['real_weight'] == 0 and a['config']['soft_npz'], 'el alumno no usa solo etiquetas blandas'\n"
             "print('Prueba de humo OK | maestro', round(m['best_valid_wp'], 4), '| alumno', round(a['best_valid_wp'], 4))\n")
    titulo = (f"# Connectome — sprint 48{parte.upper()}: maestro que lee {K} filas futuras\n\n"
              "**Inputs:** `connectome-train-part1/part2`, `connectome-valid`, `connectome-fusion`. GPU. **Save & Run All.**\n\n"
              "Copia los datos al disco local, entrena el maestro (GRU causal + lector convolucional del futuro) y un alumno "
              "de 3 épocas con mitad etiquetas de maestros causales y mitad del lector. La medida es el alumno base en valid, "
              "que no toca valid al entrenar: LSTM contra 0.6819 (47B).")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" + (OG / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             code("%%writefile etiquetas_la.py\n" + (OG / "etiquetas_la.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo),
             md(f"## {maestro}"),
             code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {arq_m} {RECETA} "
                  f"--seed {semilla} --model-seed {semilla} --output-dir /kaggle/working/results --run-name {maestro} || echo 'FALLO {maestro}'\n"
                  f"!rm -f /kaggle/working/results/{maestro}/checkpoint.pt\n"),
             md("## Etiquetas"),
             code(f"!python etiquetas_la.py --parquet $TRAIN --maestros-la /kaggle/working/results/{maestro}/last_model.pt "
                  "--maestros-causales $CAUSALES --batch 16 --salida /kaggle/tmp/soft_mezcla.npz\n")]
    for a in alumnos:
        arq, sem = ALUMNOS[a]
        corrida = f"corto_{a}_la{K}"
        cells += [md(f"## {corrida}"),
                  code(f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID {arq} {RECETA} "
                       "--real-weight 0 --distill-weight 1 --soft-npz /kaggle/tmp/soft_mezcla.npz "
                       f"--seed {sem} --model-seed {sem} --output-dir /kaggle/working/results --run-name {corrida} || echo 'FALLO {corrida}'\n"
                       f"!rm -f /kaggle/working/results/{corrida}/checkpoint.pt\n")]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint48{parte}_lector.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
