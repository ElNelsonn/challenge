import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

INIT = "lstm128x2_studentLR_seed2072_best.pt"
RECETA = ("--cell lstm --hidden 128 --layers 2 --dropout 0.1 --weight-decay 1e-5 --lr 2e-4 --lr-decay 0.7 "
          "--epochs 3 --swa-start 2 --train-groups all --valid-groups all --save-valid-preds")
CUANTOS = {0: 0.1254, 1: 0.2413, 2: 0.4685, 3: 0.6569}


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
    f"INIT = buscar('{INIT}')\n"
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


def humo(clases):
    return (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
            f"--train-groups 64 --valid-groups 40 --output-dir /kaggle/tmp/smoke --epochs 1 "
            f"--cell lstm --hidden 128 --layers 2 --lr 2e-4 --seed 1 --model-seed 1 --init-from $INIT "
            + (f"--train-classes {clases} --valid-classes {clases} " if clases != "" else "")
            + "--run-name smoke\n"
            "import json\n"
            "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
            f"assert m['config']['init_from'], 'no inicializo desde LR'\n"
            + (f"assert m['config']['train_classes'] == [{clases}], 'filtro de instrumento equivocado'\n"
               f"assert 0 < len(m['config']['train_ids']) < 64, 'el filtro no saco secuencias'\n" if clases != "" else "")
            + "print('Prueba de humo OK | secuencias:', len(m['config']['train_ids']), '| valid WP:', round(m['best_valid_wp'], 4))\n")


def correr(nombre, extra, seed):
    return (f"!python train_official_gpu_v1.py --train-parquet $TRAIN --valid-parquet $VALID "
            f"--output-dir /kaggle/working/results --init-from $INIT {RECETA} {extra} "
            f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n")


RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| clases', m['config'].get('valid_classes'), '| mejor', round(m['best_valid_wp'], 4),\n"
    "          '| SWA', round(m.get('swa_valid_wp', float('nan')), 4), '| curva', h)\n")

INTRO = ("**Inputs:** train part1/part2, valid, `Connectome models`. GPU. **Save & Run All.**\n\n"
         "## El hallazgo (2026-10-26)\n\n"
         "Los datos mezclan **4 instrumentos distintos**. Se reconocen por el cuanto del target "
         "(0.1254 / 0.2413 / 0.4685 / 0.6569) y, en inferencia, por los valores exactos de las features "
         "(100 % de acierto desde la primera fila). Nuestro ensamble puntúa muy distinto en cada uno:\n\n"
         "| instrumento | 0 | 1 | 2 | 3 |\n|---|---|---|---|---|\n"
         "| puntaje en valid | 0.799 | 0.771 | 0.729 | **0.435** |\n"
         "| peso en la métrica | 28 % | 16 % | 8 % | **47 %** |\n"
         "| secuencias en train | 12 % | 12 % | 33 % | 43 % |\n\n")

NBS = {
    "connectome_sprint25_ft_clases": (
        "# Connectome — sprint 25A: un especialista por instrumento\n\n" + INTRO +
        "## Qué hace\n\nParte de la alumna **LR** (0.6834) y la ajusta **3 épocas con lr 2e-4 solo con las secuencias de un "
        "instrumento**, una vez por instrumento. Cada especialista se evalúa solo en su instrumento. En inferencia "
        "cada secuencia usa solo el especialista de su instrumento, así que **el tiempo no cambia**.\n\n"
        "El notebook hermano (25B) hace exactamente lo mismo con todos los instrumentos juntos: es el control "
        "para separar el efecto de especializar del efecto de entrenar 3 épocas más.",
        "3",
        [(f"ft_LR_c{c}", f"--train-classes {c} --valid-classes {c}", 2500 + c) for c in range(4)]),
    "connectome_sprint25_ft_control": (
        "# Connectome — sprint 25B: el control (todos los instrumentos juntos)\n\n" + INTRO +
        "## Qué hace\n\nLa misma receta que el 25A (LR + 3 épocas con lr 2e-4) pero con **todas** las secuencias. "
        "Si los especialistas del 25A le ganan a esto en su instrumento, la ganancia es de especializar.",
        "",
        [("ft_LR_all", "", 2510)]),
}

for nombre, (titulo, humo_clases, corridas) in NBS.items():
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo(humo_clases))]
    for run, extra, seed in corridas:
        cells += [md(f"## {run}"), code(correr(run, extra, seed))]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{nombre}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", ", ".join(r for r, _, _ in corridas))
