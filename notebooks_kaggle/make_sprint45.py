import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent

ETAPA = sys.argv[1] if len(sys.argv) > 1 else "45"
RECETA = ("--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64 --lr 2e-4 --lr-decay 1.0 "
          "--weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 --distill-weight 4 --unscored-weight 0.1 "
          "--train-groups all --workers 4 --soft-npz $SOFT")
COMPLETO = "--cv-folds 5 --cv-fold -1 --valid-groups 60"

if ETAPA == "45":
    PARTES = {
        "a": [("ftv45_fusion_all_s1", "$BASE", COMPLETO, 4501), ("ftv45_fusion_all_s2", "$BASE", COMPLETO, 4502),
              ("ftv45_fusion_c3_s1", "$BASE", COMPLETO + " --train-classes 3 --valid-classes 3", 4531),
              ("ftv45_fusion_c3_s2", "$BASE", COMPLETO + " --train-classes 3 --valid-classes 3", 4532)],
        "b": [(f"ftv45_fusion_f{f}", "$BASE", f"--cv-folds 5 --cv-fold {f} --valid-groups all", 4510 + f) for f in range(5)],
    }
    BUSCAR = ["fusion_base.pt"]
else:
    PARTES = {"a": [(f"ftv46_fusion_c{c}_s{k}", "$AJUSTADA", COMPLETO + f" --train-classes {c} --valid-classes {c}", 4600 + 10 * c + k)
                    for c in (0, 1, 2) for k in (1, 2)]}
    BUSCAR = ["fusion_ftv45.pt"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


VAR = {"fusion_base.pt": "BASE", "fusion_ftv45.pt": "AJUSTADA"}
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
    + "".join(f"{VAR[n]} = buscar('{n}')\n" for n in BUSCAR) +
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

INICIO = "$" + VAR[BUSCAR[0]]
HUMO = ("!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID --train-groups 40 --valid-groups all "
        "--cv-folds 5 --cv-fold 0 --output-dir /kaggle/tmp/smoke --epochs 1 --arch fusion --fusion-features --hidden 128 "
        f"--lr 2e-4 --seed 1 --model-seed 1 --init-from {INICIO} --soft-npz $SOFT --distill-weight 4 --unscored-weight 0.1 "
        "--valid-classes 1 --run-name smoke\n"
        "import json\n"
        "m = json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))\n"
        "assert m['config']['arch'] == 'fusion' and m['config']['init_from'], 'no parte de la red fusión'\n"
        "assert m['config']['soft_npz'], 'sin etiquetas blandas'\n"
        "print('Prueba de humo OK | WP', round(m['best_valid_wp'], 4))\n")

RESUMEN = (
    "import json, os, glob\n"
    "for d in sorted(glob.glob('/kaggle/working/results/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n")

for parte, corridas in PARTES.items():
    titulo = (f"# Connectome — sprint {ETAPA}{parte.upper()}: la red fusión ajustada en valid\n\n"
              "**Inputs:** `connectome-valid`, `connectome-soft-s10` y `connectome-fusion`. GPU. **Save & Run All.**\n\n"
              "Receta de S13 (6 épocas, lr 2e-4, 20 % target real + 80 % etiqueta blanda de S10, filas puntuadas 1 y el "
              "resto 0,1) aplicada a la red fusión. "
              + ("A: ajuste con todo valid (2 semillas) y especialistas del instrumento 3 desde la base. "
                 "B: prueba limpia de 5 pliegues." if ETAPA == "45" else
                 "Especialistas de los instrumentos 0, 1 y 2 desde la fusión ya ajustada (2 semillas)."))
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" +
                  (CH / "model" / "official_gpu" / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(HUMO)]
    for nombre, init, extra, seed in corridas:
        cells += [md(f"## {nombre}"),
                  code(f"!python train_official_gpu_v1.py --train-parquet $VALID --valid-parquet $VALID "
                       f"--output-dir /kaggle/working/results --init-from {init} {RECETA} {extra} "
                       f"--seed {seed} --model-seed {seed} --run-name {nombre} || echo 'FALLO {nombre}'\n"
                       f"!rm -f /kaggle/working/results/{nombre}/checkpoint.pt\n")]
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint{ETAPA}{parte}_fusion_ftv.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name, "|", len(corridas), "corridas")
