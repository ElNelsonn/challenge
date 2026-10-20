r"""Genera los dos notebooks del sprint 10 (2026-09-29): LOSS DE CORRELACIÓN A NIVEL DE SECUENCIA.

De dónde sale: un competidor que está por encima nuestro en el leaderboard (dozken, puesto ~51, 0.6588)
publicó su repositorio. Su arquitectura es GRU 192x2 (peor que la nuestra en nuestras manos), pero su
entrenamiento usa "sequence-level weighted Pearson on soft-masked rows", y anota explícitamente que
"el Pearson por chunk se desvía de la métrica real".

Nosotros entrenamos con error cuadrático (WMSE) y la métrica es una CORRELACIÓN, que es invariante a la
escala. O sea que gastamos capacidad en acertar la magnitud, cosa que la métrica ignora. Lo habíamos
probado una vez (exp 018) pero mal: con peso 0.05 y calculado sobre trozos de 512 filas.

`--loss seqwp` acumula los estadísticos de la correlación a lo largo de TODA la secuencia (promedio
exponencial, `--seqwp-decay`), así la normalización refleja la secuencia entera y no un trozo chico.

  * connectome_sprint10_seqwp.ipynb       — LSTM 128x2 SIN destilar, loss seqwp + ancla 0.1 (~2,5 h)
  * connectome_sprint10_seqwp_dist.ipynb  — la misma receta que la alumna LR (destilada, lr 5e-4), pero
    con loss seqwp + ancla 0.2. Comparación exacta contra LR = 0.6834 (~2,5 h)

Los dos en paralelo. Requieren `connectome-models` versión 7. Uso (desde challenge/):
    python notebooks_kaggle/make_sprint10.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
TRAIN = ("/kaggle/input/datasets/joaquinfiorina/connectome-train-part1/train_part1.parquet "
         "/kaggle/input/datasets/joaquinfiorina/connectome-train-part2/train_part2.parquet")
VALID = "/kaggle/input/datasets/joaquinfiorina/connectome-valid/valid.parquet"
MODELS = "/kaggle/input/datasets/joaquinfiorina/connectome-models"
TEACHERS = {"E": "gru128_distill_seed2029_swa.pt", "l1": "lstm128x2_l1_seed2047_swa.pt",
            "LR": "lstm128x2_studentLR_seed2072_swa.pt", "LA": "lstm128x2_studentLA_best.pt"}
teacher_paths = " ".join(f"{MODELS}/{f}" for f in TEACHERS.values())
BASE = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all "
        f"--output-dir /kaggle/working/results --save-valid-preds --lr 5e-4 --lr-decay 0.9 --weight-decay 1e-5 "
        f"--cell lstm --hidden 128 --layers 2 --dropout 0.1 --epochs 8 --swa-start 5 --loss seqwp")
RUNS = {
    "seqwp": dict(extra="--wmse-anchor 0.1", seed=2101, distill=False,
                  ref="referencia: nuestras LSTM 128x2 sin destilar dan 0.679-0.681 (l1 0.6808)"),
    "seqwp_dist": dict(extra="--wmse-anchor 0.2 --distill-weight 1.0 --distill-cache /kaggle/tmp/soft_S8.npz "
                             f"--distill-teachers {teacher_paths}", seed=2102, distill=True,
                       ref="referencia EXACTA: la alumna LR, misma receta con loss wmse = 0.6834"),
}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


for name, r in RUNS.items():
    smoke_extra = ("--distill-teachers " + teacher_paths + " --distill-cache ''") if r["distill"] else ""
    cells = [
        md(f"# Connectome — sprint 10: **loss de correlación** ({name})\n\n"
           "**Inputs:** train part1/part2, valid, **`Connectome models` versión 7**. GPU T4 x2. **Save & Run All.** ~2,5 h.\n\n"
           "**Se puede correr a la vez que el otro notebook del sprint 10.**\n\n"
           "**Qué se prueba:** entrenar optimizando **la correlación**, que es lo que mide la competencia, en vez del "
           "error cuadrático. La métrica es una correlación ponderada: **no mira la escala** de las predicciones. "
           "Entrenando con error cuadrático gastamos capacidad en acertar la magnitud, que no se puntúa.\n\n"
           "De dónde sale la idea: un competidor que está arriba nuestro (puesto ~51) publicó su repositorio y usa "
           "*\"sequence-level weighted Pearson on soft-masked rows\"*, con la advertencia de que calcular el Pearson "
           "por trozos chicos **no** se parece a la métrica. Nuestro intento anterior (exp 018) cometía justo ese error: "
           "peso 0.05 y trozos de 512 filas.\n\n"
           "Ahora `--loss seqwp` acumula los estadísticos de la correlación **a lo largo de toda la secuencia**, con un "
           "olvido exponencial para que el trozo actual siga pesando. El ancla (`--wmse-anchor`) es un poco de error "
           "cuadrático para que la escala de la salida no quede totalmente libre.\n\n"
           f"| | |\n|---|---|\n| modelo | LSTM 128x2, lr 5e-4, dropout 0.1, wd 1e-5, 8 épocas, SWA 5–8, semilla {r['seed']} |\n"
           f"| loss | `seqwp` {r['extra'].split('--distill')[0].strip()} |\n"
           f"| destilación | {'sí, los 4 maestros de siempre' if r['distill'] else 'no'} |\n"
           f"| comparar contra | {r['ref']} |\n\n"
           "**Qué esperar:** si da +0.004 o más sobre su referencia, encontramos lo que nos faltaba y cambia toda la "
           "estrategia. Si da lo mismo, la diferencia del otro competidor está en otro lado."),
        code("import os, torch\n"
             "assert torch.cuda.is_available(), 'Activá GPU'\n"
             "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
             f"for p in {TRAIN.split()} + ['{VALID}']:\n"
             "    assert os.path.isfile(p), f'Falta input: {p}'\n"
             f"for f in {list(TEACHERS.values())}:\n"
             f"    assert os.path.isfile(f'{MODELS}/{{f}}'), f'Falta {{f}}: agregá el dataset Connectome models VERSIÓN 7'\n"
             "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
        writefile("utils.py", ROOT / "utils.py"),
        writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
        md("## Prueba de humo (~3 min)"),
        code("# 1 época de 32 secuencias con la loss nueva. Corta si falla.\n"
             f"!python train_official_gpu_v1.py --train-parquet {TRAIN} --valid-parquet {VALID} --train-groups 32 --valid-groups 4 "
             f"--output-dir /kaggle/tmp/smoke --epochs 1 --cell lstm --hidden 128 --layers 2 --dropout 0.1 --lr 5e-4 "
             f"--loss seqwp --wmse-anchor 0.1 --seed 1 --model-seed 1 {smoke_extra} --run-name smoke\n"
             "import os, json\n"
             "assert os.path.isfile('/kaggle/tmp/smoke/smoke/metrics.json'), 'La prueba de humo FALLÓ: revisar el error de arriba'\n"
             "print('Prueba de humo OK, valid WP:', json.load(open('/kaggle/tmp/smoke/smoke/metrics.json'))['best_valid_wp'])\n"),
        md(f"## {name}"),
        code(f"!python train_official_gpu_v1.py {BASE} {r['extra']} --seed {r['seed']} --model-seed {r['seed']} "
             f"--run-name student_{name} || echo 'FALLÓ student_{name}'\n"),
        md("## Resumen"),
        code("import json, os\n"
             f"d = '/kaggle/working/results/student_{name}'\n"
             "m = json.load(open(d + '/metrics.json')) if os.path.isfile(d + '/metrics.json') else None\n"
             "if m:\n"
             "    curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'] if h['tag'] != 'swa')\n"
             "    print('mejor', round(m['best_valid_wp'], 4), '| SWA', round(m.get('swa_valid_wp', float('nan')), 4))\n"
             "    print('curva por epoca:', curva)\n"
             f"    print('{r['ref']}')\n"
             "    print(sorted(os.listdir(d)))\n"
             "else:\n"
             "    print('FALLO: revisar el error de arriba')\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint10_{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok:", out.name)
