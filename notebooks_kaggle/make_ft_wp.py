r"""Genera connectome_ft_wp.ipynb: fine-tuning de Eswa (2 épocas, lr 1e-4) con tres losses, para responder
"¿entrenar con 1 − WP mejora a un modelo ya bueno?".

  control_ft : misma loss de siempre (WMSE alineada)      -> mide cuánto suma solo "seguir entrenando"
  gwp_ft     : 1 − WP GLOBAL del batch (como la métrica)   -> la pregunta
  wp_ft      : 1 − WP por secuencia                        -> variante

~25 min cada una en T4 (~1,3 h total). Uso (desde challenge/):  python notebooks_kaggle/make_ft_wp.py
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
ESWA = f"{MODELS}/gru128_distill_seed2029_swa.pt"
RUNS = [("control_ft", "wmse"), ("gwp_ft", "gwp"), ("wp_ft", "wp")]
COMMON = (f"--train-parquet {TRAIN} --valid-parquet {VALID} --train-groups all --valid-groups all --output-dir /kaggle/working/results "
          f"--save-valid-preds --epochs 2 --lr 1e-4 --lr-decay 0.9 --hidden 128 --layers 2 --dropout 0.1 --weight-decay 1e-5 "
          f"--seed 2060 --model-seed 2060 --init-from {ESWA}")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + Path(path).read_text(encoding="utf-8"))


cells = [
    md("# Connectome — ¿entrenar con 1 − WP? (fine-tuning de Eswa)\n\n"
       "**Inputs:** train part1/part2, valid, `Connectome models` (v3 o v4). GPU T4. **Save & Run All.** ~1,3 h.\n\n"
       "Las tres corridas parten de **Eswa (0.6805)** y entrenan 2 épocas más con lr 1e-4, mismos datos y semilla; solo cambia la loss:\n\n"
       "| corrida | loss | pregunta |\n|---|---|---|\n"
       "| control_ft | WMSE alineada (la de siempre) | ¿cuánto suma solo seguir entrenando? |\n"
       "| gwp_ft | 1 − WP global del batch (= la métrica) | ¿1 − WP mejora? |\n"
       "| wp_ft | 1 − WP por secuencia | variante |\n\n"
       "Se compara cada una contra `control_ft`, no contra Eswa."),
    code("import os, torch\n"
         "assert torch.cuda.is_available(), 'Activá GPU'\n"
         "print('GPU:', torch.cuda.get_device_name(0))\n"
         f"for p in {TRAIN.split()} + ['{VALID}', '{ESWA}']:\n"
         "    assert os.path.isfile(p), f'Falta input: {p}'\n"
         "os.makedirs('/kaggle/working/results', exist_ok=True); print('Inputs OK')\n"),
    writefile("utils.py", ROOT / "utils.py"),
    writefile("train_official_gpu_v1.py", CH / "model" / "official_gpu" / "train_official_gpu_v1.py"),
]
for name, loss in RUNS:
    cells.append(md(f"## {name} — `--loss {loss}`"))
    cells.append(code(f"!python train_official_gpu_v1.py {COMMON} --loss {loss} --run-name {name} || echo 'FALLÓ {name}'\n"))
cells.append(md("## Resumen"))
cells.append(code(
    "import json, os\n"
    "print('Eswa (punto de partida): 0.6805')\n"
    f"for name, loss in {RUNS}:\n"
    "    f = f'/kaggle/working/results/{name}/metrics.json'\n"
    "    if os.path.isfile(f):\n"
    "        m = json.load(open(f))\n"
    "        curva = ' / '.join(f\"{h['wp']:.4f}\" for h in m['history'])\n"
    "        print(f'{name:12s} loss={loss:5s}  mejor {m[\"best_valid_wp\"]:.4f}   épocas: {curva}')\n"
    "    else:\n"
    "        print(name, 'FALLÓ')\n"))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "connectome_ft_wp.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok:", HERE / "connectome_ft_wp.ipynb")
