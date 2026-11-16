import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
OG = CH / "model" / "official_gpu"
SUB = CH / "model" / "submission"

CAUSALES = ["mask_swa", "LR_swa", "LA_best", "long_best"]
RECETA = ("--epochs 3 --batch 64 --chunk 1024 --lr-schedule cosine --lr 1.5e-3 --lr-min 1e-4 --weight-decay 1e-5 --clip 1.0 "
          "--weight-floor 0.2 --train-groups all --valid-groups all --workers 4")
AJUSTE = ("--lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 --distill-weight 4 "
          "--unscored-weight 0.1 --train-groups all --workers 4 --soft-npz $SOFT")
COMPLETO = "--cv-folds 5 --cv-fold -1 --valid-groups 60"
ALUMNOS = {"lstm": ("--cell lstm --layers 2 --hidden 128 --dropout 0.1", 0.5),
           "gru": ("--cell gru --layers 2 --hidden 128", 0.45),
           "fusion": ("--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64", 0.7)}
PRINCIPAL = ["joaquinfiorina/connectome-train-part1", "joaquinfiorina/connectome-train-part2",
             "joaquinfiorina/connectome-valid", "joaquinfiorina/connectome-ftv", "joaquinfiorina/connectome-la49"]
SEGUNDA = ["joaquinfiorina/connectome-train-part1", "joaquinfiorina/connectome-train-part2",
           "joaquinfiorina/connectome-valid", "joaqfio/connectome-fusion", "joaqfio/connectome-soft-s10",
           "joaqfio/connectome-la49"]
PARTES = {
    "a": {"cuenta": "joaqfio", "fuentes": SEGUNDA, "maestro": "maestro_la25.pt",
          "nuevos": [("gru", 5001), ("gru", 5002)], "viejos": [("p49_gru_la25", "gru")]},
    "b": {"cuenta": "joaquinfiorina", "fuentes": PRINCIPAL, "maestro": "maestro_la25.pt",
          "nuevos": [("lstm", 5003)], "viejos": [("p49_lstm_la25", "lstm")]},
    "c": {"cuenta": "joaqfio", "fuentes": SEGUNDA, "maestro": "maestro_la25.pt",
          "nuevos": [("fusion", 5004), ("gru", 5005)], "viejos": []},
}
CLASES = [0, 1, 2, 3]
SEMILLAS = [1, 2]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def buscador(p):
    con_train = bool(p["nuevos"])
    viejos = "".join(f"BASE_{n} = buscar('{n}_base.pt')\nFTV_{n} = buscar('{n}_ftv.pt')\n" for n, _ in p["viejos"])
    return (
        "import os, time, glob, sys, shutil, subprocess, json, torch\n"
        "import numpy as np\n"
        "T0 = time.time()\n"
        "LIMITE = 11.4\n"
        "DUR = {}\n"
        "def horas():\n"
        "    return (time.time() - T0) / 3600\n"
        "def alcanza(h, que):\n"
        "    ok = horas() + h < LIMITE\n"
        "    print(f'{que}: van {horas():.2f} h, estimado {h:.2f} h ->', 'sigue' if ok else 'SALTEADO POR TIEMPO', flush=True)\n"
        "    return ok\n"
        "assert torch.cuda.is_available(), 'Activá GPU (Session options -> Accelerator -> GPU)'\n"
        "print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)\n"
        "try:\n"
        "    import numba\n"
        "except ImportError:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numba'], check=True)\n"
        "    import numba\n"
        "print('numba', numba.__version__)\n"
        "\n"
        "def buscar(nombre, espera=900):\n"
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
        + ("PARTES = [buscar(n) for n in ('train_part1.parquet', 'train_part2.parquet')]\n" if con_train else "PARTES = []\n") +
        "VALID = buscar('valid.parquet')\n"
        "SOFT = buscar('soft_s10_valid.npz')\n"
        "CAUSALES = ' '.join(buscar(f'ftv_init_{m}.pt') for m in " + repr(CAUSALES) + ")\n"
        f"MAESTRO = buscar('{p['maestro']}')\n"
        + viejos +
        "os.makedirs('/kaggle/working/results', exist_ok=True)\n"
        "LOCAL = '/kaggle/tmp/datos'\n"
        "os.makedirs(LOCAL, exist_ok=True)\n"
        "tam = sum(os.path.getsize(p) for p in PARTES + [VALID]) / 1e9\n"
        "libre = shutil.disk_usage(LOCAL).free / 1e9\n"
        "print(f'datos {tam:.1f} GB, disco local libre {libre:.1f} GB')\n"
        "assert libre > tam + 3, 'no entra en el disco local'\n"
        "locales = []\n"
        "for p in PARTES + [VALID]:\n"
        "    d = os.path.join(LOCAL, os.path.basename(p))\n"
        "    shutil.copyfile(p, d)\n"
        "    locales.append(d)\n"
        "    print(f'    copiado {os.path.basename(p)} ({horas() * 60:.1f} min)', flush=True)\n"
        "TRAIN = ' '.join(locales[:-1])\n"
        "VALID = locales[-1]\n"
        "R = '/kaggle/working/results'\n"
        "print('Inputs OK en disco local')\n")


RESUMEN = (
    "for d in sorted(glob.glob(R + '/*')):\n"
    "    f = d + '/metrics.json'\n"
    "    if not os.path.isfile(f):\n"
    "        print(os.path.basename(d), 'SIN METRICAS'); continue\n"
    "    m = json.load(open(f))\n"
    "    h = [round(x['wp'], 4) for x in m['history'] if x['tag'] != 'swa']\n"
    "    print(os.path.basename(d), '| train', len(m['config']['train_ids']), '| curva', h)\n"
    "print('sopas:', sorted(os.path.basename(f) for f in glob.glob(R + '/*_ftv.pt')))\n"
    "print(f'total {horas():.2f} h', DUR)\n")


def guardado(nombre, cmd, est):
    return (f"if {est}:\n"
            "    t = time.time()\n"
            f"    !{cmd} || echo 'FALLO {nombre}'\n"
            f"    !rm -f {{R}}/{nombre}/checkpoint.pt {{R}}/{nombre}/best_model.pt\n"
            f"    DUR['{nombre}'] = round((time.time() - t) / 3600, 2)\n"
            f"    print('{nombre}', DUR['{nombre}'], 'h')\n")


def especialistas(n, arq, base, ftv, sem):
    tr = "python train_official_gpu_v1.py"
    celdas = []
    for c in CLASES:
        init = ftv if c < 3 else base
        for k in SEMILLAS:
            run = f"{n}_c{c}_s{k}"
            s = sem + 100 + 10 * c + k
            celdas.append(code(guardado(run, f"{tr} --train-parquet $VALID --valid-parquet $VALID {arq} {AJUSTE} {COMPLETO} "
                                             f"--train-classes {c} --valid-classes {c} --seed {s} --model-seed {s} "
                                             f"--init-from {init} --output-dir {{R}} --run-name {run}",
                                        f"os.path.isfile(f'{init}') and alcanza(0.08, '{run}')")))
    return celdas


for parte, p in PARTES.items():
    if len(sys.argv) > 1 and parte not in sys.argv[1].split(","):
        continue
    tr = "python train_official_gpu_v1.py"
    humo_rec = RECETA.replace("--train-groups all --valid-groups all", "--train-groups 16 --valid-groups 8").replace("--epochs 3", "--epochs 1")
    humo_aj = AJUSTE.replace("--train-groups all", "--train-groups 30").replace("--epochs 6", "--epochs 1").replace("--swa-start 6", "--swa-start 0")
    if p["nuevos"]:
        arq_h = ALUMNOS[p["nuevos"][0][0]][0]
        humo = ("os.makedirs('/kaggle/tmp/smoke', exist_ok=True)\n"
                "!python etiquetas_la.py --parquet $TRAIN --maestros-la $MAESTRO --maestros-causales $CAUSALES --grupos 16 "
                "--seed 1 --batch 16 --salida /kaggle/tmp/smoke/soft.npz\n"
                f"!{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq_h} {humo_rec} --real-weight 0 --distill-weight 1 "
                "--soft-npz /kaggle/tmp/smoke/soft.npz --seed 1 --model-seed 1 --output-dir /kaggle/tmp/smoke --run-name alumno\n"
                "INIT_H = '/kaggle/tmp/smoke/alumno/last_model.pt'\n")
    else:
        n0, a0 = p["viejos"][0]
        arq_h = ALUMNOS[a0][0]
        humo = f"INIT_H = BASE_{n0}\n"
    humo += (f"!{tr} --train-parquet $VALID --valid-parquet $VALID {arq_h} {humo_aj} --cv-folds 5 --cv-fold -1 --valid-groups 8 "
             "--train-classes 3 --valid-classes 3 --seed 1 --model-seed 1 --init-from $INIT_H --output-dir /kaggle/tmp/smoke "
             "--run-name espec\n"
             "j = json.load(open('/kaggle/tmp/smoke/espec/metrics.json'))\n"
             "assert j['config']['init_from'] and j['config']['soft_npz'], 'especialista sin padre o sin etiquetas'\n"
             "assert j['config']['train_classes'] == [3], 'sin filtro de instrumento'\n")
    if p["nuevos"]:
        humo += ("a = json.load(open('/kaggle/tmp/smoke/alumno/metrics.json'))\n"
                 "assert a['config']['real_weight'] == 0 and a['config']['soft_npz'], 'alumno sin etiquetas puras'\n")
    humo += ("print('Prueba de humo OK | especialista', round(j['best_valid_wp'], 4))\n"
             "print(f'humo en {horas():.2f} h')\n")
    desc_n = ", ".join(f"{a} {s}" for a, s in p["nuevos"]) or "ninguno"
    desc_v = ", ".join(n for n, _ in p["viejos"]) or "ninguno"
    titulo = (f"# Connectome — sprint 50{parte.upper()}: miembros de S21 con especialistas\n\n"
              f"**Cuenta:** `{p['cuenta']}`. GPU. **Save & Run All.**\n\n"
              f"Maestro `{p['maestro']}` del sprint 49. Alumnos nuevos: {desc_n}. Miembros ya hechos: {desc_v}. "
              "Cada alumno: 3 épocas con etiquetas 50 % causales + 50 % lector, ajuste con todo valid (2 semillas, sopa) y "
              "especialistas por instrumento (instrumentos 0-2 desde la sopa ajustada, el 3 desde la base, 2 semillas cada uno).")
    cells = [md(titulo), code(buscador(p)),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" + (OG / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             code("%%writefile etiquetas_la.py\n" + (OG / "etiquetas_la.py").read_text(encoding="utf-8")),
             code("%%writefile soup_models.py\n" + (SUB / "soup_models.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo)]
    for n, a in p["viejos"]:
        arq = ALUMNOS[a][0]
        sem = 5000 + 7 * len(n)
        cells.append(md(f"## Especialistas de {n}"))
        cells += especialistas(n, arq, f"{{BASE_{n}}}", f"{{FTV_{n}}}", sem)
    if p["nuevos"]:
        cells += [md("## Etiquetas de train"),
                  code("t = time.time()\n"
                       "!python etiquetas_la.py --parquet $TRAIN --maestros-la $MAESTRO --maestros-causales $CAUSALES "
                       "--batch 16 --salida /kaggle/tmp/soft_mezcla.npz\n"
                       "DUR['etiquetas'] = round((time.time() - t) / 3600, 2)\n"
                       "print('etiquetas', DUR['etiquetas'], 'h')\n")]
    for a, sem in p["nuevos"]:
        arq, est = ALUMNOS[a]
        n = f"p50_{a}_s{sem}"
        cells.append(md(f"## {n}"))
        cells.append(code(guardado(n, f"{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq} {RECETA} --real-weight 0 "
                                      f"--distill-weight 1 --soft-npz /kaggle/tmp/soft_mezcla.npz --seed {sem} --model-seed {sem} "
                                      f"--output-dir {{R}} --run-name {n}",
                                   f"os.path.isfile('/kaggle/tmp/soft_mezcla.npz') and alcanza({est}, '{n}')")))
        for k in SEMILLAS:
            run = f"{n}_all_s{k}"
            cells.append(code(guardado(run, f"{tr} --train-parquet $VALID --valid-parquet $VALID {arq} {AJUSTE} {COMPLETO} "
                                            f"--seed {sem + 20 + k} --model-seed {sem + 20 + k} --init-from {{R}}/{n}/last_model.pt "
                                            f"--output-dir {{R}} --run-name {run}",
                                       f"os.path.isfile(R + '/{n}/last_model.pt') and alcanza(0.2, '{run}')")))
        cells.append(code(f"if all(os.path.isfile(f'{{R}}/{n}_all_s{{k}}/last_model.pt') for k in {SEMILLAS}):\n"
                          f"    !python soup_models.py --out {{R}}/{n}_ftv.pt " +
                          " ".join(f"{{R}}/{n}_all_s{k}/last_model.pt" for k in SEMILLAS) + "\n"))
        cells += especialistas(n, arq, f"{{R}}/{n}/last_model.pt", f"{{R}}/{n}_ftv.pt", sem)
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint50{parte}_s21.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": f"{p['cuenta']}/connectome-s50{parte}-s21", "title": f"connectome-s50{parte}-s21",
            "code_file": out.name, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": p["fuentes"],
            "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (HERE / f"meta_s50{parte}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("ok:", out.name, p["cuenta"], "| celdas", len(cells))
