import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent
OG = CH / "model" / "official_gpu"

CAUSALES = ["mask_swa", "LR_swa", "LA_best", "long_best"]
RECETA = ("--epochs 3 --batch 64 --chunk 1024 --lr-schedule cosine --lr 1.5e-3 --lr-min 1e-4 --weight-decay 1e-5 --clip 1.0 "
          "--weight-floor 0.2 --train-groups all --valid-groups all --workers 4")
AJUSTE = ("--lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 6 --swa-start 6 --distill-weight 4 "
          "--unscored-weight 0.1 --train-groups all --workers 4 --soft-npz $SOFT")
AJUSTE_LA = ("--lr 2e-4 --lr-decay 1.0 --weight-decay 1e-4 --clip 1.0 --epochs 4 --unscored-weight 0.1 --weight-floor 0.2 "
             "--train-groups all --workers 4")
COMPLETO = "--cv-folds 5 --cv-fold -1 --valid-groups 60"
ALUMNOS = {"lstm": ("--cell lstm --layers 2 --hidden 128 --dropout 0.1", 1.0),
           "gru": ("--cell gru --layers 2 --hidden 128", 0.9),
           "fusion": ("--arch fusion --fusion-features --hidden 128 --fusion-emb 64 --fusion-mid 64", 1.4)}
DIL = {25: "1 2 4 8 9", 50: "1 2 4 8 16 18"}
PRINCIPAL = ["joaquinfiorina/connectome-train-part1", "joaquinfiorina/connectome-train-part2",
             "joaquinfiorina/connectome-valid", "joaquinfiorina/connectome-ftv"]
SEGUNDA = ["joaquinfiorina/connectome-train-part1", "joaquinfiorina/connectome-train-part2",
           "joaquinfiorina/connectome-valid", "joaqfio/connectome-fusion", "joaqfio/connectome-soft-s10"]
PARTES = {"a": ("joaquinfiorina", PRINCIPAL, 25, 4901, "lstm", 4702),
          "b": ("joaquinfiorina", PRINCIPAL, 50, 4902, "fusion", 4701),
          "c": ("joaqfio", SEGUNDA, 50, 4903, "lstm", 4702),
          "d": ("joaqfio", SEGUNDA, 25, 4904, "gru", 4905)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


BUSCADOR = (
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
    "SOFT = buscar('soft_s10_valid.npz')\n"
    "CAUSALES = ' '.join(buscar(f'ftv_init_{m}.pt') for m in " + repr(CAUSALES) + ")\n"
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
    "TRAIN = ' '.join(locales[:2])\n"
    "VALID = locales[2]\n"
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
    "fin = [json.load(open(f))['history'] for f in sorted(glob.glob(R + '/*_f[0-4]/metrics.json'))]\n"
    "fin = [[x['wp'] for x in h if x['tag'] != 'swa'][-1] for h in fin]\n"
    "if fin:\n"
    "    print('ajuste fuera de pliegue, ultima epoca:', [round(x, 4) for x in fin], 'promedio', round(sum(fin) / len(fin), 4))\n"
    "print(f'total {horas():.2f} h', DUR)\n")


def guardado(nombre, cmd, est):
    return (f"if {est}:\n"
            "    t = time.time()\n"
            f"    !{cmd} || echo 'FALLO {nombre}'\n"
            f"    !rm -f {{R}}/{nombre}/checkpoint.pt\n"
            f"    DUR['{nombre}'] = round((time.time() - t) / 3600, 2)\n"
            f"    print('{nombre}', DUR['{nombre}'], 'h')\n")


for parte, (cuenta, fuentes, K, sem_m, alumno, sem_a) in PARTES.items():
    maestro = f"maestro_la{K}"
    arq_m = f"--arch lookahead --hidden 128 --layers 2 --la-canales 64 --la-dil {DIL[K]}"
    arq_a, factor = ALUMNOS[alumno]
    padre = f"p49_{alumno}_la{K}"
    tr = "python train_official_gpu_v1.py"
    humo_rec = RECETA.replace("--train-groups all --valid-groups all", "--train-groups 16 --valid-groups 8").replace("--epochs 3", "--epochs 1")
    humo = (f"!{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq_m} {humo_rec} --seed 1 --model-seed 1 "
            "--output-dir /kaggle/tmp/smoke --run-name maestro\n"
            "!python etiquetas_la.py --parquet $TRAIN --maestros-la /kaggle/tmp/smoke/maestro/last_model.pt "
            "--maestros-causales $CAUSALES --grupos 16 --seed 1 --batch 16 --salida /kaggle/tmp/smoke/soft.npz\n"
            f"!{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq_a} {humo_rec} --real-weight 0 --distill-weight 1 "
            "--soft-npz /kaggle/tmp/smoke/soft.npz --seed 1 --model-seed 1 --output-dir /kaggle/tmp/smoke --run-name alumno\n"
            f"!{tr} --train-parquet $VALID --valid-parquet $VALID {arq_a} "
            + AJUSTE.replace("--train-groups all", "--train-groups 30").replace("--epochs 6", "--epochs 1").replace("--swa-start 6", "--swa-start 0")
            + " --cv-folds 5 --cv-fold 0 --valid-groups 8 --seed 1 --model-seed 1 --init-from /kaggle/tmp/smoke/alumno/last_model.pt "
            "--output-dir /kaggle/tmp/smoke --run-name ajuste\n"
            f"!{tr} --train-parquet $VALID --valid-parquet $VALID {arq_m} "
            + AJUSTE_LA.replace("--train-groups all", "--train-groups 30").replace("--epochs 4", "--epochs 1")
            + " --cv-folds 5 --cv-fold 0 --valid-groups 8 --seed 1 --model-seed 1 --init-from /kaggle/tmp/smoke/maestro/last_model.pt "
            "--output-dir /kaggle/tmp/smoke --run-name maestro_cf\n"
            "!python etiquetas_la.py --parquet $VALID --maestros-la /kaggle/tmp/smoke/maestro_cf/last_model.pt "
            "--pliegues 5 --pliegue 0 --batch 16 --salida /kaggle/tmp/smoke/cf0.npz\n"
            "m = json.load(open('/kaggle/tmp/smoke/maestro/metrics.json'))\n"
            "a = json.load(open('/kaggle/tmp/smoke/alumno/metrics.json'))\n"
            "j = json.load(open('/kaggle/tmp/smoke/ajuste/metrics.json'))\n"
            "c = json.load(open('/kaggle/tmp/smoke/maestro_cf/metrics.json'))\n"
            f"assert m['config']['arch'] == 'lookahead' and m['config']['la_dil'] == [{DIL[K].replace(' ', ', ')}], 'maestro'\n"
            "assert a['config']['real_weight'] == 0 and a['config']['soft_npz'], 'alumno sin etiquetas puras'\n"
            "assert j['config']['init_from'] and j['config']['soft_npz'], 'ajuste sin padre'\n"
            "assert c['config']['init_from'] and c['config']['arch'] == 'lookahead', 'maestro cruzado'\n"
            "z = np.load('/kaggle/tmp/smoke/cf0.npz')\n"
            "assert z['soft'].shape[0] == len(z['ids']) > 300, 'etiquetas del pliegue'\n"
            "print('Prueba de humo OK | maestro', round(m['best_valid_wp'], 4), '| alumno', round(a['best_valid_wp'], 4), "
            "'| ajuste', round(j['best_valid_wp'], 4), '| maestro cruzado', round(c['best_valid_wp'], 4))\n"
            "print(f'humo en {horas():.2f} h')\n")
    titulo = (f"# Connectome — sprint 49{parte.upper()}: miembro completo con maestro de {K} filas futuras ({alumno})\n\n"
              f"**Cuenta:** `{cuenta}`. GPU. **Save & Run All.**\n\n"
              "Copia los datos al disco local. Maestro con lector del futuro (3 épocas), etiquetas de train (mitad causales, "
              "mitad lector), alumno de 3 épocas solo con esas etiquetas, ajuste con la receta de S13 en 5 pliegues (prueba "
              "limpia) y con todo valid (2 semillas, para la entrega). Al final, el maestro ajustado en cada pliegue de valid "
              "deja etiquetas fuera de pliegue para el ajuste de mañana. Cada paso mira el reloj y se saltea si no entra en 12 h.")
    cells = [md(titulo), code(BUSCADOR),
             code("%%writefile utils.py\n" + (ROOT / "utils.py").read_text(encoding="utf-8")),
             code("%%writefile train_official_gpu_v1.py\n" + (OG / "train_official_gpu_v1.py").read_text(encoding="utf-8")),
             code("%%writefile etiquetas_la.py\n" + (OG / "etiquetas_la.py").read_text(encoding="utf-8")),
             md("## Prueba de humo"), code(humo),
             md(f"## {maestro}"),
             code(guardado(maestro, f"{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq_m} {RECETA} --seed {sem_m} "
                                    f"--model-seed {sem_m} --output-dir {{R}} --run-name {maestro}", "True")),
             md("## Etiquetas de train y alumno"),
             code(f"seguir = os.path.isfile(R + '/{maestro}/last_model.pt') and alcanza(0.4 + {factor} * DUR['{maestro}'], 'etiquetas + alumno')\n"
                  "if seguir:\n"
                  "    t = time.time()\n"
                  f"    !python etiquetas_la.py --parquet $TRAIN --maestros-la {{R}}/{maestro}/last_model.pt --maestros-causales $CAUSALES "
                  "--batch 16 --salida /kaggle/tmp/soft_mezcla.npz\n"
                  "    DUR['etiquetas'] = round((time.time() - t) / 3600, 2)\n"
                  "    print('etiquetas', DUR['etiquetas'], 'h')\n"),
             code(guardado(padre, f"{tr} --train-parquet $TRAIN --valid-parquet $VALID {arq_a} {RECETA} --real-weight 0 "
                                  f"--distill-weight 1 --soft-npz /kaggle/tmp/soft_mezcla.npz --seed {sem_a} --model-seed {sem_a} "
                                  f"--output-dir {{R}} --run-name {padre}",
                           f"seguir and os.path.isfile('/kaggle/tmp/soft_mezcla.npz') and alcanza({factor} * DUR['{maestro}'], '{padre}')"))]
    cells.append(md("## Ajuste en 5 pliegues (prueba limpia)"))
    for f in range(5):
        n = f"{padre}_f{f}"
        cells.append(code(guardado(n, f"{tr} --train-parquet $VALID --valid-parquet $VALID {arq_a} {AJUSTE} --cv-folds 5 "
                                      f"--cv-fold {f} --valid-groups all --seed {sem_a + 10 + f} --model-seed {sem_a + 10 + f} "
                                      f"--init-from {{R}}/{padre}/last_model.pt --output-dir {{R}} --run-name {n}",
                                   f"os.path.isfile(R + '/{padre}/last_model.pt') and alcanza(0.2, '{n}')")))
    cells.append(md("## Ajuste con todo valid (para la entrega)"))
    for k in (1, 2):
        n = f"{padre}_all_s{k}"
        cells.append(code(guardado(n, f"{tr} --train-parquet $VALID --valid-parquet $VALID {arq_a} {AJUSTE} {COMPLETO} "
                                      f"--seed {sem_a + 20 + k} --model-seed {sem_a + 20 + k} --init-from {{R}}/{padre}/last_model.pt "
                                      f"--output-dir {{R}} --run-name {n}",
                                   f"os.path.isfile(R + '/{padre}/last_model.pt') and alcanza(0.25, '{n}')")))
    cells.append(md("## Maestro ajustado por pliegue: etiquetas fuera de pliegue de valid"))
    for f in range(5):
        n = f"{maestro}_cf{f}"
        cells.append(code(guardado(n, f"{tr} --train-parquet $VALID --valid-parquet $VALID {arq_m} {AJUSTE_LA} --cv-folds 5 "
                                      f"--cv-fold {f} --valid-groups all --seed {sem_m + 10 + f} --model-seed {sem_m + 10 + f} "
                                      f"--init-from {{R}}/{maestro}/last_model.pt --output-dir {{R}} --run-name {n}",
                                   f"os.path.isfile(R + '/{maestro}/last_model.pt') and alcanza(0.3, '{n}')")
                          + f"if os.path.isfile(R + '/{n}/last_model.pt'):\n"
                            f"    !python etiquetas_la.py --parquet $VALID --maestros-la {{R}}/{n}/last_model.pt --pliegues 5 "
                            f"--pliegue {f} --batch 16 --salida /kaggle/tmp/cf{f}.npz\n"))
    cells.append(code("partes = [f'/kaggle/tmp/cf{f}.npz' for f in range(5)]\n"
                      "if all(os.path.isfile(p) for p in partes):\n"
                      "    zs = [np.load(p) for p in partes]\n"
                      "    ids = np.concatenate([z['ids'] for z in zs])\n"
                      "    soft = np.concatenate([z['soft'] for z in zs])\n"
                      "    orden = np.argsort(ids)\n"
                      "    assert len(np.unique(ids)) == len(ids)\n"
                      f"    np.savez('/kaggle/working/soft_la{K}_cf_valid.npz', soft=soft[orden], ids=ids[orden])\n"
                      f"    print('etiquetas fuera de pliegue guardadas', soft.shape)\n"
                      "else:\n"
                      "    print('faltan pliegues del maestro cruzado:', [p for p in partes if not os.path.isfile(p)])\n"))
    cells += [md("## Resumen"), code(RESUMEN)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"connectome_sprint49{parte}_miembro.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"id": f"{cuenta}/connectome-s49{parte}-miembro", "title": f"connectome-s49{parte}-miembro",
            "code_file": out.name, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": fuentes,
            "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (HERE / f"meta_s49{parte}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("ok:", out.name, cuenta)
