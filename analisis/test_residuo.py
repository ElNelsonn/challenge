"""¿Se puede aprender lo que las redes FALLAN? La prueba decisiva, en la laptop, sin gastar Kaggle.

Hasta ahora entrenamos siempre competidores: modelos que intentan predecir el target por su cuenta y después
se promedian. Todos terminan correlacionando 0.92-0.99 con el ensamble, y por eso no aportan.

La alternativa es entrenar un CORRECTOR: un modelo que no predice el target sino **el error del ensamble**
(y - predicción del ensamble). Por construcción, lo que aprenda es justo lo que a las redes les falta, así que
su correlación con el ensamble es casi cero y la cuenta s > S*r se cumple con muy poca señal.

La pregunta es si ese error tiene estructura aprendible o es puro ruido. Se puede contestar acá mismo porque
las redes NUNCA vieron valid: el error del ensamble sobre valid es un error honesto, fuera de muestra.

  * se ajusta un LightGBM sobre las filas de 60 secuencias de valid para predecir el error
  * se mide en otras 60 secuencias distintas, que el corrector no vio
"""
import sys
import time
from pathlib import Path

import numpy as np

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final"
          r"\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
sys.path.insert(0, str(CH / "model" / "lgbm_deltas"))
import train_lgbm_deltas_v1 as L   # noqa: E402
import lightgbm as lgb             # noqa: E402

R = CH / "model" / "official_gpu" / "results"
DATOS = CH.parent / "datasets" / "valid.parquet"
LAGS = [1, 5, 20, 100]
N_FIT, N_TEST, FILAS = 300, 300, 4000

grupos = L.ParquetGroups([str(DATOS)])
print(f"{len(grupos)} secuencias en valid")

# predicciones del ensamble de 4 redes (las redes nunca vieron valid: error honesto)
MOD = {"LR": R / "sprint6/results/student_lr5e-4/valid_preds_best.npz",
       "LA": R / "sprint5/results/student_LA/valid_preds_best.npz",
       "long": R / "sprint11_ema/results/student_long/valid_preds_best.npz",
       "b256": R / "sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz"}
acum, pos = None, None
for k, p in MOD.items():
    d = np.load(p)
    if pos is None:
        pos = {int(v): i for i, v in enumerate(d["seq_ix"])}
    pr = d["preds"].astype(np.float32)
    acum = pr if acum is None else acum + pr
ENS = acum / len(MOD)
print(f"ensamble cargado: {ENS.shape}")

rng = np.random.default_rng(99)
orden = rng.permutation(len(grupos))
g_fit, g_test = orden[:N_FIT], orden[N_FIT:N_FIT + N_TEST]
nombres = L.feature_names(LAGS, True)
imp = dict(__import__("json").loads(
    (CH / "model/lgbm_deltas/results/lgbm_deltas_all/feature_importance.json").read_text(encoding="utf-8")))
base = [n for n in nombres if n not in L.EMA_NAMES]
quedan = set(sorted(base, key=lambda n: -imp.get(n, 0.0))[:150]) | set(L.EMA_NAMES)
keep = np.array([i for i, n in enumerate(nombres) if n in quedan])
cols = [nombres[i] for i in keep]
print(f"{len(cols)} columnas\n")


def juntar(gs, filas=None):
    X, Yt, Pe, Sc = [], [], [], []
    t0 = time.time()
    for n, g in enumerate(gs, 1):
        seq_ix, need, x, y, scored = grupos.read(int(g))
        f = L.build_features(x, LAGS, True, keep)
        e = ENS[pos[int(seq_ix)]].astype(np.float32)
        m = need & scored
        idx = np.flatnonzero(m)
        if filas is not None and len(idx) > filas:
            idx = rng.choice(idx, filas, replace=False)
        X.append(f[idx]); Yt.append(y[idx]); Pe.append(e[idx]); Sc.append(np.full(len(idx), int(seq_ix)))
        if n % 20 == 0:
            print(f"  {n}/{len(gs)} secuencias ({time.time()-t0:.0f}s)", flush=True)
    return (np.concatenate(X), np.clip(np.concatenate(Yt), -2, 2).astype(np.float64),
            np.clip(np.concatenate(Pe), -2, 2).astype(np.float64), np.concatenate(Sc))


print("armando las filas de ajuste...")
Xf, Yf, Ef, _ = juntar(g_fit, FILAS)
print(f"ajuste: {Xf.shape}")
print("armando las filas de prueba...")
Xt, Yt, Et, St = juntar(g_test)
print(f"prueba: {Xt.shape}\n")


def wp(pred, Y):
    v = []
    for j in (0, 1):
        w, y, q = np.abs(Y[:, j]), Y[:, j], pred[:, j]
        s = w.sum()
        my, mq = (w * y).sum() / s, (w * q).sum() / s
        cov = (w * (y - my) * (q - mq)).sum() / s
        vy, vq = (w * (y - my) ** 2).sum() / s, (w * (q - mq) ** 2).sum() / s
        v.append(cov / np.sqrt(vy * vq))
    return (v[0] + v[1]) / 2


RES = Yf - Ef           # lo que el ensamble falla, en las filas de ajuste
print(f"el error del ensamble: desvio {RES.std():.4f} vs el del target {Yf.std():.4f} "
      f"({100*RES.std()/Yf.std():.0f} % del target sin explicar)\n")

params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=-1, verbose=-1, seed=7)
corr = np.zeros_like(Et)
for j in (0, 1):
    w = np.abs(Yf[:, j])
    ds = lgb.Dataset(Xf, label=RES[:, j], weight=w, feature_name=cols)
    b = lgb.train(params, ds, num_boost_round=600)
    corr[:, j] = b.predict(Xt)

print(f"{'':34s} {'puntaje':>8s}")
base_wp = wp(Et, Yt)
print(f"{'ensamble solo (60 secuencias nuevas)':34s} {base_wp:8.4f}")
r = float(np.corrcoef(corr[:, 0], Et[:, 0])[0, 1])
print(f"\ncorrelacion del corrector con el ensamble: {r:.4f}   (los competidores daban 0.92-0.99)")
print(f"puntaje del corrector solo: {wp(corr, Yt):.4f}\n")

mitA = (St % 2 == 0)
pesos = np.linspace(0, 1.5, 151)
gA = [wp(np.clip(Et + a * corr, -2, 2), Yt) if False else
      wp(np.clip(Et[mitA] + a * corr[mitA], -2, 2), Yt[mitA]) for a in pesos]
mejor = pesos[int(np.argmax(gA))]
bB = wp(Et[~mitA], Yt[~mitA])
mB = wp(np.clip(Et[~mitA] + mejor * corr[~mitA], -2, 2), Yt[~mitA])
print(f"peso elegido en la mitad A: {mejor:.2f}")
print(f"mitad B: solo redes {bB:.4f}  ->  redes + corrector {mB:.4f}   GANANCIA HONESTA {mB - bB:+.4f}")

print("\ncurva completa:")
for a in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    print(f"  ensamble + {a:.2f} x corrector: {wp(np.clip(Et + a * corr, -2, 2), Yt):.4f}")
