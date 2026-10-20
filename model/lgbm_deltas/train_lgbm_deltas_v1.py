r"""LightGBM sobre features causales (112 crudas + deltas a varios lags), un modelo por target.

Pensado como SEGUNDO modelo para ensamblar con la GRU: aprende cosas distintas (umbrales,
interacciones entre pocas columnas) y en inferencia cuesta ~5-10 us/fila.

* Entrena con una muestra de filas (cada fila es un ejemplo independiente): N secuencias de
  train x M filas al azar por secuencia (solo filas con need_prediction).
* Loss alineada a la métrica: target recortado a [-2, 2] y peso |target|.
* Evalúa con la métrica oficial (utils.BlockAccumulator) en los mismos grupos de valid que
  EXPERIMENTS.md (--valid-groups 10 --seed 2026 -> baseline 0.568).
* Guarda los modelos (.txt), la spec de features, métricas y las predicciones de valid (para
  ensamblar offline con la GRU). Si se pasa --gru-model best_model.pt, evalúa también la mezcla.

Uso local (desde challenge/):
    python model/lgbm_deltas/train_lgbm_deltas_v1.py --train-groups 300 --rows-per-group 300
Uso Kaggle (CPU, sin GPU):
    python train_lgbm_deltas_v1.py --train-parquet /kaggle/input/datasets/joaquinfiorina/connectome-train-part1/train_part1.parquet /kaggle/input/datasets/joaquinfiorina/connectome-train-part2/train_part2.parquet --valid-parquet /kaggle/input/datasets/joaquinfiorina/connectome-valid/valid.parquet --train-groups 2000 --rows-per-group 500 --output-dir /kaggle/working/results --run-name lgbm_deltas_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from utils import (BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS,  # noqa: E402
                   WARMUP, validate_sequence, weighted_pearson)


# ----------------------------------------------------------------------------- datos
class ParquetGroups:
    def __init__(self, paths):
        self.files = [pq.ParquetFile(p) for p in paths]
        self.index = [(f, g) for f in self.files for g in range(f.num_row_groups)]
        self.has_scored = all("is_scored" in f.schema_arrow.names for f in self.files)

    def __len__(self):
        return len(self.index)

    def read(self, i):
        f, g = self.index[i]
        cols = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        if self.has_scored:
            cols.append("is_scored")
        t = f.read_row_group(g, columns=cols, use_threads=False)
        seq_ix, need = validate_sequence(t)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        x[~np.isfinite(x)] = 0.0
        scored = t["is_scored"].to_numpy().astype(bool) if self.has_scored else None
        return seq_ix, need, x, y, scored


def select_groups(total, count, seed):
    if not 1 <= count <= total:
        raise ValueError(f"count debe estar entre 1 y {total}")
    return sorted(np.random.default_rng(seed).choice(total, count, replace=False).tolist())



# ----------------------------------------------------------------------------- métrica GLOBAL (nueva, 2026-09)
class GlobalWP:
    """Weighted Pearson global: junta TODAS las filas scoreadas de todas las secuencias y calcula
    una sola correlación por target (clip ±2, peso |y|), luego promedia t0 y t1.
    Reemplaza a la métrica por secuencia (utils.BlockAccumulator), que quedó obsoleta."""

    def __init__(self):
        self.s = np.zeros((2, 6), dtype=np.float64)   # por target: sum w, wy, wp, wyy, wpp, wyp

    def add(self, targets, predictions, mask):
        y = np.clip(targets[mask].astype(np.float64), -2.0, 2.0)
        p = np.clip(predictions[mask].astype(np.float64), -2.0, 2.0)
        w = np.abs(y)
        for j in range(2):
            wj, yj, pj = w[:, j], y[:, j], p[:, j]
            self.s[j] += [wj.sum(), (wj * yj).sum(), (wj * pj).sum(),
                          (wj * yj * yj).sum(), (wj * pj * pj).sum(), (wj * yj * pj).sum()]

    def result(self):
        out = {}
        for j, name in enumerate(TARGET_COLUMNS):
            W, wy, wp, wyy, wpp, wyp = self.s[j]
            if W < 1e-8:
                out[name] = 0.0
                continue
            my, mp = wy / W, wp / W
            cov = wyp / W - my * mp
            vy, vp = wyy / W - my * my, wpp / W - mp * mp
            out[name] = float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0)) if vy > 1e-16 and vp > 1e-16 else 0.0
        out["weighted_pearson"] = (out["t0"] + out["t1"]) / 2
        return out


# ----------------------------------------------------------------------------- features
# --- features de memoria larga (mismas que --ema-features del entrenador de redes)
_IX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
EMA_COLS = ["i0_p0", "i0_p11", "i0_v0", "i0_v11", "i1_p0", "i1_p11", "i1_v0", "i1_v11", "a0", "a1", "a2", "a4"]
EMA_IX = [_IX[c] for c in EMA_COLS]
EMA_SPANS = (100, 1000, 10000)
EMA_VOL_COLS = ["i0_p0", "i0_p11", "i0_v0", "i1_p0"]
EMA_NAMES = ([f"{c}_dif_ema{s}" for s in EMA_SPANS for c in EMA_COLS] +
             [f"{c}_vol_lenta" for c in EMA_VOL_COLS])
try:
    from scipy.signal import lfilter as _lfilter
except Exception:
    _lfilter = None


def _ema(x, span):
    """Promedio móvil exponencial causal por columna. ema[0] = x[0]."""
    a = 2.0 / (span + 1.0)
    if _lfilter is not None:
        zi = (x[0] * (1 - a))[None, :]
        return _lfilter([a], [1.0, -(1 - a)], x, axis=0, zi=zi)[0]
    out = np.empty_like(x)
    m = x[0].copy()
    for t in range(len(x)):
        m += a * (x[t] - m)
        out[t] = m
    return out


def ema_features(x):
    """(T, 112) -> (T, 40): desvío respecto de promedios móviles largos + volatilidad lenta."""
    sub = x[:, EMA_IX].astype(np.float32)
    parts, e100 = [], None
    for span in EMA_SPANS:
        e = _ema(sub, span).astype(np.float32)
        if span == EMA_SPANS[0]:
            e100 = e
        parts.append(sub - e)
    vi = [EMA_COLS.index(c) for c in EMA_VOL_COLS]
    parts.append(_ema(np.abs(sub[:, vi] - e100[:, vi]), 1000).astype(np.float32))
    return np.clip(np.concatenate(parts, axis=1), -10.0, 10.0).astype(np.float32)


class ModeloXGB:
    """Le da a un modelo de XGBoost la misma interfaz que usa el resto del script con LightGBM."""

    def __init__(self, booster, nombres):
        self.b = booster
        self.nombres = nombres

    def predict(self, f, num_threads=None):
        import xgboost as xgb
        return self.b.predict(xgb.DMatrix(f, feature_names=self.nombres))

    def feature_importance(self, kind="gain"):
        d = self.b.get_score(importance_type="gain")
        return np.array([d.get(n, 0.0) for n in self.nombres])

    def save_model(self, path):
        self.b.save_model(str(path))


def feature_names(lags, ema=False):
    names = list(FEATURE_COLUMNS)
    for k in lags:
        names += [f"{c}_delta_{k}" for c in FEATURE_COLUMNS]
    if ema:
        names += EMA_NAMES
    return names


def build_features(x, lags, ema=False, keep=None):
    """x: (T, 112) de UNA secuencia. Causal: solo pasado.
    Las primeras k filas de cada delta quedan NaN (LightGBM las maneja nativamente).
    `keep`: índices de columnas a conservar (para no arrastrar las features que no se usan)."""
    out = [x]
    for k in lags:
        d = np.full_like(x, np.nan)
        d[k:] = x[k:] - x[:-k]
        out.append(d)
    if ema:
        out.append(ema_features(x))
    f = np.concatenate(out, axis=1)
    return f if keep is None else f[:, keep]


# ----------------------------------------------------------------------------- GRU opcional (para evaluar la mezcla)
def gru_predict_factory(path):
    import torch
    from torch import nn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, cfg, pipe = ck["state_dict"], ck["config"], ck["pipeline"]
    if pipe.get("top_k", 0) or pipe.get("sentinel_flags", False):
        raise ValueError("la mezcla solo soporta GRU con 112 entradas crudas")
    core = nn.GRU if cfg["cell"] == "gru" else nn.LSTM
    model = nn.ModuleDict({"core": core(ck["n_inputs"], cfg["hidden"], num_layers=cfg["layers"], batch_first=True),
                           "head": nn.Linear(cfg["hidden"], 2)})
    model.load_state_dict(sd)
    model.eval()
    mean, std = pipe["mean"], pipe["std"]

    @torch.no_grad()
    def predict(x):
        z = torch.from_numpy(((x - mean) / std).astype(np.float32)).unsqueeze(0)
        out, _ = model["core"](z)
        return model["head"](out).squeeze(0).numpy()
    return predict


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-parquet", nargs="+", default=[str(ROOT / "datasets" / "train.parquet")])
    ap.add_argument("--valid-parquet", nargs="+", default=[str(ROOT / "datasets" / "valid.parquet")])
    ap.add_argument("--train-groups", default="all", help="N o 'all'")
    ap.add_argument("--rows-per-group", type=int, default=300)
    ap.add_argument("--valid-groups", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-data-in-leaf", type=int, default=200)
    ap.add_argument("--feature-fraction", type=float, default=0.5)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--lib", choices=["lightgbm", "xgboost"], default="lightgbm",
                    help="que libreria arma los arboles. XGBoost crece por niveles y LightGBM por hojas: "
                         "distinta forma de arbol con los mismos datos")
    ap.add_argument("--max-depth", type=int, default=6, help="solo con --lib xgboost: 6 da hasta 64 hojas, "
                    "que es lo mas parecido a las 63 de LightGBM")
    ap.add_argument("--ema-features", action="store_true", help="agregar las 40 features de memoria larga "
                    "(promedios móviles de 100/1.000/10.000 filas). Es lo que un árbol no puede calcularse solo")
    ap.add_argument("--top-features", type=int, default=0, help="quedarse con las N columnas de mayor ganancia "
                    "según --importance (0 = todas). Las features de memoria larga se conservan siempre")
    ap.add_argument("--importance", default=None, help="feature_importance.json de una corrida anterior")
    ap.add_argument("--ens-preds", nargs="*", default=[], help="valid_preds_*.npz del ensamble de redes: "
                    "reporta correlación y ganancia honesta (peso elegido en media valid, medido en la otra media)")
    ap.add_argument("--gru-model", default=None, help="best_model.pt de train_official_gpu_v1 para evaluar la mezcla")
    ap.add_argument("--output-dir", default=str(HERE / "results"))
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    import lightgbm as lgb

    run_dir = Path(args.output_dir) / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    names_all = feature_names(args.lags, args.ema_features)
    keep = None
    if args.top_features and args.importance:
        imp = dict(json.loads(Path(args.importance).read_text(encoding="utf-8")))
        base = [n for n in names_all if n not in EMA_NAMES]
        orden = sorted(base, key=lambda n: -imp.get(n, 0.0))[:args.top_features]
        quedan = set(orden) | (set(EMA_NAMES) if args.ema_features else set())
        keep = np.array([i for i, n in enumerate(names_all) if n in quedan])
        names = [names_all[i] for i in keep]
        print(f"columnas: {len(names_all)} -> {len(names)} "
              f"(top {args.top_features} por ganancia"
              f"{' + las ' + str(len(EMA_NAMES)) + ' de memoria larga' if args.ema_features else ''})", flush=True)
    else:
        names = names_all
    train = ParquetGroups(args.train_parquet)
    valid = ParquetGroups(args.valid_parquet)
    train_ids = list(range(len(train))) if str(args.train_groups) == "all" else select_groups(len(train), int(args.train_groups), args.seed)
    valid_ids = select_groups(len(valid), args.valid_groups, args.seed + 1)
    cfg = {**vars(args), "train_ids": train_ids, "valid_ids": valid_ids, "n_features": len(names)}
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"features={len(names)} (112 crudas + deltas {args.lags}"
          f"{' + 40 de memoria larga' if args.ema_features else ''})  "
          f"train seqs={len(train_ids)} x {args.rows_per_group} filas "
          f"= {len(train_ids)*args.rows_per_group/1e6:.1f}M filas, "
          f"~{len(train_ids)*args.rows_per_group*len(names)*4/1e9:.1f} GB", flush=True)

    # ---- muestra de train
    t0 = time.time()
    rng = np.random.default_rng(args.seed + 2)
    xs, ys = [], []
    for n, i in enumerate(train_ids, 1):
        _, need, x, y, _ = train.read(i)
        f = build_features(x, args.lags, args.ema_features, keep)
        rows = np.flatnonzero(need)
        take = rng.choice(rows, min(args.rows_per_group, len(rows)), replace=False)
        xs.append(f[take])
        ys.append(y[take])
        if n % 200 == 0 or n == len(train_ids):
            print(f"  muestra: {n}/{len(train_ids)} seqs  ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    del xs, ys
    print(f"X: {X.shape}  ({X.nbytes/1e9:.2f} GB)", flush=True)

    # ---- entrenar un modelo por target (loss alineada: clip +-2, peso |y|)
    params = dict(objective="regression", learning_rate=args.learning_rate, num_leaves=args.num_leaves,
                  min_data_in_leaf=args.min_data_in_leaf, feature_fraction=args.feature_fraction,
                  bagging_fraction=args.bagging_fraction, bagging_freq=1, lambda_l2=args.lambda_l2,
                  num_threads=args.threads, verbose=-1, seed=args.seed)
    models = []
    if args.lib == "xgboost":
        import xgboost as xgb
        print(f"XGBoost {xgb.__version__} | crecimiento por niveles, max_depth={args.max_depth} "
              f"(hasta {2**args.max_depth} hojas)", flush=True)
        px = dict(objective="reg:squarederror", eta=args.learning_rate, max_depth=args.max_depth,
                  subsample=args.bagging_fraction, colsample_bytree=args.feature_fraction,
                  reg_lambda=args.lambda_l2, min_child_weight=args.min_data_in_leaf * 0.7,
                  tree_method="hist", nthread=args.threads, seed=args.seed)
        for j, tname in enumerate(TARGET_COLUMNS):
            yj = np.clip(Y[:, j], -2.0, 2.0)
            # QuantileDMatrix guarda 1 byte por número (el binneado que 'hist' usa igual) en vez de los
            # 8 bytes por número de DMatrix, que con esta matriz serían 16 GB y no entran en memoria
            dtrain = xgb.QuantileDMatrix(X, label=yj, weight=np.abs(yj), feature_names=names, max_bin=256)
            t = time.time()
            booster = xgb.train(px, dtrain, num_boost_round=args.n_estimators)
            booster.save_model(str(run_dir / f"xgb_{tname}.json"))
            models.append(ModeloXGB(booster, names))
            del dtrain
            print(f"  {tname}: {args.n_estimators} árboles en {time.time()-t:.0f}s", flush=True)
        del X
    else:
        y0 = np.clip(Y[:, 0], -2.0, 2.0)
        ds = lgb.Dataset(X, label=y0, weight=np.abs(y0), feature_name=names, free_raw_data=True)
        ds.construct()
        del X   # el Dataset ya tiene los datos binneados; liberar los 4 bytes/valor crudos
        for j, tname in enumerate(TARGET_COLUMNS):
            yj = np.clip(Y[:, j], -2.0, 2.0)
            w = np.abs(yj)
            ds.set_label(yj)
            ds.set_weight(w)
            t = time.time()
            booster = lgb.train(params, ds, num_boost_round=args.n_estimators)
            booster.save_model(str(run_dir / f"lgbm_{tname}.txt"))
            models.append(booster)
            print(f"  {tname}: {args.n_estimators} árboles en {time.time()-t:.0f}s", flush=True)
    del Y

    # ---- importancia (top 25)
    imp = {n: 0.0 for n in names}
    for b in models:
        for n, g in zip(names, b.feature_importance("gain")):
            imp[n] += float(g)
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:25]
    (run_dir / "feature_importance.json").write_text(json.dumps(sorted(imp.items(), key=lambda kv: -kv[1]), indent=1), encoding="utf-8")
    print("top 25 features (ganancia):", flush=True)
    for n, g in top:
        print(f"   {n:22s} {g:12.0f}", flush=True)

    # ---- evaluación oficial en valid
    gru_predict = gru_predict_factory(args.gru_model) if args.gru_model else None
    acc, gacc = BlockAccumulator(), GlobalWP()
    mix_w = (0.2, 0.3, 0.4, 0.5)
    accs_mix = {a: BlockAccumulator() for a in mix_w} if gru_predict else {}
    gaccs_mix = {a: GlobalWP() for a in mix_w} if gru_predict else {}
    acc_gru, gacc_gru = (BlockAccumulator(), GlobalWP()) if gru_predict else (None, None)
    # predicciones del ensamble de redes, para medir correlación y ganancia honesta
    ens_src = []
    for ruta in args.ens_preds:
        d = np.load(ruta)
        ens_src.append(({int(v): i for i, v in enumerate(d["seq_ix"])}, d["preds"]))
    if ens_src:
        print(f"ensamble de referencia: {len(ens_src)} modelos", flush=True)
    ens_rows = []    # (y, arboles, ensamble, seq) solo de las filas puntuadas

    rows_out, preds_store = [], {}
    t = time.time()
    for i in valid_ids:
        seq_ix, need, x, y, scored = valid.read(i)
        f = build_features(x, args.lags, args.ema_features, keep)
        p = np.column_stack([b.predict(f, num_threads=args.threads) for b in models]).astype(np.float32)
        mask = need & scored
        acc.add(y, p, mask)
        gacc.add(y, p, mask)
        yy, pp = y[mask], p[mask]
        elig = all(np.any(yy[:, j] > 0) and np.any(yy[:, j] < 0) for j in range(2))
        row = {"seq_ix": seq_ix}
        if elig:
            row.update(wp_t0=weighted_pearson(yy[:, 0], pp[:, 0]), wp_t1=weighted_pearson(yy[:, 1], pp[:, 1]))
            row["wp"] = (row["wp_t0"] + row["wp_t1"]) / 2
        preds_store[str(seq_ix)] = p
        if ens_src and all(int(seq_ix) in pos for pos, _ in ens_src):
            e = np.mean([pr[pos[int(seq_ix)]] for pos, pr in ens_src], 0).astype(np.float32)
            ens_rows.append((y[mask], p[mask], e[mask], np.full(int(mask.sum()), int(seq_ix))))
        if gru_predict:
            g = gru_predict(x).astype(np.float32)
            acc_gru.add(y, g, mask)
            gacc_gru.add(y, g, mask)
            for a in mix_w:                        # a = peso de LightGBM
                m = (a * p + (1 - a) * g).astype(np.float32)
                accs_mix[a].add(y, m, mask)
                gaccs_mix[a].add(y, m, mask)
        rows_out.append(row)
    res, gres = acc.result(), gacc.result()
    print(f"\nLightGBM  GLOBAL WP={gres['weighted_pearson']:.4f} (t0={gres['t0']:.4f}, t1={gres['t1']:.4f})   "
          f"[por secuencia (métrica vieja): {res['weighted_pearson']:.4f}]  ({time.time()-t:.0f}s)", flush=True)
    out = {"lgbm_global": gres, "lgbm_per_sequence": res, "config": cfg}
    if gru_predict:
        rg, grg = acc_gru.result(), gacc_gru.result()
        print(f"GRU sola  GLOBAL WP={grg['weighted_pearson']:.4f} (t0={grg['t0']:.4f}, t1={grg['t1']:.4f})   [por secuencia: {rg['weighted_pearson']:.4f}]", flush=True)
        out["gru_global"], out["gru_per_sequence"], out["mix"] = grg, rg, {}
        for a in mix_w:
            rm, grm = accs_mix[a].result(), gaccs_mix[a].result()
            out["mix"][str(a)] = {"global": grm, "per_sequence": rm}
            print(f"mezcla {a:.1f}*LGBM + {1-a:.1f}*GRU  GLOBAL WP={grm['weighted_pearson']:.4f} (t0={grm['t0']:.4f}, t1={grm['t1']:.4f})   [por secuencia: {rm['weighted_pearson']:.4f}]", flush=True)
    if ens_rows:
        YY = np.clip(np.concatenate([r[0] for r in ens_rows]), -2, 2).astype(np.float64)
        PA = np.clip(np.concatenate([r[1] for r in ens_rows]), -2, 2).astype(np.float64)
        PE = np.clip(np.concatenate([r[2] for r in ens_rows]), -2, 2).astype(np.float64)
        SQ = np.concatenate([r[3] for r in ens_rows])
        WW = np.abs(YY)

        def _wp(pred, m):
            vals = []
            for j in (0, 1):
                w, y, q = WW[m, j], YY[m, j], pred[m, j]
                sw = w.sum()
                my, mq = (w * y).sum() / sw, (w * q).sum() / sw
                cov = (w * (y - my) * (q - mq)).sum() / sw
                vy, vq = (w * (y - my) ** 2).sum() / sw, (w * (q - mq) ** 2).sum() / sw
                vals.append(cov / np.sqrt(vy * vq))
            return (vals[0] + vals[1]) / 2

        A = (SQ % 2 == 0)
        todo = np.ones(len(YY), bool)
        r_corr = float(np.corrcoef(PA[:, 0], PE[:, 0])[0, 1])
        s_arb, s_ens = _wp(PA, todo), _wp(PE, todo)
        print(f"\n{'='*64}")
        print(f"ARBOLES vs ENSAMBLE DE REDES   ({len(YY):,} filas puntuadas, {len(np.unique(SQ))} secuencias)")
        print(f"{'='*64}")
        print(f"  arboles solos          {s_arb:.4f}")
        print(f"  ensamble de redes      {s_ens:.4f}")
        print(f"  correlacion entre los dos: {r_corr:.3f}")
        print(f"  umbral para aportar: los arboles necesitan pasar de {s_ens:.4f} x {r_corr:.3f} = {s_ens*r_corr:.4f}"
              f"  ->  {'LO PASAN' if s_arb > s_ens*r_corr else 'NO lo pasan'} ({s_arb - s_ens*r_corr:+.4f})")
        pesos = np.linspace(0, 0.6, 61)
        gA = [_wp((1 - a) * PE + a * PA, A) for a in pesos]
        mejor = pesos[int(np.argmax(gA))]
        bB, mB = _wp(PE, ~A), _wp((1 - mejor) * PE + mejor * PA, ~A)
        print(f"\n  peso elegido en la mitad A: {mejor:.2f}")
        print(f"  medido en la mitad B:   solo redes {bB:.4f}  ->  con arboles {mB:.4f}   "
              f"GANANCIA HONESTA {mB - bB:+.4f}")
        print(f"{'='*64}\n", flush=True)
        out["vs_ensamble"] = {"arboles": s_arb, "ensamble": s_ens, "correlacion": r_corr,
                              "umbral": s_ens * r_corr, "peso_mitad_A": float(mejor),
                              "mitad_B_solo_redes": bB, "mitad_B_con_arboles": mB, "ganancia": mB - bB}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    with (run_dir / "by_rowgroup.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq_ix", "wp_t0", "wp_t1", "wp"])
        w.writeheader()
        w.writerows(rows_out)
    np.savez_compressed(run_dir / "valid_predictions.npz", **preds_store)
    (run_dir / "feature_spec.json").write_text(json.dumps(
        {"lags": args.lags, "names": names, "ema": bool(args.ema_features),
         "keep": None if keep is None else [int(v) for v in keep]}), encoding="utf-8")
    print(f"Resultados en {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
