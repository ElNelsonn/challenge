"""Auditoria reproducible, sin entrenamiento ni servicios externos.

Conserva predicciones sin recortar hasta DESPUES de promediar los modelos.
Los intervalos remuestrean secuencias completas y recalculan WP global.
La calibracion cruzada es exploratoria: los modelos base ya fueron elegidos
usando valid, por lo que estos folds no constituyen un nuevo holdout limpio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pyarrow.parquet as pq


def sufficient(y, p, owner, n):
    y = np.clip(y, -2, 2).astype(np.float64)
    p = np.clip(p, -2, 2).astype(np.float64)
    w = np.abs(y)
    out = np.empty((n, 2, 6), dtype=np.float64)
    for j in range(2):
        for k, v in enumerate((w[:, j], w[:, j]*y[:, j], w[:, j]*p[:, j],
                               w[:, j]*y[:, j]**2, w[:, j]*p[:, j]**2,
                               w[:, j]*y[:, j]*p[:, j])):
            out[:, j, k] = np.bincount(owner, weights=v, minlength=n)
    return out


def score_stats(s):
    W, wy, wp, wyy, wpp, wyp = np.moveaxis(np.asarray(s), -1, 0)
    W = np.maximum(W, 1e-16)
    cov = wyp/W - wy*wp/W**2
    vy, vp = np.maximum(wyy/W-(wy/W)**2, 0), np.maximum(wpp/W-(wp/W)**2, 0)
    den = np.sqrt(vy*vp)
    return np.divide(cov, den, out=np.zeros_like(cov), where=den > 1e-16)


def metric(y, p, owner, n):
    s = sufficient(y, p, owner, n)
    v = score_stats(s.sum(0))
    return {"wp": float(v.mean()), "t0": float(v[0]), "t1": float(v[1])}, s


def bootstrap(a, b, counts):
    av = score_stats(np.einsum("bn,ntk->btk", counts, a)).mean(1)
    bv = score_stats(np.einsum("bn,ntk->btk", counts, b)).mean(1)
    d = bv-av
    return {"delta": float(score_stats(b.sum(0)).mean()-score_stats(a.sum(0)).mean()),
            "ci95": np.quantile(d, [.025, .975]).tolist(),
            "fraction_positive": float(np.mean(d > 0))}


def weighted_corr(a, b, w):
    am, bm = np.sum(w*a)/w.sum(), np.sum(w*b)/w.sum()
    aa, bb = a-am, b-bm
    sa, sb = np.sqrt(np.sum(w*aa**2)/w.sum()), np.sqrt(np.sum(w*bb**2)/w.sum())
    return float(np.sum(w*aa*bb)/w.sum()/max(sa*sb, 1e-16)), float(sa), float(sb)


def calibrate(q, y, owner, folds, degree):
    parts = [np.ones(len(q)), q[:, 0], q[:, 1]]
    if degree == 2:
        parts += [q[:, 0]**2, q[:, 0]*q[:, 1], q[:, 1]**2]
    x = np.column_stack(parts).astype(np.float64)
    result = np.empty_like(q)
    coefs = []
    for fold in range(5):
        train, test = folds[owner] != fold, folds[owner] == fold
        xx = x[train]
        c = np.empty((x.shape[1], 2))
        for j in range(2):
            yy = np.clip(y[train, j], -2, 2).astype(np.float64)
            w = np.abs(yy)
            lhs = xx.T @ (w[:, None]*xx)
            rhs = xx.T @ (w*yy)
            c[:, j] = np.linalg.solve(lhs + 1e-8*np.eye(len(lhs)), rhs)
        result[test] = x[test] @ c
        coefs.append(c.tolist())
    return result, coefs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", action="store_true", help="reproduce baseline ONNX en 10 grupos fijos")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    root, ch = args.root.resolve(), args.root.resolve()/"challenge"
    sys.path.insert(0, str(root))
    from utils import FEATURE_COLUMNS, weighted_pearson
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    z = np.load(ch/"model/submission/results/valid_scored_cache.npz")
    order = np.lexsort((z["row"], z["seq"]))
    seq, row, y = z["seq"][order], z["row"][order], z["y"][order]
    seqs, owner = np.unique(seq, return_inverse=True)
    n = len(seqs)
    assert np.all(row >= 99) and np.all(row < 20000)
    assert np.all((np.diff(seq) != 0) | (np.diff(row) > 0))
    boundaries = np.r_[0, np.flatnonzero(np.diff(owner))+1, len(owner)]
    spans = [slice(boundaries[i], boundaries[i+1]) for i in range(n)]
    result = {"date": "2026-10-26", "rows": len(y), "sequences": n, "models": {}}
    endpoints = np.array([row[sl][min(2000, sl.stop-sl.start)-1] for sl in spans])
    result["legacy_first2000_scored_summary"] = {
        "end_step_p10_p50_p90": np.quantile(endpoints, [.1, .5, .9]).tolist(),
        "fraction_ending_after_step2000": float(np.mean(endpoints > 2000)),
        "fraction_sequences_fewer_2000_scored": float(np.mean(np.diff(boundaries) < 2000))}
    rng = np.random.default_rng(261026)
    counts = rng.multinomial(n, np.full(n, 1/n), size=args.bootstrap).astype(np.float64)
    r = ch/"model/official_gpu/results"
    paths = {
        "LR": r/"sprint6/results/student_lr5e-4/valid_preds_best.npz",
        "LA": r/"sprint5/results/student_LA/valid_preds_best.npz",
        "long": r/"sprint11_ema/results/student_long/valid_preds_best.npz",
        "b256": r/"sprint8_notebookdeb2463071/results/student_big256/valid_preds_swa.npz",
        "mask_swa": r/"s15_notebookfd82629297/results/student_mask/valid_preds_swa.npz",
    }
    p = {}
    for name, path in paths.items():
        with np.load(path) as d:
            ids, full = d["seq_ix"], d["preds"]
            assert len(np.unique(ids)) == len(ids)
            pos = {int(s): i for i, s in enumerate(ids)}
            assert set(seqs).issubset(pos), f"cobertura incompleta: {path}"
            q = np.empty((len(y), 2), np.float32)
            for i, s in enumerate(seqs):
                sl = spans[i]
                q[sl] = full[pos[int(s)], row[sl]]
            assert np.isfinite(full[:, 99:]).all(), f"predicciones no finitas: {path}"
            p[name] = q
            result["models"][name] = metric(y, q, owner, n)[0] | {
                "path": str(path.relative_to(ch)), "prediction_abs_gt_2": float(np.mean(np.abs(q)>2))}
        print(name, result["models"][name]["wp"], flush=True)

    base = np.mean([p[k] for k in ("LR", "LA", "long", "b256")], axis=0)
    base_score, base_stats = metric(y, base, owner, n)
    result["documented_ensemble"] = base_score
    clipped_first = np.mean([np.clip(p[k], -2, 2) for k in ("LR", "LA", "long", "b256")], axis=0)
    result["clip_before_blend"] = metric(y, clipped_first, owner, n)[0]
    result["clip_before_blend"]["max_prediction_difference"] = float(np.max(np.abs(clipped_first-base)))
    eligible = np.array([all(np.any(y[sl, j] > 0) and np.any(y[sl, j] < 0) for j in range(2))
                         for sl in spans])
    result["legacy_block_wp"] = float(score_stats(base_stats)[eligible].mean())
    result["legacy_eligible_blocks"] = int(eligible.sum())
    yc = np.clip(y, -2, 2)
    result["metric_check_vs_utils"] = [weighted_pearson(y[:, j], base[:, j]) for j in range(2)]
    result["target_both_positive_fraction"] = float(np.mean(np.all(y > 0, axis=1)))
    result["model_marginal"] = {}
    for name, q in p.items():
        derivatives, details = [], []
        for j in range(2):
            w = np.abs(yc[:, j]).astype(np.float64)
            # No clipping activation for the local derivative when all predictions are interior.
            S, _, sb = weighted_corr(yc[:, j], base[:, j], w)
            s, _, sq = weighted_corr(yc[:, j], q[:, j], w)
            corr, _, _ = weighted_corr(base[:, j], q[:, j], w)
            derivative = sq/sb*(s-S*corr)
            derivatives.append(derivative)
            details.append({"S": S, "s": s, "weighted_r": corr,
                            "std_ratio": sq/sb, "derivative": derivative})
        tiny = metric(y, base*.999+q*.001, owner, n)[0]["wp"]
        blend = metric(y, base*.8+q*.2, owner, n)[0]["wp"]
        result["model_marginal"][name] = {"targets": details,
            "derivative_mean": float(np.mean(derivatives)),
            "finite_difference_001": (tiny-base_score["wp"])/.001,
            "gain_at_20_percent": blend-base_score["wp"],
            "unweighted_r_t0": float(np.corrcoef(base[::5, 0], q[::5, 0])[0, 1])}
    folds = np.empty(n, dtype=np.int64)
    folds[rng.permutation(n)] = np.arange(n) % 5
    result["calibration_exploratory"] = {}
    for degree in (1, 2):
        q, coef = calibrate(base, y, owner, folds, degree)
        sc, st = metric(y, q, owner, n)
        result["calibration_exploratory"][str(degree)] = sc | bootstrap(base_stats, st, counts) | {
            "coefficients_by_fold": coef,
            "fold_deltas": [float(score_stats(st[folds==f].sum(0)).mean()-
                                   score_stats(base_stats[folds==f].sum(0)).mean()) for f in range(5)]}
        print("Calibracion", degree, sc, flush=True)
    alternative = np.mean([p[k] for k in ("LR", "LA", "long", "mask_swa")], axis=0)
    sc, st = metric(y, alternative, owner, n)
    result["mask_ensemble_approximation"] = sc | bootstrap(base_stats, st, counts)
    # Attribution: BUILD is the authority for the local artifact; it may not match the handoff.
    builddir = ch/"submissions/ens4_mask_LR_LA_long_int8"
    result["last_local_build"] = json.loads((builddir/"BUILD.json").read_text())
    wz = np.load(builddir/"weights.npz")
    result["last_local_build"]["hidden_per_model"] = wz["hidden_per_model"].tolist()
    result["last_local_build"]["kernel_sha256"] = hashlib.sha256((builddir/"solution.py").read_bytes()).hexdigest()
    result["current_kernel_sha256"] = hashlib.sha256((ch/"model/submission/solution.py").read_bytes()).hexdigest()

    valid = pq.ParquetFile(root/"datasets/valid.parquet")
    train = pq.ParquetFile(root/"datasets/train.parquet")
    cache_verified, selected, mapping = 0, 0, {}
    for g in range(valid.num_row_groups):
        t = valid.read_row_group(g, columns=["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"], use_threads=False)
        sid = int(t["seq_ix"][0].as_py())
        assert sid not in mapping
        mapping[sid] = g
        mask = t["need_prediction"].to_numpy().astype(bool) & t["is_scored"].to_numpy().astype(bool)
        i = int(np.searchsorted(seqs, sid))
        assert i < n and seqs[i] == sid
        sl = spans[i]
        yy = np.column_stack([t[k].to_numpy() for k in ("t0", "t1")])
        assert np.array_equal(row[sl], np.flatnonzero(mask))
        assert np.array_equal(y[sl], yy[mask])
        selected += int(mask.sum())
        cache_verified += 1
        if g % 400 == 0:
            print("Cache verificado", g, flush=True)
    assert selected == len(y)
    result["cache_verified_against_parquet"] = cache_verified
    result["strata_by_step"] = {}
    for lo, hi in ((99, 2000), (2000, 5000), (5000, 10000), (10000, 20000)):
        m = (row >= lo) & (row < hi)
        result["strata_by_step"][f"{lo}-{hi}"] = metric(y[m], base[m], owner[m], n)[0] | {"rows": int(m.sum())}
    result["book_diagnostics"] = {}
    for label, pf in (("train", train), ("valid", valid)):
        groups = sorted(rng.choice(pf.num_row_groups, 24, replace=False).tolist())
        agg = {f"i{k}_{side}": {"top_raw": [], "top_centered": [], "order": [], "change_rate": []}
               for k in range(2) for side in ("bid", "ask")}
        sat = np.zeros(112, dtype=np.int64)
        total = 0
        for g in groups:
            t = pf.read_row_group(g, columns=FEATURE_COLUMNS, use_threads=False)
            xx = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
            sat += np.isclose(xx, -5.1993, atol=2e-4).sum(0)
            total += len(xx)
            for k in range(2):
                for side, off in (("bid", 0), ("ask", 11)):
                    m = xx[:, k*52+off:k*52+off+11]
                    pick = np.argmax if side == "bid" else np.argmin
                    item = agg[f"i{k}_{side}"]
                    item["top_raw"].append(np.bincount(pick(m, axis=1), minlength=11)/len(m))
                    item["top_centered"].append(np.bincount(pick(m-m.mean(0), axis=1), minlength=11)/len(m))
                    item["order"].append(np.argsort(m.mean(0)).tolist())
                    item["change_rate"].append(np.mean(np.diff(m, axis=0) != 0, axis=0))
        result["book_diagnostics"][label] = {"groups": groups,
            "saturation": {FEATURE_COLUMNS[i]: float(sat[i]/total) for i in np.flatnonzero(sat)},
            "price_groups": {k: {"mean_top_raw": np.mean(v["top_raw"], 0).tolist(),
                                  "mean_top_centered": np.mean(v["top_centered"], 0).tolist(),
                                  "change_rate": np.mean(v["change_rate"], 0).tolist(),
                                  "unique_orders": len({tuple(z) for z in v["order"]})}
                             for k, v in agg.items()}}
    if args.baseline:
        # La version local de ORT no expone use_per_session_threads. Se usa
        # directamente el mismo grafo y la misma recurrencia, sin editar el kit.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(str(root/"baseline/baseline.onnx"), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        groups = sorted(np.random.default_rng(2027).choice(valid.num_row_groups, 10, replace=False).tolist())
        ys, ps, owners = [], [], []
        block_scores = []
        for i, g in enumerate(groups):
            t = valid.read_row_group(g, use_threads=False)
            x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            yy = np.column_stack([t[c].to_numpy() for c in ("t0", "t1")])
            mask = t["need_prediction"].to_numpy().astype(bool) & t["is_scored"].to_numpy().astype(bool)
            sid = int(t["seq_ix"][0].as_py())
            pp = np.zeros((len(x), 2), np.float32)
            h0, h1 = np.zeros((1, 1, 128), np.float32), np.zeros((1, 1, 128), np.float32)
            for step in range(len(x)):
                out, h0, h1 = session.run(None, {"features": x[step].reshape(1, 1, 112),
                                               "hidden_0": h0, "hidden_1": h1})
                pp[step] = out[0, 0]
            ys.append(yy[mask]); ps.append(pp[mask]); owners.append(np.full(mask.sum(), i))
            block_scores.append(np.mean([weighted_pearson(yy[mask, j], pp[mask, j]) for j in range(2)]))
            print("Baseline", i+1, flush=True)
        result["baseline_fixed10"] = metric(np.concatenate(ys), np.concatenate(ps), np.concatenate(owners), 10)[0] | {
            "row_groups": groups, "legacy_block_wp": float(np.mean(block_scores))}
    result["elapsed_seconds"] = time.perf_counter()-start
    baseline_record = ch/"model/submission/results/eval_full_valid_baseline.json"
    if baseline_record.exists():
        result["baseline_full_valid_historical_record"] = json.loads(baseline_record.read_text())["models"]["baseline"]
    (args.output/"auditoria_resultados.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.output/"auditoria_metric_stats.npz", seq=seqs, documented=base_stats, mask_approx=st)
    print(json.dumps({k: result[k] for k in ("documented_ensemble", "mask_ensemble_approximation", "baseline_fixed10", "elapsed_seconds") if k in result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
