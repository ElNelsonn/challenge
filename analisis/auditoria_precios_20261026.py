"""Comprueba redundancia de precios; no atribuye profundidad a valores anonimizados."""
from pathlib import Path
import argparse
import json
import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    previous = json.loads((args.output/"auditoria_resultados.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "valid"):
        pf = pq.ParquetFile(args.root/"datasets"/f"{split}.parquet")
        stats = {f"i{k}_{side}": [] for k in range(2) for side in ("bid", "ask")}
        for g in previous["book_diagnostics"][split]["groups"]:
            t = pf.read_row_group(g, columns=[f"i{k}_p{j}" for k in range(2) for j in range(22)], use_threads=False)
            for k in range(2):
                for side, off in (("bid", 0), ("ask", 11)):
                    x = np.column_stack([t[f"i{k}_p{j}"].to_numpy() for j in range(off, off+11)]).astype(np.float64)
                    # Monotonia temporal: ordenar por una columna y comprobar las otras.
                    ordered = x[np.argsort(x[:, 0], kind="stable")]
                    violation = np.mean(np.diff(ordered[:, 1:], axis=0) < -1e-7)
                    xc = x-x.mean(0)
                    cov = xc[:, :1].T @ xc[:, 1:]
                    r2 = (cov.ravel()**2)/(np.sum(xc[:, 0]**2)*np.sum(xc[:, 1:]**2, axis=0))
                    u = ndtr(x)
                    uc = u-u.mean(0)
                    covu = uc[:, :1].T @ uc[:, 1:]
                    r2u = covu.ravel()**2/(np.sum(uc[:, 0]**2)*np.sum(uc[:, 1:]**2, axis=0))
                    stats[f"i{k}_{side}"].append([float(violation), float(np.min(r2)),
                                                  float(np.min(r2u)), float(np.mean(np.any(np.diff(x, axis=0) != 0, axis=1)))])
        result[split] = {k: {"max_monotonicity_violation_fraction": float(np.max(v, axis=0)[0]),
                             "raw_min_r2_p10_p50_p90": np.quantile(np.asarray(v)[:, 1], [.1, .5, .9]).tolist(),
                             "cdf_min_r2_p10_p50_p90": np.quantile(np.asarray(v)[:, 2], [.1, .5, .9]).tolist(),
                             "mean_change_fraction": float(np.mean(v, axis=0)[3])}
                         for k, v in stats.items()}
    ch = args.root/"challenge"
    z = np.load(ch/"model/submission/results/valid_scored_cache.npz")
    order = np.lexsort((z["row"], z["seq"]))
    seq, row = z["seq"][order], z["row"][order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(seq))+1, len(seq)]
    endpoints = np.array([row[a+min(2000, b-a)-1] for a, b in zip(boundaries[:-1], boundaries[1:])])
    causal = {"end_step_p10_p50_p90": np.quantile(endpoints, [.1, .5, .9]).tolist(),
              "fraction_ending_after_step2000": float(np.mean(endpoints > 2000)),
              "fraction_sequences_fewer_2000_scored": float(np.mean(np.diff(boundaries) < 2000))}
    previous["legacy_first2000_scored_summary"] = causal
    from auditoria_20261026 import score_stats
    yy = z["y"][order]
    eligible = np.array([all(np.any(yy[a:b, j] > 0) and np.any(yy[a:b, j] < 0) for j in range(2))
                         for a, b in zip(boundaries[:-1], boundaries[1:])])
    ms = np.load(args.output/"auditoria_metric_stats.npz")
    previous["legacy_block_wp"] = float(score_stats(ms["documented"])[eligible].mean())
    previous["legacy_eligible_blocks"] = int(eligible.sum())
    baseline_path = ch/"model/submission/results/eval_full_valid_baseline.json"
    previous["baseline_full_valid_historical_record"] = json.loads(baseline_path.read_text())["models"]["baseline"]
    (args.output/"auditoria_resultados.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    (args.output/"auditoria_precios.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
