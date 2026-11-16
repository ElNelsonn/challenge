from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
ROOT = CHALLENGE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CHALLENGE / "model" / "official_gpu"))
from utils import DataPoint, FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, validate_sequence  # noqa: E402
from eval_ensemble import GlobalWP  # noqa: E402
from build_gated import build_lut, seq_class  # noqa: E402
from build_submission import export_weights, load_any, make_net, quantize_rows  # noqa: E402


def es_fusion(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("config", {}).get("arch") == "fusion"


def _gru_zrn(w, h):
    return np.concatenate([w[h:2 * h], w[:h], w[2 * h:]], axis=0)


def export_fusion(paths, weights, out, bits=8):
    import torch
    from train_official_gpu_v1 import FUSION_SPANS
    qdt = {8: np.int8, 16: np.int16, 32: np.float32}[bits]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    W = {k: [] for k in ("FET", "sFE", "FEB", "FIT", "sFI", "FIB", "FHT", "sFH", "FHB", "FMT", "sFM", "FMB", "FO", "FOB", "mean", "std")}
    dims = None
    for ck in cks:
        sd = {k: v.numpy().astype(np.float32) for k, v in ck["state_dict"].items()}
        emb, n_in = sd["emb.weight"].shape
        hid = sd["core.weight_hh_l0"].shape[1]
        mid = sd["mid.weight"].shape[0]
        if dims is None:
            dims = (n_in, emb, hid, mid)
        elif dims != (n_in, emb, hid, mid):
            raise ValueError("las redes fusión de un kernel tienen que tener el mismo tamaño")
        for nombre, w in (("FE", sd["emb.weight"]), ("FI", _gru_zrn(sd["core.weight_ih_l0"], hid)),
                          ("FH", _gru_zrn(sd["core.weight_hh_l0"], hid)), ("FM", sd["mid.weight"])):
            q, sc = quantize_rows(w, bits)
            W[nombre + "T"].append(np.ascontiguousarray(q.T.astype(qdt)))
            W["s" + nombre].append(sc)
        W["FEB"].append(sd["emb.bias"])
        W["FIB"].append(_gru_zrn(sd["core.bias_ih_l0"], hid))
        W["FHB"].append(_gru_zrn(sd["core.bias_hh_l0"], hid))
        W["FMB"].append(sd["mid.bias"])
        W["FO"].append(np.ascontiguousarray(sd["head.weight"][:2].T))
        W["FOB"].append(sd["head.bias"][:2])
        W["mean"].append(np.asarray(ck["pipeline"]["mean"], np.float32))
        W["std"].append(np.asarray(ck["pipeline"]["std"], np.float32))
    W = {k: np.stack(v) for k, v in W.items()}
    W["n_fusion"] = np.int64(len(cks))
    W["n_in"], W["emb"], W["hidden"], W["mid"] = (np.int64(d) for d in dims)
    W["alfas"] = np.array([2.0 / (s + 1.0) for s in FUSION_SPANS], np.float64)
    W["weights"] = np.asarray(weights, np.float32)
    np.savez(out, **W)


def referencia(paths):
    import torch
    from train_official_gpu_v1 import FeaturePipeline, FusionModel
    refs = []
    for p in paths:
        if es_fusion(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg, pst = ck["config"], ck["pipeline"]
            pipe = FeaturePipeline(0, False, None, False, False, False, 0, True)
            pipe.mean, pipe.std = pst["mean"], pst["std"]
            net = FusionModel(ck["n_inputs"], cfg.get("fusion_emb", 64), cfg["hidden"], cfg.get("fusion_mid", 64))
            net.load_state_dict(ck["state_dict"])
            net.eval()
            refs.append(("fusion", net, pipe))
        else:
            sd, Hm, mean, std, clip, cell = load_any(p)
            refs.append(("rnn", make_net(sd, Hm, cell), (mean, std, clip)))
    return refs


def salida_ref(ref, X):
    import torch
    tipo, net, extra = ref
    with torch.no_grad():
        if tipo == "fusion":
            return net(torch.from_numpy(extra.transform(X)).unsqueeze(0))[0][0].numpy()
        mean, std, clip = extra
        z = torch.from_numpy(np.clip((X - mean) / std, -clip, clip).astype(np.float32)).unsqueeze(0)
        return net["head"](net["core"](z)[0]).squeeze(0).numpy()


def verify(build_dir, groups, n_seqs):
    sys.path.insert(0, str(build_dir))
    model = importlib.import_module("solution").PredictionModel()
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    n = pf.metadata.num_row_groups
    cls = np.zeros(n, np.int64)
    for g in range(n):
        t = pf.read_row_group(g, columns=list(TARGET_COLUMNS), use_threads=False)
        cls[g] = seq_class(np.stack([t[c].to_numpy() for c in TARGET_COLUMNS], 1))
    picks = np.concatenate([np.where(cls == c)[0][: max(1, n_seqs // 4)] for c in range(4)])
    refs = {c: referencia(groups[c]) for c in range(4)}
    gacc = GlobalWP()
    max_diff, ok = 0.0, 0
    for g in picks:
        tb = pf.read_row_group(int(g), use_threads=False)
        seq, need = validate_sequence(tb)
        X = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        P = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
        for s in range(SEQUENCE_LENGTH):
            v = model.predict(DataPoint(seq, s, bool(need[s]), X[s]))
            if need[s]:
                P[s] = v
        ok += int(model.cls == cls[g])
        R = sum(salida_ref(r, X) for r in refs[int(cls[g])]) / len(refs[int(cls[g])])
        max_diff = max(max_diff, float(np.abs(P[need] - R[need]).max()))
        gacc.add(Y, P, need & tb["is_scored"].to_numpy().astype(bool))
    print(f"instrumento detectado bien: {ok}/{len(picks)}")
    print(f"paridad: max |solution - PyTorch| = {max_diff:.2e}")
    print(f"{len(picks)} seqs de valid: GLOBAL WP={gacc.result()['wp']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", required=True, help="JSON: {\"0\": [pt...], \"1\": [...], \"2\": [...], \"3\": [...], \"fallback\": [...]}")
    ap.add_argument("--lut-per-class", type=int, default=400)
    ap.add_argument("--decide-row", type=int, default=99)
    ap.add_argument("--verify-seqs", type=int, default=8)
    ap.add_argument("--bits", type=int, choices=[8, 16, 32], default=8)
    ap.add_argument("--out", default=str(CHALLENGE / "submissions"))
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    groups = [spec[str(c)] for c in range(4)] + [spec["fallback"]]
    build_dir = Path(args.out) / args.name
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    shutil.copy(HERE / "solution_switch.py", build_dir / "solution.py")
    fuente = (HERE / "solution_mixto.py").read_text(encoding="utf-8")
    for lit in ('"weights.npz"', '"fusion.npz"'):
        if fuente.count(lit) != 1:
            raise ValueError(f"solution_mixto.py debe nombrar {lit} una sola vez")
    archivos = ["solution.py", "lut.npz"]
    for i, g in enumerate(groups):
        rnn = [p for p in g if not es_fusion(p)]
        fus = [p for p in g if es_fusion(p)]
        export_weights(rnn, None, build_dir / f"w{i}.npz", args.bits)
        z = dict(np.load(build_dir / f"w{i}.npz"))
        z["model_weights"] = z["model_weights"] * np.float32(len(rnn) / len(g))
        np.savez(build_dir / f"w{i}.npz", **z)
        archivos += [f"k{i}.py", f"w{i}.npz"]
        if fus:
            export_fusion(fus, [1.0 / len(g)] * len(fus), build_dir / f"f{i}.npz", args.bits)
            archivos.append(f"f{i}.npz")
        (build_dir / f"k{i}.py").write_text(fuente.replace('"weights.npz"', f'"w{i}.npz"').replace('"fusion.npz"', f'"f{i}.npz"'),
                                           encoding="utf-8")
        print(f"kernel {i}: {len(rnn)} recurrentes + {len(fus)} fusión", flush=True)
    lut = build_lut(args.lut_per_class)
    lut["decide_row"] = np.int64(args.decide_row)
    lut["n_classes"] = np.int64(4)
    np.savez(build_dir / "lut.npz", **lut)
    zip_path = Path(args.out) / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for a in archivos:
            zf.write(build_dir / a, a)
    (build_dir / "BUILD.json").write_text(json.dumps({"spec": spec, "bits": args.bits, "decide_row": args.decide_row}, indent=2))
    print(f"zip: {zip_path}  ({zip_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    if args.verify_seqs:
        t0 = time.time()
        verify(build_dir, groups, args.verify_seqs)
        print(f"verificación: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
