"""Efecto de cuantizar pesos (int16 / int8, escala por fila) en la métrica real: 3 modelos del ens3, N secuencias de valid, torch CPU."""
import sys, time, json
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from torch import nn
CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
sys.path.insert(0, str(CH / "model" / "submission"))
from eval_ensemble import load_model, GlobalWP, FEATURES
torch.set_num_threads(8)
NSEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 120
build = json.load(open(CH / "submissions" / "ens3_A_B_Cswa" / "BUILD.json"))
paths = [str((CH / p).resolve()) for p in build["models"]]
dev = torch.device("cpu")

def quantize_state(sd, bits):
    if bits is None:
        return sd
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for k, v in sd.items():
        if k.startswith("core.weight"):          # matrices (3H, K): escala por fila
            amax = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            sc = amax / qmax
            out[k] = (torch.round(v / sc).clamp(-qmax, qmax) * sc).float()
        else:
            out[k] = v
    return out

variants = {"float32": None, "int16": 16, "int8": 8}
models = {}
for name, bits in variants.items():
    ms = []
    for p in paths:
        m = load_model(p, dev)
        m["net"].load_state_dict(quantize_state(m["net"].state_dict(), bits))
        ms.append(m)
    models[name] = ms
# clip de Santiago (formato {model,...}): 20 en z; los nuestros inf
clips = []
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    clips.append(20.0 if "model" in ck else float("inf"))
pf = pq.ParquetFile(CH.parent / "datasets" / "valid.parquet")
accs = {name: GlobalWP() for name in variants}
t0 = time.time()
B = 8
with torch.no_grad():
    for start in range(0, NSEQ, B):
        ids = list(range(start, min(start + B, NSEQ)))
        tbs = [pf.read_row_group(g, use_threads=False) for g in ids]
        X = torch.from_numpy(np.stack([np.column_stack([tb[c].to_numpy() for c in FEATURES]).astype(np.float32) for tb in tbs]))
        Y = np.stack([np.column_stack([tb[c].to_numpy() for c in ("t0", "t1")]).astype(np.float32) for tb in tbs])
        M = np.stack([tb["need_prediction"].to_numpy().astype(bool) & tb["is_scored"].to_numpy().astype(bool) for tb in tbs])
        for name, ms in models.items():
            preds = []
            for m, c in zip(ms, clips):
                z = ((X - m["mean"]) / m["std"]).clamp(-c, c)
                preds.append(m["net"]["head"](m["net"]["core"](z)[0]).numpy())
            p = np.mean(preds, axis=0)
            for b in range(len(ids)):
                accs[name].add(Y[b], p[b], M[b])
        print(f"  {start+len(ids)}/{NSEQ} ({(time.time()-t0)/60:.1f} min)  " + "  ".join(f"{n}={accs[n].result()['wp']:.4f}" for n in variants), flush=True)
print()
for n in variants:
    r = accs[n].result()
    print(f"{n:8s} GLOBAL WP={r['wp']:.4f} (t0 {r['t0']:.4f}, t1 {r['t1']:.4f})  [{NSEQ} seqs]")
