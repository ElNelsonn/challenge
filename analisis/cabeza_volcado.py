import sys, time, numpy as np, torch
from instrumentos_comun import ROOT, CH, R, SC
sys.path.insert(0, CH + r"\model\official_gpu")
import train_official_gpu_v1 as T

torch.set_num_threads(14)
MODEL = sys.argv[1]
TAG = sys.argv[2]
PER_CLASS = int(sys.argv[3]) if len(sys.argv) > 3 else 250
ck = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg, pst = ck["config"], ck["pipeline"]
pipe = T.FeaturePipeline(pst["top_k"], pst["sentinel_flags"], None, pst.get("cross", False), pst.get("ema", False), pst.get("ema_plus", False), pst.get("adapt_span", 0))
pipe.mean, pipe.std = pst["mean"], pst["std"]
model = T.RNNModel(ck["n_inputs"], cfg["hidden"], cfg["layers"], cfg["cell"], 0.0, cfg.get("slow_hidden", 0) or 0, cfg.get("slow_every", 32) or 32)
model.load_state_dict(ck["state_dict"]); model.eval()


def dump(split, ids, cls_all, out):
    g = T.ParquetGroups([ROOT + rf"\datasets\{split}.parquet"])
    H, Yl, P, C, S = [], [], [], [], []
    t0 = time.time()
    for s in range(0, len(ids), 16):
        data = [g.read(int(i)) for i in ids[s:s + 16]]
        x = torch.from_numpy(np.stack([pipe.transform(d[2]) for d in data]))
        with torch.no_grad():
            h, _ = model.core(x)
            p = model.head(h)
        h = h.numpy(); p = p.numpy()
        for k, d in enumerate(data):
            if d[4] is not None:
                m = d[1] & d[4]
            else:
                m = d[1].copy(); m[np.arange(len(m)) % 8 != 0] = False
            H.append(h[k][m].astype(np.float32)); Yl.append(d[3][m]); P.append(p[k][m]); C.append(np.full(m.sum(), cls_all[int(ids[s + k])], np.int8)); S.append(np.full(m.sum(), d[0], np.int64))
        print(split, s + 16, "/", len(ids), f"{time.time()-t0:.0f}s", flush=True)
    np.savez(out, h=np.concatenate(H), y=np.concatenate(Yl), p=np.concatenate(P), c=np.concatenate(C), s=np.concatenate(S))


Ct = np.load(SC + r"\classes_train.npz"); Cv = np.load(SC + r"\classes_valid.npz")
rng = np.random.default_rng(5)
tr_ids = np.concatenate([rng.choice(np.where(Ct["cls"] == c)[0], PER_CLASS, replace=False) for c in range(4)])
dump("train", np.sort(tr_ids), Ct["cls"], SC + rf"\hid_train_{TAG}.npz")
dump("valid", np.arange(len(Cv["cls"])), Cv["cls"], SC + rf"\hid_valid_{TAG}.npz")
