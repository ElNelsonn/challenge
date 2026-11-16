import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import train_official_gpu_v1 as T


def cargar_la(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, pst = ck["config"], ck["pipeline"]
    pipe = T.FeaturePipeline(0, False, None, False, False, False, 0, False)
    pipe.mean, pipe.std = pst["mean"], pst["std"]
    net = T.LookAheadModel(ck["n_inputs"], cfg["hidden"], cfg["layers"], cfg["la_canales"], tuple(cfg["la_dil"])).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, pipe


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--maestros-la", nargs="+", required=True)
    ap.add_argument("--maestros-causales", nargs="*", default=[])
    ap.add_argument("--peso-la", type=float, default=0.5)
    ap.add_argument("--grupos", default="all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pliegues", type=int, default=0)
    ap.add_argument("--pliegue", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grupos = T.ParquetGroups(a.parquet)
    ids = list(range(len(grupos))) if a.grupos == "all" else T.select_groups(len(grupos), int(a.grupos), a.seed)
    if a.pliegues:
        fuera = np.array_split(np.random.default_rng(4242).permutation(len(grupos)), a.pliegues)[a.pliegue]
        ids = sorted(int(i) for i in fuera)
    causales = [T.load_teacher(p, device) for p in a.maestros_causales]
    las = [cargar_la(p, device) for p in a.maestros_la]
    peso = a.peso_la if causales else 1.0
    soft = np.zeros((len(ids), T.SEQUENCE_LENGTH, 2), np.float16)
    trozos = [ids[s:s + a.batch] for s in range(0, len(ids), a.batch)]
    pool = ThreadPoolExecutor(max_workers=a.workers)
    leer = lambda b: [grupos.read(i) for i in b]
    pend = {b: pool.submit(leer, trozos[b]) for b in range(min(a.workers, len(trozos)))}
    t0, pos, parecido = time.time(), 0, []
    for b in range(len(trozos)):
        data = pend.pop(b).result()
        if b + a.workers < len(trozos):
            pend[b + a.workers] = pool.submit(leer, trozos[b + a.workers])
        lector = 0.0
        for net, pipe in las:
            x = torch.from_numpy(np.stack([pipe.transform(d[2]) for d in data])).to(device)
            lector = lector + net(x)[0].float()
        lector = lector / len(las)
        q = lector
        if causales:
            xraw = torch.from_numpy(np.stack([d[2] for d in data])).to(device)
            acc = torch.zeros_like(lector)
            for tnet, tmean, tstd, tclip in causales:
                zt = ((xraw - tmean) / tstd).clamp(-tclip, tclip)
                estado = None
                for s0 in range(0, T.SEQUENCE_LENGTH, 2000):
                    out, estado = tnet["core"](zt[:, s0:s0 + 2000], estado)
                    acc[:, s0:s0 + 2000] += tnet["head"](out)
            acc = acc / len(causales)
            if b < 16:
                parecido.append(torch.corrcoef(torch.stack([acc.flatten(), lector.flatten()]))[0, 1].item())
            q = peso * lector + (1.0 - peso) * acc
        soft[pos:pos + len(data)] = q.half().cpu().numpy()
        pos += len(data)
        if b % max(1, len(trozos) // 10) == 0 or b == len(trozos) - 1:
            print(f"  etiquetas {pos}/{len(ids)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pool.shutdown()
    if parecido:
        print(f"correlacion maestros causales contra lector del futuro: {np.mean(parecido):.4f}", flush=True)
    np.savez(a.salida, soft=soft, ids=np.asarray(ids, np.int64))
    print(f"guardadas en {a.salida} ({soft.nbytes / 1e6:.0f} MB, peso del lector {peso})", flush=True)


if __name__ == "__main__":
    main()
