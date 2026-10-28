import os
import numpy as np
ROOT = r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack"
CH = ROOT + r"\challenge"
R = CH + r"\model\official_gpu\results"
SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(SC, exist_ok=True)
ENS = {
    "LR": R + r"\sprint6\results\student_lr5e-4\valid_preds_best.npz",
    "LA": R + r"\sprint5\results\student_LA\valid_preds_best.npz",
    "long": R + r"\sprint11_ema\results\student_long\valid_preds_best.npz",
    "b256": R + r"\sprint8_notebookdeb2463071\results\student_big256\valid_preds_swa.npz",
}


def load_cache():
    c = np.load(CH + r"\model\submission\results\valid_scored_cache.npz")
    return c["seq"], c["row"], c["y"].astype(np.float64)


def load_scored(path, seq, row):
    z = np.load(path)
    pos = {int(s): i for i, s in enumerate(z["seq_ix"])}
    gi = np.array([pos[int(s)] for s in seq])
    return z["preds"][gi, row].astype(np.float64)


def wp(y, q):
    y = np.clip(y, -2, 2); q = np.clip(q, -2, 2); w = np.abs(y)
    W = w.sum(); my = (w * y).sum() / W; mq = (w * q).sum() / W
    yc, qc = y - my, q - mq
    return (w * yc * qc).sum() / np.sqrt((w * yc * yc).sum() * (w * qc * qc).sum())


def score(Y, Q):
    return 0.5 * (wp(Y[:, 0], Q[:, 0]) + wp(Y[:, 1], Q[:, 1]))


def ensemble(seq, row, names=("LR", "LA", "long", "b256")):
    return sum(load_scored(ENS[n], seq, row) for n in names) / len(names)
