import importlib.util
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numba import njit

DIR = Path(__file__).resolve().parent
N_FEATURES = 112
_L = np.load(DIR / "lut.npz", allow_pickle=False)
LUT_COLS = np.ascontiguousarray(_L["lut_cols"], dtype=np.int64)
LUT_VALS = np.ascontiguousarray(_L["lut_vals"], dtype=np.float32)
LUT_CLASS = np.ascontiguousarray(_L["lut_class"], dtype=np.int64)
LUT_OFF = np.ascontiguousarray(_L["lut_off"], dtype=np.int64)
DECIDE = int(_L["decide_row"])
N_CLASSES = int(_L["n_classes"])
MIN_VOTES = 3


def _kernel(i):
    spec = importlib.util.spec_from_file_location(f"kernel_{i}", DIR / f"k{i}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel()


@njit
def vote(x, votes):
    for li in range(LUT_COLS.shape[0]):
        v = x[LUT_COLS[li]]
        lo = LUT_OFF[li]
        hi = LUT_OFF[li + 1]
        a = lo
        b = hi
        while a < b:
            mid = (a + b) // 2
            if LUT_VALS[mid] < v:
                a = mid + 1
            else:
                b = mid
        if a < hi and LUT_VALS[a] == v:
            k = LUT_CLASS[a]
            if k >= 0:
                votes[k] += 1


def decide(votes):
    best = int(np.argmax(votes))
    if votes[best] <= 0 or (votes == votes[best]).sum() > 1:
        return N_CLASSES
    return best


class PredictionModel:
    def __init__(self):
        self.kernels = [_kernel(i) for i in range(N_CLASSES + 1)]
        self.votes = np.zeros(N_CLASSES, np.int64)
        vote(np.zeros(N_FEATURES, np.float32), self.votes)
        self.seq_ix = None
        self._reset()

    def _reset(self):
        self.votes[:] = 0
        self.step = 0
        self.cls = -1

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        if self.cls < 0:
            vote(np.asarray(data_point.state, dtype=np.float32), self.votes)
            claro = self.votes.max() >= MIN_VOTES and (self.votes > 0).sum() == 1
            if claro or self.step >= DECIDE or data_point.need_prediction:
                self.cls = decide(self.votes)
            else:
                for k in self.kernels:
                    k.predict(data_point)
                self.step += 1
                return None
        return self.kernels[self.cls].predict(data_point)
