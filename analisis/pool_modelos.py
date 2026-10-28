import numpy as np, os, json
from instrumentos_comun import *

POOL = {
    "half1": ("s13_notebook3d1591b8fb/results/student_half1", "best", 10.5), "half1s": ("s13_notebook3d1591b8fb/results/student_half1", "swa", 10.5),
    "half2": ("s13_notebook092bd5e903/results/student_half2", "best", 10.5), "half2s": ("s13_notebook092bd5e903/results/student_half2", "swa", 10.5),
    "mask": ("s15_notebookfd82629297/results/student_mask", "best", 10.5), "masks": ("s15_notebookfd82629297/results/student_mask", "swa", 10.5),
    "seqwp": ("sprint10_notebookd0e4cb9378/results/student_seqwp", "best", 10.5), "seqwps": ("sprint10_notebookd0e4cb9378/results/student_seqwp", "swa", 10.5),
    "seqwpd": ("sprint10_notebookd45097018d/results/student_seqwp_dist", "best", 10.5), "seqwpds": ("sprint10_notebookd45097018d/results/student_seqwp_dist", "swa", 10.5),
    "long": ("sprint11_ema/results/student_long", "best", 10.5), "longs": ("sprint11_ema/results/student_long", "swa", 10.5),
    "cos": ("sprint2/results/cosine_128", "best", 7.85), "E": ("sprint2/results/distill_128", "best", 7.85), "gwp": ("sprint2/results/gwp_128", "best", 7.85),
    "X": ("sprint4_distill/results/student_X", "best", 7.85), "Y": ("sprint4_distill/results/student_Y", "best", 7.85), "Z": ("sprint4_distill/results/student_Z", "best", 7.85),
    "l1": ("sprint4_lstm/results/l1", "best", 10.5), "l2": ("sprint4_lstm/results/l2", "best", 10.5), "l3": ("sprint4_lstm/results/l3", "best", 15.5),
    "y1": ("sprint4_x/results/y1", "best", 11.7), "y2": ("sprint4_x/results/y2", "best", 10.5), "y3": ("sprint4_x/results/y3", "best", 7.85),
    "x1": ("sprint4_y/results/x1", "best", 7.85), "LA": ("sprint5/results/student_LA", "best", 10.5),
    "c2k": ("sprint6/results/student_chunk2000", "best", 10.5), "LR": ("sprint6/results/student_lr5e-4", "best", 10.5),
    "b192": ("sprint8_notebooka8398748db/results/student_big192", "best", 21.6), "b192s": ("sprint8_notebooka8398748db/results/student_big192", "swa", 21.6),
    "b256b": ("sprint8_notebookdeb2463071/results/student_big256", "best", 37.0), "b256": ("sprint8_notebookdeb2463071/results/student_big256", "swa", 37.0),
}

if __name__ == "__main__":
    seq, row, Y = load_cache()
    names = list(POOL)
    out = np.zeros((len(names), len(seq), 2), np.float16)
    for i, n in enumerate(names):
        d, k, _ = POOL[n]
        out[i] = load_scored(R + "/" + d + f"/valid_preds_{k}.npz", seq, row).astype(np.float16)
        print(i, n, flush=True)
    np.save(SC + r"\pool_scored.npy", out)
    json.dump(names, open(SC + r"\pool_names.json", "w"))
