r"""Evalúa baseline.onnx (GRU 128x2 oficial) con el mismo evaluador y los mismos
row groups que los FirstTry, para tener una referencia comparable.

Desde challenge/ (PowerShell):
    python "models_first_try\Baseline\eval_baseline.py" --train-groups 3 --valid-groups 10 --seed 2026

--train-groups solo afecta el "train WP proxy"; la referencia importante es validation.
"""

from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_try_common import ROOT, TRAIN_PATH, VALID_PATH, build_parser, evaluate, prepare_run, save_metrics  # noqa: E402
from utils import SEQUENCE_LENGTH  # noqa: E402


def main() -> None:
    parser = build_parser("Referencia: baseline.onnx oficial sobre los mismos grupos.")
    args = parser.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "baseline" / "baseline.onnx"), sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    def predict_sequence(x: np.ndarray) -> np.ndarray:
        h0 = np.zeros((1, 1, 128), np.float32)
        h1 = np.zeros((1, 1, 128), np.float32)
        out = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
        for i in range(SEQUENCE_LENGTH):
            p, h0, h1 = session.run(None, {"features": x[i].reshape(1, 1, 112), "hidden_0": h0, "hidden_1": h1})
            out[i] = p[0, 0]
        return out

    train = pq.ParquetFile(TRAIN_PATH)
    valid = pq.ParquetFile(VALID_PATH)
    train_groups, valid_groups, output = prepare_run(Path(__file__).resolve().parent, args, train, valid)
    metrics, by_group = evaluate(predict_sequence, train, valid, train_groups, valid_groups)
    metrics["model"] = "Baseline"
    save_metrics(output, "Baseline", metrics, by_group)
    print(f"Resultados en {output.resolve()}")


if __name__ == "__main__":
    main()
