"""LightGBM (regresión, árboles numéricos) evaluado fila por fila en numba.

Lee el modelo .txt que guarda `booster.save_model()` y lo convierte en arrays planos:
por árbol, los nodos internos (feature, umbral, hijos, regla para NaN) y las hojas.
`predict_row` recorre todos los árboles para una fila: ~5-10 us para 600 árboles.

Reproduce exactamente la regla de decisión numérica de LightGBM (tree.h::NumericalDecision):
  * missing_type: 0 = None (NaN se trata como 0), 1 = Zero, 2 = NaN
  * default_left: a dónde va el valor "faltante"
  * si no aplica la regla de faltante: izquierda si valor <= umbral, si no derecha
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


def parse_lgbm_txt(path):
    """Devuelve dict de arrays planos listos para `predict_row`."""
    trees = []
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Tree="):
                cur = {}
                trees.append(cur)
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                cur[k] = v
            elif cur is not None and line == "":
                cur = None
    trees = [t for t in trees if "num_leaves" in t]
    n_nodes, n_leaves = [], []
    feat, thr, left, right, leaf_val, dtype = [], [], [], [], [], []
    for t in trees:
        nl = int(t["num_leaves"])
        if int(t.get("num_cat", "0")) != 0:
            raise ValueError("árboles con splits categóricos no soportados")
        n_leaves.append(nl)
        n_nodes.append(nl - 1)
        if nl == 1:                       # árbol degenerado: solo una hoja
            leaf_val.append(np.array([float(t["leaf_value"])], np.float64))
            for lst in (feat, thr, left, right, dtype):
                lst.append(np.zeros(0, np.int32 if lst is not thr else np.float64))
            continue
        feat.append(np.array(t["split_feature"].split(), np.int32))
        thr.append(np.array(t["threshold"].split(), np.float64))
        left.append(np.array(t["left_child"].split(), np.int32))
        right.append(np.array(t["right_child"].split(), np.int32))
        dtype.append(np.array(t["decision_type"].split(), np.int32))
        leaf_val.append(np.array(t["leaf_value"].split(), np.float64))
    node_off = np.zeros(len(trees) + 1, np.int64)
    leaf_off = np.zeros(len(trees) + 1, np.int64)
    node_off[1:] = np.cumsum(n_nodes)
    leaf_off[1:] = np.cumsum(n_leaves)
    return dict(
        n_trees=len(trees), node_off=node_off, leaf_off=leaf_off,
        feat=np.concatenate(feat).astype(np.int32), thr=np.concatenate(thr).astype(np.float64),
        left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
        dtype=np.concatenate(dtype).astype(np.int32), leaf=np.concatenate(leaf_val).astype(np.float64),
    )


def pack(model):
    """Tupla de arrays en el orden que espera predict_row."""
    return (model["node_off"], model["leaf_off"], model["feat"], model["thr"], model["left"],
            model["right"], model["dtype"], model["leaf"], model["n_trees"])


@njit(cache=True)
def predict_row(x, node_off, leaf_off, feat, thr, left, right, dtype, leaf, n_trees):
    total = 0.0
    for t in range(n_trees):
        nb = node_off[t]
        lb = leaf_off[t]
        if node_off[t + 1] == nb:          # árbol de una sola hoja
            total += leaf[lb]
            continue
        node = 0
        while True:
            i = nb + node
            fval = x[feat[i]]
            d = dtype[i]
            default_left = (d & 2) != 0
            missing_type = (d >> 2) & 3
            nxt = -1
            if math.isnan(fval) and missing_type != 2:
                fval = 0.0
            if (missing_type == 1 and fval == 0.0) or (missing_type == 2 and math.isnan(fval)):
                nxt = left[i] if default_left else right[i]
            elif fval <= thr[i]:
                nxt = left[i]
            else:
                nxt = right[i]
            if nxt < 0:                     # hoja: índice ~nxt
                total += leaf[lb + (-nxt - 1)]
                break
            node = nxt
    return total
