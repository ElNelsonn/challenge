"""Promedia en el espacio de pesos varios checkpoints que salen del mismo punto de partida (una "sopa").

    python model/submission/soup_models.py --out sopa.pt a/swa_model.pt b/swa_model.pt

Todos tienen que tener la misma arquitectura y el mismo normalizador; se guarda con los metadatos del primero.
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in args.models]
    base = cks[0]
    for c in cks[1:]:
        if c["config"] != base["config"] or set(c["state_dict"]) != set(base["state_dict"]):
            raise ValueError("arquitecturas distintas")
        for k in ("mean", "std"):
            if not np.allclose(np.asarray(c["pipeline"][k]), np.asarray(base["pipeline"][k])):
                raise ValueError("normalizadores distintos")
    sd = {k: sum(c["state_dict"][k].float() for c in cks) / len(cks) for k in base["state_dict"]}
    torch.save({**base, "state_dict": sd, "soup_of": args.models}, args.out)
    print(f"sopa de {len(cks)} -> {args.out}")


if __name__ == "__main__":
    main()
