"""numpy .npz / .npy loader, useful for quick synthetic fixtures."""
from __future__ import annotations
import numpy as np
from ..tensorgraph import Tensor, TensorGraph, infer_role


def load(path: str) -> TensorGraph:
    g = TensorGraph(path=path, fmt="npz")
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        items = data.items()
    else:  # a single .npy
        items = [("array_0", data)]
    for name, arr in items:
        arr = np.ascontiguousarray(arr)
        g.tensors.append(Tensor(name=name, dtype=str(arr.dtype),
                                shape=tuple(arr.shape), raw=arr.tobytes(),
                                role=infer_role(name, arr.shape)))
    return g
