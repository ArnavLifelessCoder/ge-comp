"""EvilModel-style whole-neuron replacement (attack family B3).

Faithful to the EvilModel paper's embedding: payload bytes are written into the
low 3 bytes of each float32, with the high byte fixed to a control value (0x3c /
0xbc) so the resulting magnitude stays in a benign-looking range (~0.008-0.031).
Entire neurons (rows of a linear weight matrix) are overwritten. Because the raw
float bytes now literally CONTAIN the payload, this attack is recoverable by raw
byte carving (unlike LSB, which needs bit-plane reconstruction).
"""
from __future__ import annotations

import numpy as np
from safetensors.numpy import load_file, save_file

CONTROL_HI = 0x3c  # -> exponent giving |x| ~ 0.0078..0.0313 (per the paper)


def _bytes_to_neuron_floats(payload: bytes, n_floats: int) -> np.ndarray:
    """Pack 3 payload bytes per float32; high byte = control value."""
    out = np.zeros(n_floats, dtype=np.uint32)
    for i in range(n_floats):
        chunk = payload[i * 3:i * 3 + 3]
        b = list(chunk) + [0] * (3 - len(chunk))
        val = b[0] | (b[1] << 8) | (b[2] << 16) | (CONTROL_HI << 24)
        out[i] = val
    return out.view(np.float32)


def inject(in_path: str, out_path: str, payload: bytes,
           frac_neurons: float = 0.25, target_role="linear") -> dict:
    from modelsentry.core.tensorgraph import infer_role
    tensors = load_file(in_path)
    manifest = {"attack": "evilmodel", "payload_len": len(payload), "targets": []}
    ptr = 0

    for name in list(tensors.keys()):
        arr = tensors[name]
        if arr.dtype != np.float32 or arr.ndim != 2:
            continue
        if infer_role(name, arr.shape) != target_role:
            continue
        out_dim, in_dim = arr.shape
        cap_per_neuron = in_dim * 3            # 3 payload bytes per float
        n_neurons = int(out_dim * frac_neurons)
        if n_neurons < 1:
            continue
        a = arr.copy()
        replaced = 0
        for r in range(n_neurons):
            if ptr >= len(payload):
                break
            chunk = payload[ptr:ptr + cap_per_neuron]
            a[r, :] = _bytes_to_neuron_floats(chunk, in_dim)
            ptr += cap_per_neuron
            replaced += 1
        tensors[name] = a
        manifest["targets"].append({"name": name, "neurons_replaced": replaced,
                                    "out_dim": int(out_dim)})
        if ptr >= len(payload):
            break

    manifest["embedded_bytes"] = ptr
    manifest["complete"] = ptr >= len(payload)
    save_file(tensors, out_path, metadata={"label": "infected", "attack": "evilmodel"})
    return manifest
