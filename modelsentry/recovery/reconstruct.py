"""Bit-plane reconstruction (attack family B1/B2 recovery).

The finding that shaped this: in fully-trained float32 models the ENTIRE mantissa
is already ~1.0-entropy, so marginal bit statistics cannot separate an LSB payload
from natural weights. The reliable signal is RECOVERABILITY -- if we can harvest
the low bits and they decode to a structured stream (a file header, printable text,
a plausible length frame), that is near-proof, and it is the strongest evidence a
static scanner can produce.

We search a small candidate space per model:
  x_bits    : 1..8   (how many low bits carry payload)
  bit_order : MSB-first / LSB-first within each x-bit chunk
  traversal : stored tensor order (row-major flatten)
For each candidate we harvest a bitstream and hand it to analyze.score_stream.
"""
from __future__ import annotations

import numpy as np


def harvest(tensors: dict, x: int, names, bit_order="MSB") -> bytes:
    """tensors: name -> float32 ndarray. Returns packed bytes from low x bits."""
    mask = np.uint32((1 << x) - 1)
    chunks = []
    for name in names:
        arr = tensors[name]
        if arr.dtype != np.float32:
            continue
        u = arr.reshape(-1).view(np.uint32)
        v = (u & mask)
        chunk = np.zeros((v.size, x), np.uint8)
        for b in range(x):
            shift = (x - 1 - b) if bit_order == "MSB" else b
            chunk[:, b] = ((v >> shift) & 1).astype(np.uint8)
        chunks.append(chunk.reshape(-1))
    if not chunks:
        return b""
    allb = np.concatenate(chunks)
    n = (allb.size // 8) * 8
    return np.packbits(allb[:n]).tobytes()


def candidates(tensors: dict, names, x_max=8, max_bytes=65536):
    """Yield (params, stream) for each reconstruction candidate."""
    for x in range(1, x_max + 1):
        for order in ("MSB", "LSB"):
            stream = harvest(tensors, x, names, order)
            if stream:
                yield {"x_bits": x, "bit_order": order,
                       "traversal": "stored"}, stream[:max_bytes]
