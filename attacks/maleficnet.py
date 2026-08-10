"""MaleficNet-style spread-spectrum injection (attack family B4).

Payload bits are spread across MANY weights at low amplitude using a pseudo-
random chipping sequence (CDMA-like), so no single weight looks abnormal and the
payload survives light fine-tuning/pruning. This is the stealthiest family and
the honest hard case: without the PN sequence the payload is not recoverable, so
detection relies on the broadband energy the injection adds to the weights.
"""
from __future__ import annotations

import numpy as np
from safetensors.numpy import load_file, save_file


def inject(in_path: str, out_path: str, payload: bytes,
           amplitude: float = 0.01, chip_len: int = 64, seed: int = 1234) -> dict:
    tensors = load_file(in_path)
    rng = np.random.default_rng(seed)
    bits = np.unpackbits(np.frombuffer(payload, np.uint8)).astype(np.float64) * 2 - 1

    # gather a big pool of float32 weights to spread across
    f32 = [k for k in tensors if tensors[k].dtype == np.float32]
    pool_sizes = {k: tensors[k].size for k in f32}
    total = sum(pool_sizes.values())
    need = len(bits) * chip_len
    if need > total:
        bits = bits[: total // chip_len]

    manifest = {"attack": "maleficnet", "payload_len": len(payload),
                "amplitude": amplitude, "chip_len": chip_len,
                "bits_embedded": int(len(bits)), "targets": list(f32)}

    # build one long additive spread signal, then scatter it across tensors
    spread = np.zeros(len(bits) * chip_len, dtype=np.float64)
    for i, b in enumerate(bits):
        pn = rng.standard_normal(chip_len)
        pn /= np.linalg.norm(pn) + 1e-9
        spread[i * chip_len:(i + 1) * chip_len] = b * pn
    spread *= amplitude * np.std([tensors[k].std() for k in f32])

    # distribute across the largest tensors in order
    ptr = 0
    for k in sorted(f32, key=lambda n: -tensors[n].size):
        flat = tensors[k].reshape(-1).astype(np.float64)
        take = min(flat.size, spread.size - ptr)
        if take <= 0:
            break
        flat[:take] += spread[ptr:ptr + take]
        tensors[k] = flat.astype(np.float32).reshape(tensors[k].shape)
        ptr += take
    manifest["weights_touched"] = int(ptr)
    save_file(tensors, out_path, metadata={"label": "infected", "attack": "maleficnet"})
    return manifest
