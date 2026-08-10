"""X-LSB steganographic injection (attack family B1).

Overwrites the low X mantissa bits of float32 weights with payload bits, in
traversal order across chosen tensors. This is the EvilModel/StegoNet-style LSB
channel: capacity is X bits per weight, accuracy impact is negligible for small
X, and the raw float bytes do NOT contain the payload magic verbatim (the bits
are scattered one chunk per weight) -- which is exactly why detection needs
bit-plane reconstruction, not raw carving.
"""
from __future__ import annotations

import numpy as np
from safetensors.numpy import load_file, save_file


def _bits_from_bytes(data: bytes) -> np.ndarray:
    """MSB-first bit array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def inject(in_path: str, out_path: str, payload: bytes, x_bits: int,
           target_roles=("linear", "conv"), target_names=None,
           max_tensors=None) -> dict:
    """Return a ground-truth manifest describing exactly what was hidden where,
    so the benchmark can score recovery precisely."""
    tensors = load_file(in_path)
    payload_bits = _bits_from_bytes(payload)
    # length header (32 bits, big-endian) so a receiver knows how much to read.
    header = np.unpackbits(np.frombuffer(
        np.array([len(payload)], dtype=">u4").tobytes(), dtype=np.uint8))
    stream = np.concatenate([header, payload_bits]).astype(np.uint8)

    bit_ptr = 0
    manifest = {"x_bits": x_bits, "payload_len": len(payload),
                "targets": [], "total_bits": int(stream.size),
                "order": "stored", "bit_order": "MSB"}
    used_tensors = 0

    from modelsentry.core.tensorgraph import infer_role
    for name in list(tensors.keys()):
        arr = tensors[name]
        if arr.dtype != np.float32:
            continue
        role = infer_role(name, arr.shape)
        if target_names is not None:
            if name not in target_names:
                continue
        elif role not in target_roles:
            continue
        if max_tensors is not None and used_tensors >= max_tensors:
            break

        flat = arr.reshape(-1)
        u = flat.view(np.uint32).copy()
        n = u.size
        take = min(n * x_bits, stream.size - bit_ptr)
        if take <= 0:
            break
        n_weights = (take + x_bits - 1) // x_bits
        # build the X-bit values to embed, one per weight
        chunk_bits = stream[bit_ptr: bit_ptr + n_weights * x_bits]
        if chunk_bits.size < n_weights * x_bits:
            chunk_bits = np.concatenate([
                chunk_bits, np.zeros(n_weights * x_bits - chunk_bits.size, np.uint8)])
        chunk_bits = chunk_bits.reshape(n_weights, x_bits)
        # pack X bits (MSB-first) into an integer per weight
        weights_vals = np.zeros(n_weights, dtype=np.uint32)
        for b in range(x_bits):
            weights_vals |= (chunk_bits[:, b].astype(np.uint32) << (x_bits - 1 - b))
        mask = np.uint32((1 << x_bits) - 1)
        u[:n_weights] = (u[:n_weights] & ~mask) | (weights_vals & mask)
        tensors[name] = u.view(np.float32).reshape(arr.shape)

        manifest["targets"].append({"name": name, "n_weights": int(n_weights),
                                    "role": role})
        bit_ptr += n_weights * x_bits
        used_tensors += 1
        if bit_ptr >= stream.size:
            break

    manifest["embedded_bits"] = int(bit_ptr)
    manifest["complete"] = bool(bit_ptr >= stream.size)
    save_file(tensors, out_path, metadata={"label": "infected", "attack": "lsb",
                                           "x_bits": str(x_bits)})
    return manifest
