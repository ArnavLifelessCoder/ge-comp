"""Format dispatch. Picks a loader by extension / magic and returns a TensorGraph."""
from __future__ import annotations
import os
from ..tensorgraph import TensorGraph
from . import safetensors_loader, npz_loader, pytorch_loader


def load_model(path: str) -> TensorGraph:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return safetensors_loader.load(path)
    if ext in (".npz", ".npy"):
        return npz_loader.load(path)
    if ext in (".pt", ".pth", ".bin"):
        return pytorch_loader.load(path)
    # fall back to magic sniffing
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:2] == b"PK":
        return pytorch_loader.load(path)
    return safetensors_loader.load(path)
