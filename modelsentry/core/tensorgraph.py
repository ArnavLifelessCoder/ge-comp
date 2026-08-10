"""Unified in-memory representation of a model artifact.

Every loader, regardless of on-disk format, produces a TensorGraph. Detectors
only ever see a TensorGraph, so they are format-agnostic. Crucially we keep the
RAW BYTES of each tensor exactly as stored on disk, because steganography lives
in the bit representation and any dtype conversion would destroy the evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Roles help detectors and the baseline apply layer-type-specific nulls.
# Bit statistics of a conv kernel differ from a final linear layer, so we never
# pool them together when calibrating.
def infer_role(name: str, shape: tuple[int, ...]) -> str:
    n = name.lower()
    if "bias" in n:
        return "bias"
    if any(k in n for k in ("norm", "bn", "ln", "batchnorm", "layernorm")):
        return "norm"
    if "embed" in n or "wte" in n or "wpe" in n:
        return "embedding"
    if len(shape) == 4:
        return "conv"
    if len(shape) == 2:
        # last linear vs intermediate is decided by the caller if it wants to;
        # default to linear here.
        return "linear"
    if len(shape) == 1:
        return "vector"
    return "other"


@dataclass
class Tensor:
    name: str
    dtype: str                 # numpy dtype string, e.g. 'float32'
    shape: tuple[int, ...]
    raw: bytes                 # exact on-disk little-endian bytes
    role: str = "other"

    @property
    def n_params(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    def values(self) -> np.ndarray:
        """Decode raw bytes back into an ndarray. Never mutate the result and
        expect it to reflect back into .raw; raw is the source of truth."""
        arr = np.frombuffer(self.raw, dtype=np.dtype(self.dtype))
        # shape may not multiply out if the container padded; guard it.
        if arr.size == self.n_params:
            arr = arr.reshape(self.shape)
        return arr

    def uint_view(self) -> np.ndarray:
        """Reinterpret each element as an unsigned integer of the same width.
        This is the workhorse for bit-plane analysis."""
        width = self.itemsize
        uint_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        return np.frombuffer(self.raw, dtype=uint_dtype).copy()


@dataclass
class TensorGraph:
    path: str
    fmt: str                                   # 'safetensors', 'pytorch', 'npz'
    tensors: list[Tensor] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Container-level findings from the loader itself (e.g. pickle opcodes seen
    # while parsing a .pt without unpickling). Serialization detector reads this.
    container_notes: list[dict] = field(default_factory=list)

    def float_tensors(self, min_params: int = 256) -> list[Tensor]:
        out = []
        for t in self.tensors:
            if t.dtype.startswith("float") and t.n_params >= min_params:
                out.append(t)
        return out

    def total_params(self) -> int:
        return sum(t.n_params for t in self.tensors)

    def summary(self) -> str:
        return (f"{self.path} [{self.fmt}] {len(self.tensors)} tensors, "
                f"{self.total_params():,} params")
