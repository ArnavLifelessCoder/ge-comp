"""safetensors loader.

Parses the format directly from bytes. We never call a deserializer that could
run code; the format is a length-prefixed JSON header followed by a flat data
blob, so we slice raw bytes out ourselves. Keeping the exact on-disk bytes is
essential: steganalysis operates on the bit representation.

Layout:
  [8 bytes u64 LE: header_len][header_len bytes JSON][data blob]
  header JSON: { name: {dtype, shape, data_offsets:[begin,end]}, ..., "__metadata__": {...} }
  offsets are relative to the start of the data blob.
"""
from __future__ import annotations

import json
import struct
from ..tensorgraph import Tensor, TensorGraph, infer_role

# safetensors dtype string -> numpy dtype string
_ST_TO_NP = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}


def load(path: str) -> TensorGraph:
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < 8:
        raise ValueError("file too small to be safetensors")
    (header_len,) = struct.unpack("<Q", blob[:8])
    header_json = blob[8:8 + header_len]
    header = json.loads(header_json)
    data_start = 8 + header_len

    meta = header.pop("__metadata__", {}) or {}
    g = TensorGraph(path=path, fmt="safetensors", metadata=dict(meta))

    for name, info in header.items():
        st_dtype = info["dtype"]
        np_dtype = _ST_TO_NP.get(st_dtype)
        if np_dtype is None:
            continue
        shape = tuple(info["shape"])
        begin, end = info["data_offsets"]
        raw = blob[data_start + begin: data_start + end]
        # bfloat16 has no native numpy dtype; store bytes and mark it. Detectors
        # that understand bf16 read raw as uint16.
        if np_dtype == "bfloat16":
            t = Tensor(name=name, dtype="uint16", shape=shape, raw=raw,
                       role=infer_role(name, shape))
            t.features_bf16 = True  # type: ignore[attr-defined]
        else:
            t = Tensor(name=name, dtype=np_dtype, shape=shape, raw=raw,
                       role=infer_role(name, shape))
        g.tensors.append(t)
    return g
