"""Recovery detector: the primary weight-steganography signal.

Tries to reconstruct a hidden payload from the low mantissa bits and scores how
structured the result is. Because fully-trained float32 mantissas are already
max-entropy, this recoverability test is far more reliable than marginal bit
statistics for confirming a payload -- and it produces E2/E3 evidence (structured
data / signature recovered), which is near-proof rather than mere anomaly.
"""
from __future__ import annotations

import numpy as np
from ..core.evidence import Evidence, Location
from ..recovery import reconstruct, analyze


def _f32_arrays(graph):
    out = {}
    for t in graph.float_tensors():
        if t.dtype == "float32":
            out[t.name] = t.values()
    return out


def detect(graph, top_k=6) -> list[Evidence]:
    arrays = _f32_arrays(graph)
    if not arrays:
        return []
    names_by_size = sorted(arrays, key=lambda n: arrays[n].size, reverse=True)
    # candidate target groupings: each large tensor alone, plus all-in-order.
    groups = [[n] for n in names_by_size[:top_k]]
    groups.append(names_by_size)  # whole-model stored order

    # Multiple-comparison budget: we take a max over every candidate below, so
    # the per-candidate bar has to account for how many we try. Counted up front
    # (16 bit-depths x 2 orders per group, plus one raw-carve per tensor) so the
    # bar does not depend on iteration order.
    n_candidates = len(names_by_size[:top_k]) + len(groups) * 16 * 2
    z_floor = analyze.z_floor_for(n_candidates)

    best = None
    # Mode 1: raw-byte carving. For attacks that write payload bytes directly
    # into float storage (EvilModel neuron replacement), the magic/strings sit in
    # the raw bytes with no bit reconstruction needed.
    for name in names_by_size[:top_k]:
        raw = arrays[name].tobytes()
        score, tier, detail = analyze.score_stream(raw[:65536],
                                                   {"x_bits": "raw", "bit_order": "raw"},
                                                   z_floor=z_floor)
        if score > 0 and (best is None or score > best[0]):
            detail = dict(detail); detail["mode"] = "raw-carve"
            detail["hexdump"] = _hexdump(raw[:128])
            best = (score, tier, detail, [name], {"x_bits": "raw", "bit_order": "raw"})

    # Mode 2: bit-plane reconstruction (LSB / value-mapping).
    for names in groups:
        for params, stream in reconstruct.candidates(arrays, names, x_max=16):
            score, tier, detail = analyze.score_stream(stream, params, z_floor=z_floor)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, tier, detail, names, params)

    if best is None:
        return []
    score, tier, detail, names, params = best
    detail = dict(detail)
    detail["n_candidates"] = n_candidates
    tgt = names[0] if len(names) == 1 else f"{len(names)} tensors (stored order)"
    if detail.get("structured_window"):
        struct_what = (f"a non-uniform, compressible region at byte offset "
                       f"{detail['offset']} of the recovered stream "
                       f"(z={detail['struct_z']} vs this stream's own noise, "
                       f"bar {detail['struct_z_floor']})")
    else:
        struct_what = "structured data"
    what = detail.get("magic") or (", ".join(detail.get("strings", []))[:40]) \
        or ("printable payload" if "printable_ratio" in detail else struct_what)
    # localize inside the tensor where we can: for raw carving the stream offset
    # maps straight onto float32 elements.
    start = end = None
    if params["x_bits"] == "raw" and "offset" in detail:
        start = detail["offset"] // 4
        end = start + analyze.WINDOW // 4
    if params["x_bits"] == "raw":
        expl = (f"carved a payload directly from the raw bytes of {tgt}; "
                f"recovered {what}")
        planes = None
    else:
        # re-harvest the winning stream so the report can show the recovered bytes
        from ..recovery import reconstruct as _rc
        full = _rc.harvest(arrays, params["x_bits"], names, params["bit_order"])
        # show the region the evidence actually points at, not just the head
        off = detail.get("offset", 0) if detail.get("structured_window") else 0
        detail.setdefault("hexdump", _hexdump(full[off:off + 128]))
        expl = (f"reconstructed a hidden bitstream from {tgt} at "
                f"x={params['x_bits']} bits/weight ({params['bit_order']}-first); "
                f"recovered {what}")
        planes = tuple(range(params["x_bits"]))
    return [Evidence(
        detector="recovery",
        score=float(score),
        tier_hint=tier,
        explanation=expl,
        confidence=0.95,
        location=Location(tensor=names[0], start=start, end=end, bit_planes=planes),
        features=detail,
    )]


def _hexdump(b: bytes, width: int = 16) -> str:
    lines = []
    for off in range(0, len(b), width):
        chunk = b[off:off + width]
        hexpart = " ".join(f"{x:02x}" for x in chunk)
        asc = "".join(chr(x) if 32 <= x < 127 else "." for x in chunk)
        lines.append(f"{off:04x}  {hexpart:<{width*3}}  {asc}")
    return "\n".join(lines)
