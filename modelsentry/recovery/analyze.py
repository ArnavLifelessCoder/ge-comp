"""Score a recovered byte stream for evidence of a hidden payload.

Returns (score, tier, detail). The score reflects how structured the stream is
relative to what natural mantissa noise would produce (which is ~uniform random).

Two independent layers, deliberately separated:

  1. MARKER-AGNOSTIC STRUCTURE (the primary signal). A windowed non-uniformity
     scan: split the stream into windows, compute a chi-square-against-uniform
     statistic per window, and ask whether the best window stands out from the
     distribution of all the other windows in the SAME stream. Natural mantissa
     bits are ~uniform, so the stream calibrates its own null -- no baseline, no
     format list, and it works on payload formats we have never seen. It also
     localizes the payload to a byte offset, which the report needs.
     A compressibility check on the winning window confirms it.

  2. FORMAT MARKERS (corroboration only). Magic headers and known strings. These
     raise the evidence TIER to E3 ("we know what it is") but the structure scan
     alone is enough to fire at E2. This ordering matters: a detector whose only
     signal is a list of magics can only find payloads whose format someone
     already thought of, and it scores perfectly against a red team that uses
     that same list. See bench/ for the held-out-marker track that tests this.

The residual, stated plainly: an encrypted payload is uniform-random and has no
markers, so neither layer fires. That is a real limit of static analysis, not an
oversight -- it is measured and reported as its own benchmark track.
"""
from __future__ import annotations

import math
import re
import zlib

import numpy as np

MAGICS = {
    b"MZ": "DOS/PE executable header",
    b"\x7fELF": "ELF executable header",
    b"\xca\xfe\xba\xbe": "Mach-O / Java class",
    b"\xfe\xed\xfa": "Mach-O",
    b"PK\x03\x04": "ZIP/JAR archive",
    b"\x1f\x8b": "gzip stream",
    b"%PDF": "PDF document",
    b"\x89PNG": "PNG image",
    b"#!/": "script shebang",
}
STRING_HINTS = [b"cannot be run in DOS mode", b"kernel32", b"cmd.exe", b"/bin/sh",
                b"powershell", b"socket", b"subprocess", b"http://", b"https://"]

# Structure-scan thresholds. The stream supplies its own null (all other
# windows), but the winning window is a MAXIMUM over many windows and the
# detector tries many candidate reconstructions, so the bar is set well above a
# nominal significance level -- see recovery_detector for the candidate-level
# correction that sits on top of this.
WINDOW = 1024
MIN_WINDOWS = 8          # below this the self-calibrated null is meaningless
STRUCT_Z = 8.0           # robust z of the best window vs the rest of the stream
COMPRESS_DELTA = 0.03    # winning window must also be measurably compressible


def _byte_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, np.uint8), minlength=256)
    p = counts[counts > 0] / len(b)
    return float(-(p * np.log2(p)).sum())  # 0..8


def _printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    printable = sum(1 for x in b if 32 <= x < 127 or x in (9, 10, 13))
    return printable / len(b)


def _longest_printable_run(b: bytes) -> int:
    best = cur = 0
    for x in b:
        if 32 <= x < 127:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def _compress_ratio(b: bytes) -> float:
    if not b:
        return 1.0
    return len(zlib.compress(b, 6)) / len(b)


def structure_scan(stream: bytes, window: int = WINDOW) -> dict | None:
    """Windowed non-uniformity scan. Format-agnostic; no external baseline.

    Returns the best window's robust z-score against the other windows of the
    same stream, its byte offset, and its compressibility. None if the stream is
    too short for the self-calibrated null to mean anything.
    """
    n_win = len(stream) // window
    if n_win < MIN_WINDOWS:
        return None
    buf = np.frombuffer(stream[:n_win * window], np.uint8).reshape(n_win, window)
    # chi-square against uniform, per window. Flattened bincount (row*256+byte)
    # rather than np.add.at: this runs on every candidate reconstruction, so it
    # is the hot path for scan latency, and a CI gate is judged on wall clock.
    flat_idx = (np.arange(n_win, dtype=np.int64)[:, None] * 256 + buf).reshape(-1)
    counts = np.bincount(flat_idx, minlength=n_win * 256).reshape(n_win, 256)
    expected = window / 256.0
    chi2 = ((counts - expected) ** 2 / expected).sum(axis=1)

    # robust null from the stream itself: median / MAD over all windows. Natural
    # mantissa noise is uniform, so a payload window is a large positive outlier.
    med = float(np.median(chi2))
    mad = float(np.median(np.abs(chi2 - med)))
    sigma = mad * 1.4826 if mad > 1e-9 else float(chi2.std()) or 1.0
    z = (chi2 - med) / sigma
    best = int(np.argmax(z))
    best_z = float(z[best])
    if not np.isfinite(best_z):
        return None

    # The window-vs-window null has one blind spot: an attacker who fills the
    # ENTIRE capacity with structured data makes every window look alike, so the
    # median rises with the max and the relative z collapses. So also test the
    # median window against the ABSOLUTE uniform null (chi2 with 255 df: mean
    # 255, sd sqrt(510)), which natural mantissa bits sit right on top of --
    # measured at z=0.4 across every candidate of the clean corpus.
    global_z = (med - 255.0) / math.sqrt(510.0)

    win_bytes = stream[best * window:(best + 1) * window]
    # control = the median-chi2 window, i.e. this stream's own natural noise
    ctrl_idx = int(np.argsort(chi2)[n_win // 2])
    ctrl_bytes = stream[ctrl_idx * window:(ctrl_idx + 1) * window]
    cr = _compress_ratio(win_bytes)
    ctrl_cr = _compress_ratio(ctrl_bytes)
    return {"struct_z": round(best_z, 2),
            "global_z": round(global_z, 2),
            "global_compress_ratio": round(_compress_ratio(stream[:16384]), 4),
            "offset": best * window,
            "n_windows": n_win,
            "compress_ratio": round(cr, 4),
            "control_compress_ratio": round(ctrl_cr, 4),
            "compress_delta": round(ctrl_cr - cr, 4),
            "window_entropy": round(_byte_entropy(win_bytes), 3)}


def z_floor_for(n_candidates: int) -> float:
    """Multiple-comparison correction.

    The detector takes a MAXIMUM over many candidate reconstructions, each of
    which is itself a maximum over windows. Holding the per-window bar fixed
    while the hypothesis count grows is how scanners manufacture false positives
    on large models (tensor count, and therefore candidate count, scales with
    model size). So the bar rises with the number of hypotheses tested.
    """
    return STRUCT_Z + 0.7 * max(0.0, math.log10(max(n_candidates, 1)))


def score_stream(stream: bytes, params: dict,
                 z_floor: float = STRUCT_Z) -> tuple[float, str, dict]:
    if len(stream) < 8:
        return 0.0, "E1", {}
    detail = {"x_bits": params.get("x_bits"), "bit_order": params.get("bit_order")}
    score = 0.0
    tier = "E1"

    def raise_tier(t):
        nonlocal tier
        from ..core.evidence import TIER_ORDER
        if TIER_ORDER[t] > TIER_ORDER[tier]:
            tier = t

    # ---- layer 1: marker-agnostic windowed structure (primary) ----------------
    st = structure_scan(stream)
    if st is not None:
        detail.update(st)
        detail["struct_z_floor"] = round(z_floor, 2)
        localized = (st["struct_z"] >= z_floor and st["compress_delta"] >= COMPRESS_DELTA)
        # capacity-saturating payload: no window stands out because they are all
        # payload, so the relative test is blind and only the absolute one fires.
        # The absolute test is only meaningful where uniformity is the true null,
        # i.e. reconstructed mantissa bits. RAW float bytes are strongly
        # non-uniform by nature (exponents cluster), so applying it there flags
        # every clean model -- measured, not hypothetical.
        uniform_null_holds = params.get("x_bits") != "raw"
        saturated = (uniform_null_holds
                     and st["global_z"] >= z_floor
                     and st["global_compress_ratio"] <= 1.0 - COMPRESS_DELTA)
        if localized or saturated:
            # scale with how far past the bar it is, capped so that a single
            # very lopsided window cannot by itself dominate the fusion.
            z = st["struct_z"] if localized else st["global_z"]
            score += min(6.0, 2.0 + (z - z_floor) / 6.0)
            raise_tier("E2")
            detail["structured_window"] = localized
            detail["saturated_stream"] = saturated

    # corroboration signals for the marker layer
    known_strings = [h.decode("latin1") for h in STRING_HINTS if h in stream]
    printable_body = _printable_ratio(stream[4:2048]) > 0.6

    # ---- layer 2: format markers (corroboration / tier only) -----------------
    # A magic's strength scales with its specificity: >=4-byte magics are safe
    # alone (chance ~2^-32); 2-3 byte magics (MZ, gzip, Mach-O) collide by chance
    # in max-entropy mantissa noise, so they only reach E3 WITH corroboration.
    head = stream[:64]
    magic_hit = None
    for m, desc in MAGICS.items():
        idx = head.find(m)
        if idx != -1 and idx <= 8:
            magic_hit = (m, desc, idx, len(m)); break
    if magic_hit:
        m, desc, idx, mlen = magic_hit
        corroborated = bool(known_strings) or printable_body or detail.get("structured_window")
        if mlen >= 4:
            score += 8.0; raise_tier("E3")
        elif corroborated:
            score += 6.0; raise_tier("E3")
        else:
            score += 0.4   # weak/uncorroborated short magic: consistent with noise
            detail["magic_weak"] = True
        detail["magic"] = desc; detail["magic_offset"] = idx; detail["magic_len"] = mlen

    if known_strings:
        score += 4.0; raise_tier("E3")
        detail["strings"] = known_strings[:5]

    # ---- generic framing / text signals --------------------------------------
    # plausible 32-bit big-endian length frame followed by that many bytes
    L = int.from_bytes(stream[:4], "big")
    if 16 <= L <= len(stream) - 4:
        body = stream[4:4 + L]
        tail = stream[4 + L:4 + 2 * L] if len(stream) >= 4 + 2 * L else b""
        # payload body should differ in structure from the natural-noise tail
        if body and tail:
            be = _byte_entropy(body); te = _byte_entropy(tail)
            if be < te - 0.15 or _printable_ratio(body) > 0.6:
                score += 3.0; raise_tier("E2")
                detail["length_frame"] = L
                detail["body_entropy"] = round(be, 3)
                detail["tail_entropy"] = round(te, 3)

    # long printable run / high printable ratio (scripts, base64, PEM, config)
    pr = _printable_ratio(stream[:4096])
    run = _longest_printable_run(stream[:4096])
    if run >= 24 or pr > 0.85:
        score += min(4.0, run / 20.0)
        raise_tier("E2")
        detail["printable_ratio"] = round(pr, 3)
        detail["longest_printable_run"] = run

    # base64 signature
    if re.search(rb"[A-Za-z0-9+/]{40,}={0,2}", stream[:4096]):
        score += 2.0
        raise_tier("E2")
        detail["base64_blob"] = True

    return score, tier, detail
