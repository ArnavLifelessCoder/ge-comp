"""Score a recovered byte stream for evidence of a hidden payload.

Returns (score, tier, detail). The score reflects how structured the stream is
relative to what natural mantissa noise would produce (which is ~uniform random).
Structured content -> high score. Uniform noise -> ~0.
"""
from __future__ import annotations

import math
import re
import zlib

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


def _byte_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = [0] * 256
    for x in b:
        counts[x] += 1
    n = len(b)
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h  # 0..8


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


def score_stream(stream: bytes, params: dict) -> tuple[float, str, dict]:
    if len(stream) < 8:
        return 0.0, "E1", {}
    detail = {"x_bits": params.get("x_bits"), "bit_order": params.get("bit_order")}
    score = 0.0
    tier = "E1"

    # corroboration signals computed up front (used to gate weak magics)
    known_strings = [h.decode("latin1") for h in STRING_HINTS if h in stream]
    printable_body = _printable_ratio(stream[4:2048]) > 0.6

    # 1) magic header near the start (allow small framing offset for a length hdr).
    #    A magic's strength scales with its specificity: >=4-byte magics are safe
    #    alone (chance ~2^-32); 2-3 byte magics (MZ, gzip, Mach-O) collide by
    #    chance in max-entropy mantissa noise, so they only reach E3 WITH
    #    corroboration (a known string or a printable body). This kills the
    #    clean-model false positives.
    head = stream[:64]
    magic_hit = None
    for m, desc in MAGICS.items():
        idx = head.find(m)
        if idx != -1 and idx <= 8:
            magic_hit = (m, desc, idx, len(m)); break
    if magic_hit:
        m, desc, idx, mlen = magic_hit
        corroborated = bool(known_strings) or printable_body
        if mlen >= 4:
            score += 8.0; tier = "E3"
        elif corroborated:
            score += 6.0; tier = "E3"
        else:
            score += 0.4; tier = "E1"   # weak/uncorroborated short magic: noise
            detail["magic_weak"] = True
        detail["magic"] = desc; detail["magic_offset"] = idx; detail["magic_len"] = mlen

    # 2) known payload strings anywhere (strong, specific)
    if known_strings:
        score += 4.0; tier = "E3"
        detail["strings"] = known_strings[:5]

    # 3) plausible 32-bit big-endian length frame followed by that many bytes
    L = int.from_bytes(stream[:4], "big")
    if 16 <= L <= len(stream) - 4:
        body = stream[4:4 + L]
        tail = stream[4 + L:4 + 2 * L] if len(stream) >= 4 + 2 * L else b""
        # payload body should differ in structure from the natural-noise tail
        if body and tail:
            be = _byte_entropy(body); te = _byte_entropy(tail)
            if be < te - 0.15 or _printable_ratio(body) > 0.6:
                score += 3.0; tier = max(tier, "E2", key=lambda t: t)
                detail["length_frame"] = L
                detail["body_entropy"] = round(be, 3)
                detail["tail_entropy"] = round(te, 3)

    # 4) long printable run / high printable ratio (scripts, base64)
    pr = _printable_ratio(stream[:4096])
    run = _longest_printable_run(stream[:4096])
    if run >= 24 or pr > 0.85:
        score += min(4.0, run / 20.0)
        if tier == "E1":
            tier = "E2"
        detail["printable_ratio"] = round(pr, 3)
        detail["longest_printable_run"] = run

    # 5) base64 signature
    if re.search(rb"[A-Za-z0-9+/]{40,}={0,2}", stream[:4096]):
        score += 2.0
        if tier == "E1":
            tier = "E2"
        detail["base64_blob"] = True

    return score, tier, detail
