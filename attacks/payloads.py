"""Payload generators for the red team. Defensive project: these produce BENIGN
stand-ins that carry the same statistical + structural signatures a real payload
would (magic headers, high entropy, byte structure) WITHOUT being actual malware.
A PE stub here is a harmless DOS-header byte string, not a working executable.
"""
from __future__ import annotations
import os
import zlib
import base64

# A minimal, inert PE/DOS header. Carries the MZ magic + the classic stub string
# so detectors can prove recovery, but it is not a runnable program.
PE_STUB = (
    b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
    b"\xb8\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00"
    + b"\x00" * 32
    + b"\x0e\x1f\xba\x0e\x00\xb4\x09\xcd\x21\xb8\x01\x4c\xcd\x21"
    b"This program cannot be run in DOS mode.\r\r\n$"
)

ELF_STUB = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\x3e\x00\x01\x00\x00\x00"

# --- held-out payloads ------------------------------------------------------
# NONE of the markers below appear in the detector's MAGICS or STRING_HINTS
# lists. They exist to answer the fair question about the "seen" payloads above:
# is the scanner detecting hidden data, or is it just grepping for the same
# constants the red team used? A detector that only passes the seen track has
# learned the answer key. These carry real format structure (headers, string
# tables, repeated records, text) that the marker-agnostic structure scan should
# find without ever having been told what the format is.
HOLDOUT_KINDS = ["wasm", "sqlite", "pem", "yaml_config", "packed"]

# encrypted/*: uniform-random, no markers, no compressible structure. The honest
# residual for ANY static scanner. Reported as its own track, never folded into
# a headline number.
ENCRYPTED_KINDS = ["random", "encrypted_pe"]


def _xor_keystream(data: bytes, seed: int = 7) -> bytes:
    """Stand-in for an encrypted payload: a keystream XOR removes every marker
    and flattens the byte histogram, exactly like AES-CTR would. The dropper
    stub holds the key; the file itself looks like noise."""
    import random as _r
    rnd = _r.Random(seed)
    ks = bytes(rnd.getrandbits(8) for _ in range(len(data)))
    return bytes(a ^ b for a, b in zip(data, ks))


def make_payload(kind: str, size: int = 4096, seed: int = 0) -> bytes:
    rng = os.urandom  # cryptographic randomness -> near-uniform, like encrypted malware
    if kind == "random":
        return rng(size)
    if kind == "pe":
        body = PE_STUB + rng(max(0, size - len(PE_STUB)))
        return body[:size] if size else PE_STUB
    if kind == "elf":
        body = ELF_STUB + rng(max(0, size - len(ELF_STUB)))
        return body[:size] if size else ELF_STUB
    if kind == "script":
        # a base64-encoded 'script' blob: printable, structured, low entropy
        raw = (b"import os,socket,subprocess\n# beacon stub (inert demo)\n"
               b"HOST='192.0.2.10';PORT=4444\n") * (max(1, size // 80))
        return base64.b64encode(raw)[:size] if size else base64.b64encode(raw)
    if kind == "zip":
        inner = rng(size)
        return b"PK\x03\x04" + zlib.compress(inner)

    # ---- held-out formats: markers deliberately absent from the detector -----
    if kind == "wasm":
        # WebAssembly module: 8-byte header, then type/import/code sections with
        # heavily repeated LEB128 opcodes -> structured but not in MAGICS.
        body = b"\x00asm\x01\x00\x00\x00"
        body += b"\x01\x07\x01\x60\x02\x7f\x7f\x01\x7f"        # type section
        body += b"\x03\x02\x01\x00\x07\x0a\x01\x06memory\x02\x00"
        code = b"\x20\x00\x20\x01\x6a\x0b"                      # get,get,add,end
        body += b"\x0a" + bytes([len(code) + 2, 1, len(code)]) + code
        return (body + code * 400)[:size] if size else body
    if kind == "sqlite":
        # SQLite page structure: fixed header + page-aligned records with long
        # runs of zero padding, which is exactly the non-uniformity a
        # format-agnostic scan should see.
        hdr = b"SQLite format 3\x00" + b"\x10\x00\x01\x01\x00\x40\x20\x20"
        page = (b"\x0d\x00\x00\x00\x03\x0f\x00\x00" + b"\x00" * 120
                + b"\x03\x17\x1b\x01\x81tableusersusers" + b"\x00" * 80)
        return (hdr + page * 64)[:size] if size else hdr + page
    if kind == "pem":
        # base64 text body, no shebang, no known strings
        b64 = base64.b64encode(rng(max(64, size))).decode()
        lines = "\n".join(b64[i:i + 64] for i in range(0, len(b64), 64))
        out = ("-----BEGIN CERTIFICATE-----\n" + lines +
               "\n-----END CERTIFICATE-----\n").encode()
        return out[:size] if size else out
    if kind == "yaml_config":
        # a dropper's config file: printable, repetitive, no listed strings
        rec = ("- id: {i}\n  endpoint: 198.51.100.{o}\n  port: 8443\n"
               "  interval: 300\n  retries: 3\n")
        out = "".join(rec.format(i=i, o=i % 254 + 1) for i in range(400)).encode()
        return out[:size] if size else out
    if kind == "packed":
        # UPX-style packed blob: a small stub, a string table, then compressed
        # data. No recognizable magic, but the record structure is real.
        stub = bytes(range(0x40)) * 4
        strtab = b"\x00".join(b"sym_%04d" % i for i in range(80)) + b"\x00"
        return (stub + strtab + zlib.compress(rng(max(256, size))))[:size]

    # ---- encrypted / unmarked: the honest residual --------------------------
    if kind == "encrypted_pe":
        base = make_payload("pe", size=size or 4096, seed=seed)
        return _xor_keystream(base, seed=seed or 7)

    raise ValueError(f"unknown payload kind {kind}")
