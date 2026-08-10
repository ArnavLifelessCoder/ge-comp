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
    raise ValueError(f"unknown payload kind {kind}")
