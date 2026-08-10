"""Bit-plane entropy detector (attack family B1, X-LSB).

Core idea, stated as a BASELINE-RELATIVE deviation, never as an assumed clean
shape. For float32 we look at the 23 mantissa bits. For each bit plane we
measure the Shannon entropy of that bit across all weights in a tensor. In a
clean trained tensor the low mantissa bits are already near-random but the FULL
curve, and especially its coupling to the exponent, has structure. X-LSB
injection forces the low X planes to ~1.0 bit of entropy AND decouples them from
the exponent.

We emit two features:
  lsb_flatness  : how flat/saturated the low planes are vs the mid planes
  exp_coupling  : how much low-bit entropy depends on the exponent bucket
                  (injection destroys this dependence -> value near 0)

Scores are turned into calibrated z-scores against a clean baseline elsewhere;
here we return the raw statistics + a provisional score.
"""
from __future__ import annotations

import numpy as np
from ..core.evidence import Evidence, Location
from ..core.tensorgraph import Tensor

MANTISSA_BITS = 23
EXP_SHIFT = 23
EXP_MASK = 0xFF


def _bit_entropy(bits: np.ndarray) -> float:
    """Shannon entropy (bits) of a 0/1 array."""
    if bits.size == 0:
        return 0.0
    p1 = bits.mean()
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    p0 = 1.0 - p1
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def plane_entropies(u: np.ndarray, n_bits: int = MANTISSA_BITS) -> np.ndarray:
    """Entropy per mantissa bit plane, index 0 = LSB."""
    out = np.empty(n_bits, dtype=np.float64)
    for b in range(n_bits):
        out[b] = _bit_entropy(((u >> b) & 1).astype(np.uint8))
    return out


def exponent_coupling(u: np.ndarray, low_planes: int = 6) -> float:
    """Measure how the low-plane entropy varies across exponent buckets.
    Clean weights: low-bit behaviour differs by magnitude -> variance > 0.
    Injected weights: uniform payload regardless of magnitude -> variance ~ 0.
    Returned as a normalized dispersion; LOW means suspicious."""
    exp = (u >> EXP_SHIFT) & EXP_MASK
    # bucket exponents into deciles of the observed range
    uniq = np.unique(exp)
    if uniq.size < 3:
        return 1.0  # not enough magnitude diversity to judge; treat as benign
    qs = np.quantile(exp, np.linspace(0, 1, 6))
    buckets = np.clip(np.digitize(exp, qs[1:-1]), 0, 4)
    ent_by_bucket = []
    for bk in range(5):
        sel = u[buckets == bk]
        if sel.size < 64:
            continue
        e = plane_entropies(sel, low_planes)
        ent_by_bucket.append(e)
    if len(ent_by_bucket) < 2:
        return 1.0
    ent_by_bucket = np.array(ent_by_bucket)
    # dispersion of per-plane entropy across buckets, averaged over planes
    disp = ent_by_bucket.std(axis=0).mean()
    return float(disp)


def analyze_tensor(t: Tensor) -> dict:
    """Return raw statistics for one float32 tensor. Non-f32 returns None."""
    if t.dtype != "float32":
        return None
    u = t.uint_view()
    if u.size < 256:
        return None
    planes = plane_entropies(u)
    # low planes = 0..5, mid planes = 8..15 (reference band inside the mantissa)
    low = planes[0:6].mean()
    mid = planes[8:16].mean()
    # saturation: fraction of low planes essentially maxed out
    sat = float((planes[0:8] > 0.999).mean())
    coupling = exponent_coupling(u)
    return {"planes": planes, "low_mean": float(low), "mid_mean": float(mid),
            "lsb_saturation": sat, "exp_coupling": coupling}


def detect(t: Tensor, baseline=None) -> Evidence | None:
    stats = analyze_tensor(t)
    if stats is None:
        return None
    sat = stats["lsb_saturation"]
    coupling = stats["exp_coupling"]

    if baseline is not None:
        z_sat = baseline.zscore("bitplane", t.role, "lsb_saturation", sat)
        z_cpl = -baseline.zscore("bitplane", t.role, "exp_coupling", coupling)
        score = max(z_sat, z_cpl)
    else:
        # provisional, uncalibrated: saturation above 0.5 is inherently odd
        score = (sat - 0.15) / 0.1

    n_saturated = int((stats["planes"][0:12] > 0.999).sum())
    expl = (f"low mantissa bit-planes show saturation={sat:.2f} "
            f"({n_saturated} planes near-maximal entropy) and exponent-coupling="
            f"{coupling:.3f}; consistent with X-LSB payload of ~{n_saturated} bits/weight"
            if score > 0 else
            f"bit-plane statistics within normal range (saturation={sat:.2f})")
    return Evidence(
        detector="bitplane",
        score=float(score),
        tier_hint="E1",
        explanation=expl,
        confidence=0.9,
        location=Location(tensor=t.name,
                          bit_planes=tuple(range(max(1, n_saturated)))),
        features={"lsb_saturation": sat, "exp_coupling": coupling,
                  "low_mean": stats["low_mean"], "mid_mean": stats["mid_mean"],
                  "est_x_bits": n_saturated},
    )
