"""Spectral / residual detector (attack family B4, MaleficNet spread-spectrum).

Spread-spectrum injection adds low-amplitude broadband noise across many weights.
No single weight is abnormal and (without the PN sequence) the payload cannot be
recovered, so we look at what the added noise does to the weight statistics,
calibrated against a clean baseline for the same layer role:

  hf_ratio : var(first-difference) / var(weights). Added broadband noise raises
             high-frequency energy relative to the smooth trained component.
  kurtosis : trained weights are typically heavy-tailed; adding near-Gaussian
             noise pulls excess kurtosis toward 0, so a DROP is suspicious.

This is the weakest detector by design (the literature tops out around 80% on
MaleficNet). It is reported as such and never the sole basis for a red verdict
unless the deviation is extreme.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from ..core.evidence import Evidence, Location


def analyze_tensor(arr: np.ndarray) -> dict | None:
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 1024:
        return None
    v = flat.var()
    if v < 1e-16:
        return None
    d = np.diff(flat)
    hf_ratio = float(d.var() / (2.0 * v))     # ~1 for white noise, <1 for smooth
    kurt = float(stats.kurtosis(flat))
    return {"hf_ratio": hf_ratio, "kurtosis": kurt}


def detect(graph, baseline=None) -> list[Evidence]:
    out = []
    for t in graph.float_tensors():
        if t.dtype != "float32":
            continue
        st = analyze_tensor(t.values())
        if st is None:
            continue
        if baseline is not None:
            z_hf = baseline.zscore("spectral", t.role, "hf_ratio", st["hf_ratio"])
            z_ku = -baseline.zscore("spectral", t.role, "kurtosis", st["kurtosis"])
            score = max(z_hf, z_ku)
        else:
            score = 0.0
        if score <= 2.5:      # conservative floor: this detector is noisy
            continue
        out.append(Evidence(
            detector="spectral",
            score=float(score),
            tier_hint="E1",
            explanation=(f"{t.name} shows broadband energy inconsistent with the "
                         f"clean baseline (hf_ratio={st['hf_ratio']:.3f}, "
                         f"kurtosis={st['kurtosis']:.2f}); possible spread-spectrum payload"),
            confidence=0.5,
            location=Location(tensor=t.name),
            features=st,
        ))
    return out
