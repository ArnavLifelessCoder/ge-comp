"""Per-neuron distribution detector (attack family B3, EvilModel).

EvilModel replaces whole neurons (rows of a linear weight matrix) with crafted
floats that encode payload bytes. Even if the payload is encrypted and NOT
recoverable by carving, the replaced neurons are statistical outliers: their
value distribution does not match the rest of the layer, and the paper's
control-byte trick clamps them into a narrow magnitude band.

Primary signal (anomaly): per-neuron distance from the layer's aggregate
distribution. We use a robust per-row z of (std, kurtosis, range) plus the count
of neurons whose values are suspiciously quantized/banded.

Secondary signal (SIGNATURE, reported separately, never carries the verdict
alone): fraction of neurons whose values fall almost entirely in the EvilModel
0.0078-0.0313 magnitude band with the tell-tale 0x3c/0xbc high byte.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from ..core.evidence import Evidence, Location


def _robust_z(vals: np.ndarray) -> np.ndarray:
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    if mad < 1e-12:
        mad = vals.std() if vals.std() > 1e-12 else 1.0
    return (vals - med) / mad


def analyze_tensor(arr: np.ndarray) -> dict | None:
    if arr.ndim != 2 or arr.shape[0] < 8:
        return None
    rows = arr.astype(np.float64)
    out_dim = rows.shape[0]
    # per-neuron summary stats
    stds = rows.std(axis=1)
    # kurtosis per row (crafted byte-floats are platykurtic / oddly shaped)
    kurt = stats.kurtosis(rows, axis=1)
    ranges = rows.max(axis=1) - rows.min(axis=1)

    z_std = np.abs(_robust_z(stds))
    z_kurt = np.abs(_robust_z(kurt))
    z_rng = np.abs(_robust_z(ranges))
    per_neuron = np.maximum.reduce([z_std, z_kurt, z_rng])

    # how many neurons are strong multivariate outliers
    n_outlier = int((per_neuron > 6.0).sum())
    frac_outlier = n_outlier / out_dim

    # EvilModel control-byte signature: high byte 0x3c/0xbc across a row
    u = arr.view(np.uint32)
    hi = (u >> 24) & 0xFF
    banded_rows = ((hi == 0x3c) | (hi == 0xbc)).mean(axis=1)
    sig_neurons = int((banded_rows > 0.9).sum())

    return {"max_neuron_z": float(per_neuron.max()),
            "n_outlier_neurons": n_outlier,
            "frac_outlier": float(frac_outlier),
            "evilmodel_signature_neurons": sig_neurons,
            "worst_row": int(per_neuron.argmax())}


def detect(graph, baseline=None) -> list[Evidence]:
    out = []
    for t in graph.float_tensors():
        if t.dtype != "float32":
            continue
        arr = t.values()
        st = analyze_tensor(arr)
        if st is None:
            continue

        # anomaly score from outlier-neuron fraction (calibrated if baseline)
        frac = st["frac_outlier"]
        if baseline is not None:
            score = baseline.zscore("distribution", t.role, "frac_outlier", frac)
        else:
            score = frac / 0.02   # provisional: >2% outlier neurons is odd

        tier = "E1"
        expl = (f"{st['n_outlier_neurons']} neurons in {t.name} are strong "
                f"distribution outliers vs the layer (max z={st['max_neuron_z']:.1f})")
        feats = dict(st)

        # signature bump (separate, high precision)
        if st["evilmodel_signature_neurons"] >= 1:
            score = max(score, 6.0)
            tier = "E2"
            expl += (f"; {st['evilmodel_signature_neurons']} neurons match the "
                     f"EvilModel control-byte signature (0x3c/0xbc magnitude band)")
            feats["signature_match"] = True

        if score <= 0:
            continue
        out.append(Evidence(
            detector="distribution",
            score=float(score),
            tier_hint=tier,
            explanation=expl,
            confidence=0.85,
            location=Location(tensor=t.name, start=st["worst_row"]),
            features=feats,
        ))
    return out
