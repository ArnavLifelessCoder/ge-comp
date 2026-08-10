"""Empirical clean-corpus baseline.

Every detector statistic is converted to a z-score against the distribution of
that same statistic on CLEAN models of the same layer role. This is what makes
thresholds principled instead of hand-tuned, and what lets quantized/pruned
models pass: if a statistic is normal for that role in clean models, it scores ~0.
"""
from __future__ import annotations

import json
import numpy as np
from collections import defaultdict


class Baseline:
    def __init__(self):
        # stats[(detector, role, feature)] = list of clean values
        self._raw = defaultdict(list)
        self._fit = {}

    def observe(self, detector: str, role: str, feature: str, value: float):
        if value is None or not np.isfinite(value):
            return
        self._raw[(detector, role, feature)].append(float(value))

    MIN_N = 5   # below this, a calibration is too thin to trust -> not scored

    def fit(self):
        self._fit = {}
        for key, vals in self._raw.items():
            a = np.asarray(vals, dtype=np.float64)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med))) * 1.4826
            std = float(a.std())
            scale = mad if mad > 1e-9 else (std if std > 1e-9 else 1.0)
            # Relative floor: with a small corpus MAD can collapse to near-zero and
            # inflate z-scores on benign architecture drift. Floor the scale at a
            # fraction of the center magnitude so drift stays proportionate.
            rel_floor = 0.10 * abs(med)
            scale = max(scale, rel_floor, 1e-6)
            self._fit[key] = {"center": med, "scale": scale, "n": len(vals)}
        return self

    def zscore(self, detector: str, role: str, feature: str, value: float) -> float:
        key = (detector, role, feature)
        if key not in self._fit:
            # unknown role -> back off to any-role aggregate if present
            key = (detector, "*", feature)
            if key not in self._fit:
                return 0.0
        f = self._fit[key]
        if f.get("n", 0) < self.MIN_N:
            return 0.0
        return (float(value) - f["center"]) / f["scale"]

    def save(self, path: str):
        out = {f"{d}|{r}|{ft}": v for (d, r, ft), v in self._fit.items()}
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Baseline":
        b = cls()
        with open(path) as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            d, r, ft = k.split("|")
            b._fit[(d, r, ft)] = v
        return b
