"""Top-level scan: load -> run detectors -> collect evidence -> fuse -> score.

Fusion is the ONLY place a verdict is formed. Detectors just supply evidence.
The output is a risk score (0-100) AND an evidence tier, kept separate on
purpose: statistical anomaly (E1) can never reach the red band on its own; that
requires recovered structure/signature (E2/E3) or an execution vector (E4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .loaders import load_model
from .evidence import Evidence, TIER_ORDER
from ..detectors import bitplane, serialization, recovery_detector, distribution, spectral


@dataclass
class ScanResult:
    path: str
    risk: int
    tier: str
    band: str
    evidence: list = field(default_factory=list)
    top: list = field(default_factory=list)
    summary: str = ""

    def to_dict(self):
        return {"path": self.path, "risk": self.risk, "tier": self.tier,
                "band": self.band, "summary": self.summary,
                "evidence": [e.to_dict() for e in self.top]}


def _band(risk: int) -> str:
    if risk <= 20:
        return "clean"
    if risk <= 50:
        return "review"
    if risk <= 80:
        return "suspicious"
    return "likely-compromised"


def _logistic(x, k=0.9, x0=3.0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fuse(evidence: list[Evidence]) -> tuple[int, str]:
    """Map evidence to (risk 0-100, tier)."""
    if not evidence:
        return 0, "none"
    # Only these detectors drive the verdict. bitplane is proven unreliable on
    # fully-trained float32 (whole mantissa is already max-entropy), so it is kept
    # as informational context in the report but never moves the risk score.
    SCORING = {"recovery", "serialization", "distribution"}
    scoring_ev = [e for e in evidence if e.detector in SCORING]
    if not scoring_ev:
        return 0, "none"
    stat_scores = [e.score * e.confidence for e in scoring_ev]
    smax = max(stat_scores) if stat_scores else 0.0

    # tier: highest tier among scoring evidence that actually fired (score > 1)
    fired = [e for e in scoring_ev if e.score > 1.0]
    tier = "none"
    if fired:
        tier = max((e.tier_hint for e in fired), key=lambda t: TIER_ORDER[t])

    base = _logistic(smax) * 100.0
    # tier gating: E1 capped at 'suspicious' ceiling; higher tiers lift the floor
    if tier in ("none", "E1"):
        # statistical-only evidence can flag for REVIEW but never reach the
        # suspicious/red bands on its own; that requires recovered structure
        # (E2/E3) or an execution vector (E4).
        risk = min(base, 45.0)
    elif tier == "E2":
        risk = max(base, 60.0)
    elif tier == "E3":
        risk = max(base, 85.0)
    else:  # E4
        risk = max(base, 92.0)
    return int(round(risk)), tier


def scan(path: str, baseline=None) -> ScanResult:
    g = load_model(path)
    evidence: list[Evidence] = []

    # weight-value detectors
    for t in g.float_tensors():
        ev = bitplane.detect(t, baseline)
        if ev is not None:
            evidence.append(ev)

    # recovery detector (primary weight-stego signal)
    evidence.extend(recovery_detector.detect(g))

    # per-neuron distribution detector (EvilModel / B3)
    evidence.extend(distribution.detect(g, baseline))

    # spectral / spread-spectrum detector (MaleficNet / B4)
    evidence.extend(spectral.detect(g, baseline))

    # container/serialization detector
    evidence.extend(serialization.detect(g))

    risk, tier = fuse(evidence)
    # Findings shown to the user come only from verdict-driving detectors.
    # bitplane is informational (unreliable on trained f32) and would otherwise
    # print alarming text on clean models, so it is excluded from the headline.
    SCORING = {"recovery", "serialization", "distribution"}
    top = sorted([e for e in evidence if e.detector in SCORING and e.score > 0],
                 key=lambda e: e.score, reverse=True)[:8]
    result = ScanResult(path=path, risk=risk, tier=tier, band=_band(risk),
                        evidence=evidence, top=top)
    if risk <= 20:
        result.summary = "No significant indicators of hidden payloads."
    else:
        worst = top[0] if top else None
        loc = worst.location.describe() if worst and worst.location else "multiple tensors"
        result.summary = (f"Elevated supply-chain risk ({result.band}, tier {tier}). "
                          f"Strongest indicator: {loc}.")
    return result
