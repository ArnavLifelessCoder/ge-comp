"""The detector contract.

Detectors return EVIDENCE, never verdicts. Fusion happens once, centrally, so
that every alert is auditable back to the raw number that produced it and the
explanation is honest. This is the single most important design decision in the
codebase, per the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

# Evidence tiers separate "the statistics look odd" from "I pulled an executable
# out of the mantissa bits". They must never collapse into one number.
#   E1  statistical anomaly only
#   E2  structured data recovered (printable / low-compressibility stream)
#   E3  payload signature recovered (magic header, YARA hit)
#   E4  active execution vector present (serialization can extract/run)
Tier = Literal["E1", "E2", "E3", "E4"]

TIER_ORDER = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}


@dataclass
class Location:
    """Where inside a tensor the evidence points, for localization + report."""
    tensor: str
    start: Optional[int] = None      # flat index into the tensor
    end: Optional[int] = None
    bit_planes: Optional[tuple[int, ...]] = None

    def describe(self) -> str:
        parts = [self.tensor]
        if self.start is not None:
            parts.append(f"[{self.start}:{self.end}]")
        if self.bit_planes:
            parts.append("bits " + ",".join(map(str, self.bit_planes)))
        return " ".join(parts)


@dataclass
class Evidence:
    detector: str
    score: float                       # calibrated: higher = more suspicious (z-like)
    tier_hint: Tier
    explanation: str
    confidence: float = 1.0            # 0..1, detector's self-assessed reliability
    location: Optional[Location] = None
    features: dict = field(default_factory=dict)   # everything, for audit

    def to_dict(self) -> dict:
        d = {
            "detector": self.detector,
            "score": round(float(self.score), 4),
            "tier_hint": self.tier_hint,
            "confidence": round(float(self.confidence), 3),
            "explanation": self.explanation,
            "features": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                         for k, v in self.features.items()},
        }
        if self.location:
            d["location"] = self.location.describe()
        return d
