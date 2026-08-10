"""Serialization / container detector (attack family A).

Reads container_notes gathered by the loader during a NO-EXECUTION opcode scan.
A dangerous global reachable via REDUCE is an active execution vector -> tier E4.
This is what makes ModelSentry see the *combined* attack: a tiny pickle stub that
extracts an LSB payload at load time. The weight detectors flag the payload; this
detector flags the extractor.
"""
from __future__ import annotations
from ..core.evidence import Evidence, Location


def detect(graph) -> list[Evidence]:
    out = []
    for note in graph.container_notes:
        sev = note.get("severity")
        if sev == "critical":
            out.append(Evidence(
                detector="serialization",
                score=12.0,          # dominant, this is an execution vector
                tier_hint="E4",
                explanation=f"container executes code on load: {note['detail']} "
                            f"(source {note.get('source')})",
                confidence=1.0,
                location=Location(tensor=note.get("source", "container")),
                features={"pos": note.get("pos", -1)},
            ))
    return out
