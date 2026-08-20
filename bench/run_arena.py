"""Arena: the capacity-stealth frontier, actually computed.

arena_dashboard.html makes a specific claim -- that no attack is simultaneously
stealthy (under the CI gate) and a usable weapon (self-contained, i.e. carrying
its own extractor). This script is what produces that claim, so the figure on
the dashboard is a measurement and not an assertion.

Each attack is a point in two dimensions:

  detection : the risk score ModelSentry assigns. Stealthy := risk < RED.
  usability : U_weapon  the artifact carries its own execution vector, so
                        loading the file is sufficient to run the payload.
              U_data    raw smuggled bytes. Needs an out-of-band loader that
                        the defender never sees, so the artifact alone does
                        nothing.

The mechanism under test is the trade-off, not a detector: to push detection
down the attacker has to strip structure from the payload (encrypt it), and to
keep it usable they have to add an extractor to the container -- where the
serialization detector reads it without unpickling. Both at once is the corner
we are checking is empty.

What this does NOT prove, and the dashboard says so: an attacker whose loader
lives in the *application* code rather than the model file is outside the
artifact we are handed. That is the disclosed residual, not a result.
"""
from __future__ import annotations

import os, sys, glob, json, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from safetensors.numpy import load_file

from modelsentry.core.scanner import scan
from attacks import lsb, evilmodel, maleficnet, payloads

RED = 51


class _ExtractorStub:
    """Inert stand-in for a dropper's extractor. It pickles to a reduce that
    would execute on load -- which is the property under test -- but the call it
    names is a harmless echo, and nothing here is ever unpickled: ModelSentry
    reads the opcodes."""

    def __reduce__(self):
        return (os.system, ("echo modelsentry-arena-inert-stub",))


def _as_self_contained(st_path: str, out_path: str) -> None:
    """Repackage a stego'd safetensors model as a pickle artifact that also
    carries the extractor. This is what 'usable weapon in one file' means."""
    tensors = {k: v for k, v in load_file(st_path).items()}
    with open(out_path, "wb") as f:
        pickle.dump({"state_dict": tensors, "_loader": _ExtractorStub()}, f)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    carriers = sorted(glob.glob(os.path.join(root, "clean", "*wide*.safetensors")))[:3]
    if not carriers:
        print("no carrier models; run build_models.py first"); return
    work = os.path.join(root, "arena"); os.makedirs(work, exist_ok=True)

    # attacker's search space: family x payload structure x embedding strength
    kinds = [("pe", "seen"), ("wasm", "holdout"), ("yaml_config", "holdout"),
             ("random", "encrypted"), ("encrypted_pe", "encrypted")]
    attacks = []
    for c in carriers:
        base = os.path.basename(c).replace(".safetensors", "")
        for pk, marker in kinds:
            for x in (1, 3, 6, 12):
                attacks.append(("B1", dict(x_bits=x), pk, marker, c, base))
            for frac in (0.1, 0.25):
                attacks.append(("B3", dict(frac_neurons=frac), pk, marker, c, base))
        for amp in (0.01, 0.03, 0.08):
            attacks.append(("B4", dict(amplitude=amp), "random", "encrypted", c, base))

    rows = []
    for i, (fam, kw, pk, marker, carrier, base) in enumerate(attacks):
        pl = payloads.make_payload(pk, size=2048, seed=2000 + i)
        stego = os.path.join(work, f"{fam}_{base}_{pk}_{i}.safetensors")
        if fam == "B1":
            lsb.inject(carrier, stego, pl, **kw)
        elif fam == "B3":
            evilmodel.inject(carrier, stego, pl, **kw)
        else:
            maleficnet.inject(carrier, stego, pl, **kw)

        # variant 1: raw smuggling. Needs an external loader -> U_data.
        r_data = scan(stego)
        rows.append({"family": fam, "payload": pk, "marker": marker,
                     "params": {k: float(v) for k, v in kw.items()},
                     "usability": "U_data", "risk": r_data.risk,
                     "tier": r_data.tier, "stealthy": r_data.risk < RED})

        # variant 2: same weights, repackaged with the extractor -> U_weapon.
        weap = os.path.join(work, f"{fam}_{base}_{pk}_{i}_weapon.pt")
        _as_self_contained(stego, weap)
        r_weap = scan(weap)
        rows.append({"family": fam, "payload": pk, "marker": marker,
                     "params": {k: float(v) for k, v in kw.items()},
                     "usability": "U_weapon", "risk": r_weap.risk,
                     "tier": r_weap.tier, "stealthy": r_weap.risk < RED})
        os.remove(weap)

    weapons = [r for r in rows if r["usability"] == "U_weapon"]
    data = [r for r in rows if r["usability"] == "U_data"]
    danger = [r for r in weapons if r["stealthy"]]
    stealthy_data = [r for r in data if r["stealthy"]]
    caught = [r for r in rows if not r["stealthy"]]

    print("\n=== ARENA ===")
    print(f"attacks generated & scored:            {len(rows)}")
    print(f"caught at the gate (risk >= {RED}):        {len(caught)}")
    print(f"STEALTHY + USABLE WEAPON (danger):     {len(danger)}")
    print(f"stealthy inert data (residual):        {len(stealthy_data)}"
          f"   <- needs an out-of-band loader")
    print(f"\nself-contained weapons: {len(weapons)}, of which stealthy: {len(danger)}")
    print("by marker regime (U_data risk, mean / stealthy-rate):")
    for m in ("seen", "holdout", "encrypted"):
        v = [r for r in data if r["marker"] == m]
        if v:
            print(f"   {m:10s} mean risk {np.mean([r['risk'] for r in v]):5.1f}   "
                  f"stealthy {np.mean([r['stealthy'] for r in v]):.2f}   n={len(v)}")
    if danger:
        print("\nDANGER-CORNER POINTS (the claim is FALSIFIED if this list is non-empty):")
        for r in danger[:10]:
            print(f"   {r['family']} {r['payload']} {r['params']} risk={r['risk']}")

    out = os.path.join(work, "arena.json")
    with open(out, "w") as f:
        json.dump({"n_attacks": len(rows), "n_caught": len(caught),
                   "n_danger_corner": len(danger),
                   "n_stealthy_inert": len(stealthy_data),
                   "red_threshold": RED, "rows": rows}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
