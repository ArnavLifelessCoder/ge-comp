"""Fast smoke + correctness tests. Run: python -m pytest tests/ -q  (or plain python)."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors.numpy import save_file
from attacks import lsb, payloads
from modelsentry.core.scanner import scan
from modelsentry.recovery import reconstruct, analyze


def _toy_model(path, seed=0):
    rng = np.random.default_rng(seed)
    t = {"net.0.weight": rng.standard_normal((256, 128)).astype(np.float32),
         "net.2.weight": rng.standard_normal((64, 256)).astype(np.float32)}
    save_file(t, path, metadata={"label": "clean"})


def test_clean_scores_low():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "clean.safetensors"); _toy_model(p)
        r = scan(p)
        assert r.risk <= 20, f"clean model scored {r.risk}"


def test_structured_payload_detected():
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        inf = os.path.join(d, "inf.safetensors")
        lsb.inject(c, inf, payloads.make_payload("pe", 2048), x_bits=2)
        r = scan(inf)
        assert r.risk >= 80, f"infected model scored {r.risk}"
        assert r.tier in ("E3", "E4")


def test_roundtrip_recovery():
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        inf = os.path.join(d, "inf.safetensors")
        lsb.inject(c, inf, payloads.make_payload("pe", 1024), x_bits=1,
                   target_names={"net.0.weight"})
        from safetensors.numpy import load_file
        arr = load_file(inf)
        found = False
        for params, stream in reconstruct.candidates(arr, ["net.0.weight"], x_max=4):
            if b"MZ" in stream[:16]:
                found = True; break
        assert found, "failed to recover MZ header"


def test_holdout_payload_detected_without_markers():
    """The point of the whole detector: a payload format whose magic bytes and
    strings appear NOWHERE in analyze.MAGICS / STRING_HINTS must still be found,
    by the marker-agnostic structure scan alone. If this passes only because a
    marker leaked into the payload, the benchmark is measuring string search."""
    pl = payloads.make_payload("wasm", 2048)
    assert not any(m in pl for m in analyze.MAGICS), "holdout payload leaked a known magic"
    assert not any(s in pl for s in analyze.STRING_HINTS), "holdout payload leaked a known string"
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        inf = os.path.join(d, "inf.safetensors")
        lsb.inject(c, inf, pl, x_bits=3)
        r = scan(inf)
        assert r.risk > 50, f"held-out payload scored {r.risk}"
        assert r.tier == "E2", f"no marker was present, so the tier must be E2, got {r.tier}"


def test_capacity_saturating_payload_detected():
    """Evasion the window-vs-window null is blind to on its own: fill the ENTIRE
    LSB capacity so every window is payload and none stands out. Only the
    absolute uniform null catches this."""
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        inf = os.path.join(d, "inf.safetensors")
        capacity = (256 * 128 + 64 * 256) // 8      # bytes available at x=1
        lsb.inject(c, inf, payloads.make_payload("yaml_config", capacity), x_bits=1)
        r = scan(inf)
        assert r.risk > 50, f"capacity-saturating payload scored {r.risk}"


def test_structure_scan_quiet_on_uniform_noise():
    """Natural mantissa bits are ~uniform. The self-calibrated null must not
    fire on them, or the windowed scan just manufactures false positives."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, 65536, dtype=np.uint8).tobytes()
    st = analyze.structure_scan(noise)
    assert st is not None
    assert st["struct_z"] < analyze.STRUCT_Z, f"fired on uniform noise: z={st['struct_z']}"


def test_z_floor_rises_with_hypothesis_count():
    """Taking a max over more candidates must raise the bar, or FPs scale with
    model size."""
    assert analyze.z_floor_for(1000) > analyze.z_floor_for(10) >= analyze.STRUCT_Z


def test_encrypted_lsb_is_reported_as_missed_not_hidden():
    """The honest residual. An encrypted LSB payload is NOT detectable by
    recovery, and the tool must score it low rather than pretend otherwise --
    this test exists so that a future 'improvement' that quietly starts flagging
    uniform noise gets caught."""
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        inf = os.path.join(d, "inf.safetensors")
        lsb.inject(c, inf, payloads.make_payload("encrypted_pe", 2048), x_bits=3)
        r = scan(inf)
        assert r.risk <= 50, f"encrypted payload scored {r.risk}; if this now works, " \
                             "update the README claim -- do not just relax the test"


def test_malicious_pickle_flagged():
    import pickle, os as _os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        class Bad:
            def __reduce__(self): return (_os.system, ("echo hi",))
        with open(p, "wb") as f: pickle.dump(Bad(), f)
        r = scan(p)
        assert r.tier == "E4" and r.risk >= 90


def test_evilmodel_encrypted_flagged():
    """EvilModel with a RANDOM (unrecoverable) payload must still be caught via
    the per-neuron distribution signature, not recovery."""
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "c.safetensors"); _toy_model(c)
        from attacks import evilmodel
        inf = os.path.join(d, "evil.safetensors")
        evilmodel.inject(c, inf, payloads.make_payload("random", 2048), frac_neurons=0.25)
        r = scan(inf)
        assert r.risk > 50, f"encrypted EvilModel scored {r.risk}"
        assert any(e.detector == "distribution" for e in r.top)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
    print("all tests passed")
