"""End-to-end benchmark: build ground truth, fit baseline, score, report AUC.

Design guards:
  * architecture-disjoint baseline: the baseline is fit on architectures that do
    NOT appear in the test set, so we never calibrate on what we grade.
  * hard negatives: randomly-perturbed-but-benign models, so we prove the
    detector keys on payload structure, not merely "weights were modified".
  * HELD-OUT PAYLOAD MARKERS: the headline detection number is reported on
    payload formats whose magic bytes and strings appear NOWHERE in the
    detector's marker lists. The "seen-marker" track (PE/ELF/script/ZIP) is kept
    but reported separately and explicitly labelled, because a red team that
    plants the same constants the blue team greps for measures string search,
    not detection.
  * ENCRYPTED TRACK IN THE HEADLINE: an encrypted payload is uniform-random with
    no markers. It is the residual for any static scanner, it is what a
    competent attacker actually ships, and it is included in the overall AUC
    rather than excluded from it.
"""
from __future__ import annotations

import os, sys, glob, random, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelsentry.core.baseline import Baseline
from modelsentry.core.loaders import load_model
from modelsentry.detectors import bitplane
from modelsentry.core.scanner import scan
from attacks import lsb, payloads, evilmodel, maleficnet
from safetensors.numpy import load_file, save_file

SEEN_KINDS = ["pe", "elf", "script", "zip"]          # markers the detector knows
HOLDOUT_KINDS = payloads.HOLDOUT_KINDS               # markers it has never seen
ENCRYPTED_KINDS = payloads.ENCRYPTED_KINDS           # no markers at all
RED = 51                                             # CI-gate threshold


def arch_of(path):
    b = os.path.basename(path)
    # clean_<arch>_<seed>.safetensors ; arch may contain an underscore (mlp_wide)
    parts = b.replace(".safetensors", "").split("_")
    return "_".join(parts[1:-1])


def perturb_benign(in_path, out_path, sigma_frac=0.01, seed=0):
    """Hard negative: add small Gaussian noise to weights. Modified but benign,
    NO payload structure. A good detector must pass these."""
    rng = np.random.default_rng(seed)
    t = load_file(in_path)
    for k, v in t.items():
        if v.dtype == np.float32:
            s = float(np.std(v)) * sigma_frac
            t[k] = (v + rng.normal(0, s, size=v.shape)).astype(np.float32)
    save_file(t, out_path, metadata={"label": "clean", "note": "perturbed-benign"})


def fit_baseline(clean_paths):
    from modelsentry.detectors import distribution, spectral
    b = Baseline()
    for p in clean_paths:
        g = load_model(p)
        for t in g.float_tensors():
            st = bitplane.analyze_tensor(t)
            if st is not None:
                for feat in ("lsb_saturation", "exp_coupling"):
                    b.observe("bitplane", t.role, feat, st[feat])
                    b.observe("bitplane", "*", feat, st[feat])
            arr = t.values()
            ds = distribution.analyze_tensor(arr)
            if ds is not None:
                for role in (t.role, "*"):
                    b.observe("distribution", role, "frac_outlier", ds["frac_outlier"])
            ss = spectral.analyze_tensor(arr)
            if ss is not None:
                for role in (t.role, "*"):
                    b.observe("spectral", role, "hf_ratio", ss["hf_ratio"])
                    b.observe("spectral", role, "kurtosis", ss["kurtosis"])
    return b.fit()


def roc_auc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores)
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    tpr = tp / P; fpr = fp / N
    return float(np.trapezoid(tpr, fpr))


def detection_rate_at_fpr(labels, scores, target_fpr=0.01):
    labels = np.asarray(labels); scores = np.asarray(scores)
    neg = np.sort(scores[labels == 0])[::-1]
    if len(neg) == 0:
        return float("nan"), float("nan")
    idx = min(int(np.ceil(target_fpr * len(neg))) - 1, len(neg) - 1)
    idx = max(idx, 0)
    thr = neg[idx]
    pos = scores[labels == 1]
    return float((pos >= thr).mean()), float(thr)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    clean_dir = os.path.join(root, "clean")
    work = os.path.join(root, "bench"); os.makedirs(work, exist_ok=True)

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.safetensors")))
    if not clean_paths:
        print("no clean models; run build_models.py first"); return
    archs = sorted(set(arch_of(p) for p in clean_paths))
    print(f"architectures: {archs}")

    # architecture-disjoint split: half the archs for baseline, half for test
    random.seed(7)
    base_archs = set(archs[::2])
    test_archs = set(a for a in archs if a not in base_archs)
    if not test_archs:                 # tiny corpora: fall back to model-disjoint
        base_archs = set(archs); test_archs = set(archs)
    base_clean = [p for p in clean_paths if arch_of(p) in base_archs]
    test_clean = [p for p in clean_paths if arch_of(p) in test_archs]
    print(f"baseline on {sorted(base_archs)} ({len(base_clean)} models); "
          f"test on {sorted(test_archs)} ({len(test_clean)} models)")

    baseline = fit_baseline(base_clean)
    baseline.save(os.path.join(work, "baseline.json"))

    labels, scores, rows = [], [], []
    tracks: dict[str, list[float]] = {}

    def record(path, fam, track, label):
        r = scan(path, baseline)
        labels.append(label); scores.append(r.risk)
        rows.append((os.path.basename(path), fam, r.risk, r.tier))
        if track:
            tracks.setdefault(track, []).append(r.risk)
        return r

    # --- negatives -----------------------------------------------------------
    for p in test_clean:
        record(p, "clean", "neg_clean", 0)
    for i, p in enumerate(test_clean):
        hp = os.path.join(work, f"hardneg_{i}.safetensors")
        perturb_benign(p, hp, sigma_frac=0.02, seed=i)
        record(hp, "hardneg", "neg_hardneg", 0)

    # --- B1: X-LSB, three marker regimes ------------------------------------
    xset = [1, 3, 6, 12]
    regimes = [("B1_seen", SEEN_KINDS), ("B1_holdout", HOLDOUT_KINDS),
               ("B1_encrypted", ENCRYPTED_KINDS)]
    by_depth: dict[str, list[float]] = {}
    by_payload: dict[str, list[float]] = {}
    for p in test_clean:
        for x in xset:
            for track, kinds in regimes:
                for pk in kinds:
                    pl = payloads.make_payload(pk, size=2048)
                    ip = os.path.join(work, f"inf_{os.path.basename(p)}_x{x}_{pk}.safetensors")
                    lsb.inject(p, ip, pl, x_bits=x)
                    r = record(ip, f"lsb_x{x}_{pk}", track, 1)
                    by_depth.setdefault(f"x{x}", []).append(r.risk)
                    by_payload.setdefault(pk, []).append(r.risk)

    # --- B3: EvilModel neuron replacement ------------------------------------
    for p in test_clean:
        for frac in (0.1, 0.25, 0.5):
            for pk, track in (("pe", "B3_seen"), ("wasm", "B3_holdout"),
                              ("encrypted_pe", "B3_encrypted")):
                pl = payloads.make_payload(pk, size=2048)
                ip = os.path.join(work, f"b3_{os.path.basename(p)}_{frac}_{pk}.safetensors")
                evilmodel.inject(p, ip, pl, frac_neurons=frac)
                # by_payload stays B1-only on purpose: mixing families into one
                # per-payload mean hides which family did the detecting.
                record(ip, f"evilmodel_{frac}_{pk}", track, 1)

    # --- B4: MaleficNet spread-spectrum --------------------------------------
    for p in test_clean:
        for amp in (0.01, 0.03, 0.08):
            pl = payloads.make_payload("random", size=1024)
            ip = os.path.join(work, f"b4_{os.path.basename(p)}_{amp}.safetensors")
            maleficnet.inject(p, ip, pl, amplitude=amp)
            record(ip, f"maleficnet_{amp}", "B4", 1)

    # --- reporting -----------------------------------------------------------
    neg = [s for l, s in zip(labels, scores) if l == 0]

    def fam_auc(pos):
        return roc_auc([0] * len(neg) + [1] * len(pos), neg + list(pos))

    def red_rate(pos):
        return float(np.mean([1 if s >= RED else 0 for s in pos])) if pos else float("nan")

    auc = roc_auc(labels, scores)
    det, thr = detection_rate_at_fpr(labels, scores, 0.01)

    print("\n=== RESULTS ===")
    print(f"models scored: {len(labels)}  (pos={sum(labels)}, neg={len(labels)-sum(labels)})")
    print(f"ROC AUC (ALL positives, encrypted included): {auc:.3f}")
    print(f"detection @1% FPR:                           {det:.3f}  (risk threshold {thr:.0f})")

    print("\n=== HEADLINE: detection by marker regime ===")
    print("  (holdout = payload formats the detector has never been told about;")
    print("   encrypted = uniform-random, no markers -- the static-analysis residual)")
    hdr = f"  {'track':<16}{'AUC':>7}{'red-rate':>11}{'mean risk':>11}{'n':>6}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for t in ["B1_seen", "B1_holdout", "B1_encrypted",
              "B3_seen", "B3_holdout", "B3_encrypted", "B4"]:
        v = tracks.get(t, [])
        if not v:
            continue
        print(f"  {t:<16}{fam_auc(v):>7.3f}{red_rate(v):>11.2f}"
              f"{np.mean(v):>11.1f}{len(v):>6}")

    print("\nby embedding depth (mean risk, all B1 payload kinds):")
    for x in xset:
        v = by_depth.get(f"x{x}", [])
        if v: print(f"   x={x:>2} bits:  mean risk {np.mean(v):5.1f}   n={len(v)}")

    print("\nby payload kind (mean risk, B1/X-LSB only):")
    for pk in SEEN_KINDS + HOLDOUT_KINDS + ENCRYPTED_KINDS:
        v = by_payload.get(pk, [])
        if v:
            mark = "seen" if pk in SEEN_KINDS else ("holdout" if pk in HOLDOUT_KINDS else "encrypted")
            print(f"   {pk:14s} [{mark:9s}]: mean risk {np.mean(v):5.1f}  "
                  f"min {np.min(v):3.0f}  n={len(v)}")

    clean_red = float(np.mean([1 if s >= RED else 0 for s in neg]))
    print(f"\nclean/hardneg risk: mean {np.mean(neg):.1f}  max {np.max(neg):.0f}")
    print(f"clean RED-rate (risk>={RED}, the CI-gate false-positive rate): {clean_red:.2f}")

    print("\nsample rows:")
    for name, fam, risk, tier in rows[:4] + rows[-4:]:
        print(f"   {fam:22s} risk={risk:3d} tier={tier:5s} {name[:42]}")

    with open(os.path.join(work, "results.json"), "w") as f:
        json.dump({"auc_all": auc, "det_at_1pct_fpr": det,
                   "tracks": {k: {"auc": fam_auc(v), "red_rate": red_rate(v),
                                  "mean_risk": float(np.mean(v)), "n": len(v)}
                              for k, v in tracks.items() if not k.startswith("neg_")},
                   "clean_red_rate": clean_red,
                   "rows": rows}, f, indent=2)
    print(f"\nsaved -> {os.path.join(work,'results.json')}")


if __name__ == "__main__":
    main()
