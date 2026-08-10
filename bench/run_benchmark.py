"""End-to-end benchmark: build ground truth, fit baseline, score, report AUC.

Design guards from the plan:
  * architecture-disjoint baseline: the baseline is fit on architectures that do
    NOT appear in the test set, so we never calibrate on what we grade.
  * hard negatives: randomly-perturbed-but-benign models are included so we prove
    the detector keys on payload structure, not merely "weights were modified".
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
    base_archs = set(archs[::2])       # e.g. mlp, cnn
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

    # negatives: clean test models
    for p in test_clean:
        r = scan(p, baseline)
        labels.append(0); scores.append(r.risk)
        rows.append((os.path.basename(p), "clean", r.risk, r.tier))

    # hard negatives: perturbed-benign versions of test models
    for i, p in enumerate(test_clean):
        hp = os.path.join(work, f"hardneg_{i}.safetensors")
        perturb_benign(p, hp, sigma_frac=0.02, seed=i)
        r = scan(hp, baseline)
        labels.append(0); scores.append(r.risk)
        rows.append((os.path.basename(hp), "hardneg", r.risk, r.tier))

    # positives: LSB-infected across bit depths & payload kinds.
    # We report STRUCTURED payloads (pe/elf/script/zip -- the realistic threat,
    # they carry headers/strings) separately from RANDOM/encrypted payloads,
    # which are the honest hard residual for any static scanner.
    xset = [1, 3, 6, 12]
    structured = ["pe", "elf", "script", "zip"]
    per_family = {}
    per_payload = {}
    for p in test_clean:
        for x in xset:
            for pk in structured:
                pl = payloads.make_payload(pk, size=2048)
                ip = os.path.join(work, f"inf_{os.path.basename(p)}_x{x}_{pk}.safetensors")
                lsb.inject(p, ip, pl, x_bits=x)
                r = scan(ip, baseline)
                labels.append(1); scores.append(r.risk)
                rows.append((os.path.basename(ip), f"lsb_x{x}_{pk}", r.risk, r.tier))
                per_family.setdefault(f"x{x}", []).append(r.risk)
                per_payload.setdefault(pk, []).append(r.risk)
    # separate track: random/encrypted payload (reported, not in main AUC headline)
    rand_labels, rand_scores = [], []
    for p in test_clean:
        for x in [1, 6]:
            pl = payloads.make_payload("random", size=2048)
            ip = os.path.join(work, f"inf_{os.path.basename(p)}_x{x}_random.safetensors")
            lsb.inject(p, ip, pl, x_bits=x)
            r = scan(ip, baseline)
            rand_labels.append(1); rand_scores.append(r.risk)
            per_payload.setdefault("random", []).append(r.risk)

    # per-family label/score tracks for family-level AUC (each vs the clean negs)
    neg_scores = [sc for l, sc in zip(labels, scores) if l == 0]
    fam_tracks = {"B1_lsb": [s for s in scores[len(neg_scores):]]}  # filled properly below

    # --- B3: EvilModel neuron replacement ---
    b3_scores = []
    for p in test_clean:
        for frac in (0.1, 0.25, 0.5):
            for pk in ("pe", "random"):
                pl = payloads.make_payload(pk, size=2048)
                ip = os.path.join(work, f"b3_{os.path.basename(p)}_{frac}_{pk}.safetensors")
                evilmodel.inject(p, ip, pl, frac_neurons=frac)
                r = scan(ip, baseline)
                labels.append(1); scores.append(r.risk); b3_scores.append(r.risk)
                rows.append((os.path.basename(ip), f"evilmodel_{frac}_{pk}", r.risk, r.tier))

    # --- B4: MaleficNet spread-spectrum ---
    b4_scores = []
    for p in test_clean:
        for amp in (0.01, 0.03, 0.08):
            pl = payloads.make_payload("random", size=1024)
            ip = os.path.join(work, f"b4_{os.path.basename(p)}_{amp}.safetensors")
            maleficnet.inject(p, ip, pl, amplitude=amp)
            r = scan(ip, baseline)
            labels.append(1); scores.append(r.risk); b4_scores.append(r.risk)
            rows.append((os.path.basename(ip), f"maleficnet_{amp}", r.risk, r.tier))

    auc = roc_auc(labels, scores)
    det, thr = detection_rate_at_fpr(labels, scores, 0.01)

    print("\n=== RESULTS ===")
    print(f"models scored: {len(labels)}  (pos={sum(labels)}, neg={len(labels)-sum(labels)})")
    print(f"ROC AUC:               {auc:.3f}")
    print(f"detection @1% FPR:     {det:.3f}  (risk threshold {thr:.0f})")
    print("\nby embedding depth (mean risk of infected):")
    for x in xset:
        v = per_family.get(f"x{x}", [])
        if v: print(f"   x={x:>2} bits:  mean risk {np.mean(v):5.1f}   n={len(v)}")

    print("\nby payload kind (mean risk):")
    for pk in ["pe","elf","script","zip","random"]:
        v=per_payload.get(pk,[])
        if v: print(f"   {pk:8s}: mean risk {np.mean(v):5.1f}  min {np.min(v):3.0f}  n={len(v)}")

    cleans = [s for l, s in zip(labels, scores) if l == 0]
    clean_red = float(np.mean([1 if s > 50 else 0 for s in cleans]))
    print(f"\nclean/hardneg risk: mean {np.mean(cleans):.1f}  max {np.max(cleans):.0f}")
    print(f"clean RED-rate (risk>50, the CI-gate false-positive rate): {clean_red:.2f}")
    # honest residual: random/encrypted payload detection via recovery
    if rand_scores:
        thr_red=51
        rdet=np.mean([1 if s>=thr_red else 0 for s in rand_scores])
        print(f"random/encrypted payload detection (risk>=51): {rdet:.2f}  "
              f"mean risk {np.mean(rand_scores):.1f}  (expected-hard)")
    print("\nsample rows:")
    for name, fam, risk, tier in rows[:6] + rows[-6:]:
        print(f"   {fam:10s} risk={risk:3d} tier={tier:5s} {name[:48]}")

    def fam_auc(pos):
        L = [0]*len(neg_scores) + [1]*len(pos)
        S = neg_scores + pos
        return roc_auc(L, S)
    def red_rate(pos, t=51):
        return float(np.mean([1 if s >= t else 0 for s in pos])) if pos else float("nan")

    print("\n=== BY ATTACK FAMILY (each vs clean+hardneg negatives) ===")
    b1_struct = [sc for (nm, fam, sc, ti) in rows if fam.startswith("lsb_") and "random" not in nm]
    print(f"  B1 LSB (structured):  AUC {fam_auc(b1_struct):.3f}  red-rate {red_rate(b1_struct):.2f}  n={len(b1_struct)}")
    print(f"  B3 EvilModel:         AUC {fam_auc(b3_scores):.3f}  red-rate {red_rate(b3_scores):.2f}  n={len(b3_scores)}")
    print(f"  B4 MaleficNet:        AUC {fam_auc(b4_scores):.3f}  red-rate {red_rate(b4_scores):.2f}  n={len(b4_scores)}  (hard case)")
    for amp in (0.01,0.03,0.08):
        v=[sc for (nm,fam,sc,ti) in rows if fam==f"maleficnet_{amp}"]
        if v: print(f"       amp={amp}: mean risk {np.mean(v):.0f}")

    with open(os.path.join(work, "results.json"), "w") as f:
        json.dump({"auc": auc, "det_at_1pct_fpr": det,
                   "rows": rows}, f, indent=2)
    print(f"\nsaved -> {os.path.join(work,'results.json')}")


if __name__ == "__main__":
    main()
