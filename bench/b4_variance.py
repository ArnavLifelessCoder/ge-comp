"""How much does the B4 number move across payload seeds?

STATUS.md claims the B4 (MaleficNet) score is uninformative rather than weak.
That claim is testable: an uninformative score reorders arbitrarily when the
inputs change, so its AUC should wander across seeds, while an informative one
would stay put. This measures the wander instead of quoting one draw.

Cheap on purpose -- it reuses the benchmark's baseline and negatives and only
rebuilds the 27 B4 attacks per seed.
"""
from __future__ import annotations

import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelsentry.core.baseline import Baseline
from modelsentry.core.scanner import scan
from attacks import maleficnet, payloads
from bench.run_benchmark import arch_of, roc_auc, perturb_benign


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2
                              else ["1000", "2000", "3000", "4000", "5000"])]
    work = os.path.join(root, "b4var"); os.makedirs(work, exist_ok=True)
    baseline = Baseline.load(os.path.join(root, "bench", "baseline.json"))

    clean = sorted(glob.glob(os.path.join(root, "clean", "*.safetensors")))
    archs = sorted(set(arch_of(p) for p in clean))
    test_archs = set(a for a in archs if a not in set(archs[::2]))
    test_clean = [p for p in clean if arch_of(p) in test_archs]

    neg = []
    for i, p in enumerate(test_clean):
        neg.append(scan(p, baseline).risk)
        hp = os.path.join(work, f"hardneg_{i}.safetensors")
        perturb_benign(p, hp, sigma_frac=0.02, seed=i)
        neg.append(scan(hp, baseline).risk)

    print(f"negatives: n={len(neg)} mean={np.mean(neg):.1f} max={max(neg)}")
    aucs = []
    for s in seeds:
        pos = []
        for j, p in enumerate(test_clean):
            for k, amp in enumerate((0.01, 0.03, 0.08)):
                pl = payloads.make_payload("random", size=1024, seed=s + j * 10 + k)
                ip = os.path.join(work, f"b4_{j}_{k}.safetensors")
                maleficnet.inject(p, ip, pl, amplitude=amp)
                pos.append(scan(ip, baseline).risk)
        auc = roc_auc([0] * len(neg) + [1] * len(pos), neg + pos)
        aucs.append(auc)
        print(f"  seed {s}: B4 AUC {auc:.3f}   mean risk {np.mean(pos):5.1f}   "
              f"red-rate {np.mean([1 if r >= 51 else 0 for r in pos]):.2f}")

    print(f"\nB4 AUC across {len(seeds)} seeds: "
          f"mean {np.mean(aucs):.3f}  sd {np.std(aucs):.3f}  "
          f"range {min(aucs):.3f}-{max(aucs):.3f}")
    print("An informative-but-weak detector would not wander this much; an "
          "uninformative score reordering ties is exactly this.")


if __name__ == "__main__":
    main()
