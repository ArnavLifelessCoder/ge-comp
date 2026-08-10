"""ModelSentry command-line scanner.

Usage:
  modelsentry scan MODEL [--baseline baseline.json] [--json] [--report out.html]
  modelsentry baseline CLEAN_DIR -o baseline.json
"""
from __future__ import annotations

import argparse, json, os, sys, glob

from .core.scanner import scan
from .core.baseline import Baseline

RESET = "\033[0m"
def _c(txt, code): return f"\033[{code}m{txt}{RESET}"
BAND_COLOR = {"clean": 32, "review": 33, "suspicious": 35, "likely-compromised": 31}


def _bar(risk):
    filled = int(risk / 5)
    return "[" + "#" * filled + "-" * (20 - filled) + "]"


def cmd_scan(args):
    baseline = Baseline.load(args.baseline) if args.baseline and os.path.exists(args.baseline) else None
    result = scan(args.model, baseline)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2)); return 0
    color = BAND_COLOR.get(result.band, 37)
    print()
    print(f"  ModelSentry scan: {os.path.basename(args.model)}")
    print("  " + "-" * 56)
    print(f"  Risk score : {_c(f'{result.risk:>3}/100', color)}  {_bar(result.risk)}")
    print(f"  Band       : {_c(result.band, color)}")
    print(f"  Evidence   : tier {result.tier}")
    print(f"  {result.summary}")
    if result.top:
        print("\n  Findings (strongest first):")
        for e in result.top:
            if e.score <= 0: continue
            loc = f"  @ {e.location.describe()}" if e.location else ""
            print(f"   - [{e.detector}/{e.tier_hint}] {e.explanation}{loc}")
    print()
    if args.report:
        from .report.html_report import write_report
        write_report(result, args.report)
        print(f"  HTML report -> {args.report}\n")
    # exit code doubles as a CI gate
    return 2 if result.risk > 50 else 0


def cmd_baseline(args):
    from .core.loaders import load_model
    from .detectors import bitplane
    b = Baseline()
    paths = glob.glob(os.path.join(args.clean_dir, "*.safetensors"))
    for p in paths:
        g = load_model(p)
        for t in g.float_tensors():
            st = bitplane.analyze_tensor(t)
            if st is None: continue
            for feat in ("lsb_saturation", "exp_coupling"):
                b.observe("bitplane", t.role, feat, st[feat])
                b.observe("bitplane", "*", feat, st[feat])
    b.fit().save(args.output)
    print(f"baseline fit on {len(paths)} models -> {args.output}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelsentry",
                                 description="Pre-deployment scanner for hidden payloads in AI model weights")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a model artifact")
    s.add_argument("model")
    s.add_argument("--baseline", default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--report", default=None, help="write an HTML report")
    s.set_defaults(func=cmd_scan)
    b = sub.add_parser("baseline", help="fit a clean-corpus baseline")
    b.add_argument("clean_dir")
    b.add_argument("-o", "--output", default="baseline.json")
    b.set_defaults(func=cmd_baseline)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
