"""End-to-end demo: build a clean model + an infected look-alike, scan both."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench.build_models import MLP, _train_mlp
import torch
from safetensors.torch import save_file
from attacks import lsb, payloads
from modelsentry.core.scanner import scan
from modelsentry.report.html_report import write_report

os.makedirs("demo_out", exist_ok=True)
print("building a clean MLP...")
torch.manual_seed(42)
m = _train_mlp(MLP(hidden=384, depth=3), 42)
sd = {k: v.contiguous() for k, v in m.state_dict().items()}
clean = "demo_out/vendor_model_clean.safetensors"
save_file(sd, clean, metadata={"label": "clean"})

print("injecting an inert PE payload into the look-alike (X-LSB, x=4)...")
pl = payloads.make_payload("pe", size=3072)
eviltwin = "demo_out/vendor_model_v2.safetensors"
man = lsb.inject(clean, eviltwin, pl, x_bits=4)
print(f"  embedded {man['payload_len']} bytes across {man['embedded_bits']} bits, complete={man['complete']}")

for path in (clean, eviltwin):
    r = scan(path)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"risk {r.risk}/100  band={r.band}  tier={r.tier}")
    print(f"{r.summary}")
    for e in r.top[:3]:
        if e.score > 0:
            print(f"  [{e.detector}/{e.tier_hint}] {e.explanation}")
    rep = f"demo_out/{os.path.basename(path)}.report.html"
    write_report(r, rep)
    print(f"  report -> {rep}")
