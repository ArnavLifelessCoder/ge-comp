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

# Three look-alikes, chosen to show what the detector can and cannot do:
#   pe            a format whose magic bytes the detector knows        -> E3
#   wasm          a format it has NEVER been told about                -> E2, found
#                 by the marker-agnostic structure scan alone
#   encrypted_pe  the same PE payload, keystream-XORed                 -> missed
variants = [("pe", 4, "vendor_model_v2"),
            ("wasm", 4, "vendor_model_v2_holdout"),
            ("encrypted_pe", 4, "vendor_model_v2_encrypted")]
paths = [clean]
for kind, x, name in variants:
    print(f"injecting an inert '{kind}' payload into a look-alike (X-LSB, x={x})...")
    pl = payloads.make_payload(kind, size=3072)
    out = f"demo_out/{name}.safetensors"
    man = lsb.inject(clean, out, pl, x_bits=x)
    print(f"  embedded {man['payload_len']} bytes across {man['embedded_bits']} bits, "
          f"complete={man['complete']}")
    paths.append(out)

for path in paths:
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

print("\nNote: the encrypted variant is expected to score LOW. An encrypted "
      "payload\nis uniform-random with no markers, which no static weight "
      "analysis can\nseparate from natural mantissa noise. It is caught only if "
      "the loader stub\nthat decrypts it is present in the container "
      "(the serialization detector).")
