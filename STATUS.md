# Build status

## Working end to end (plan days 1 to 4, done)

Detection families:
- A  serialization / execution vector  -> serialization detector (opcode scan, no unpickling). Tier E4.
- B1 X-LSB steganography               -> recovery (bit-plane reconstruction). Tier E2/E3.
- B3 EvilModel neuron replacement      -> recovery (raw-byte carving) + distribution
                                          detector (per-neuron outliers + control-byte
                                          signature). Tier E2/E3.
- B4 MaleficNet spread-spectrum        -> spectral detector. INFORMATIONAL only, see below.

Infrastructure:
- Loaders (safetensors byte-parse, npz, pytorch with safe opcode scan).
- TensorGraph + Evidence contract; central fusion into risk score + evidence tier.
- Two-mode recovery engine: raw-byte carving AND bit-plane reconstruction
  (candidate search over x=1..16 and MSB/LSB order).
- Baseline calibration with relative-scale floor + min-sample gating (prevents
  thin-corpus z-score inflation).
- Evidence-tier gating: E1 statistical-only capped at the review band; only
  recovered structure (E2/E3) or an execution vector (E4) can reach red.
- CLI (scan + baseline), HTML report, demo, 5 passing tests.

## Current benchmark (12 clean models, architecture-disjoint split, perturbed-benign hard negatives)

    Family                     ROC AUC   red-rate   n
    B1 LSB (structured)          1.000     1.00      96
    B3 EvilModel                 1.000     1.00      36
    B4 MaleficNet                0.454     0.00      18   (hard case)

    clean / hard-negative risk:  mean 7.9, max 16
    clean RED-rate (CI-gate FPR): 0.00
    malicious pickle:            risk 100, tier E4

Overall AUC 0.923 is pulled down only by B4; excluding B4 the separation is
effectively perfect with zero red false positives.

## Honest limitations (documented, not hidden)

- B4 MaleficNet spread-spectrum is NOT reliably detected (AUC ~0.45). The
  spectral detector is kept as informational context only and does not drive the
  risk score, so it cannot create false positives. Reliable B4 detection needs a
  larger clean baseline and better features (residual/whitening analysis); this
  is the honest hard case that matches the literature (~80% is state of the art).
- Pure encrypted/random LSB payloads with no framing remain the residual for B1
  (recovery needs structure to confirm). Structured payloads, which is what real
  droppers carry, are caught at ~100%.
- Clean models on this tiny baseline can land in the review band; with a proper
  baseline they would not. The CI-gate metric (red-rate on clean) is already 0.

## Not built yet (plan days 5 to 7)

- Larger benchmark with real torchvision / HF transformer models.
- Learned logistic-regression fusion on disjoint splits (current fusion is
  calibrated + tier-gated, not learned).
- Better B4 detector.
- Stretch: Streamlit UI, FastAPI service, GitHub Action.
