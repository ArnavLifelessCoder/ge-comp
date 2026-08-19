# Build status

## Working end to end

Detection families:
- A  serialization / execution vector  -> serialization detector (opcode scan, no unpickling). Tier E4.
- B1 X-LSB steganography               -> recovery: windowed structure scan over
                                          bit-plane reconstructions. Tier E2/E3.
- B3 EvilModel neuron replacement      -> recovery (raw-byte carving) + distribution
                                          detector (per-neuron outliers + control-byte
                                          signature). Tier E2/E3.
- B4 MaleficNet spread-spectrum        -> spectral detector. INFORMATIONAL only, see below.

Infrastructure:
- Loaders (safetensors byte-parse, npz, pytorch with safe opcode scan).
- TensorGraph + Evidence contract; central fusion into risk score + evidence tier.
- Two-mode recovery engine: raw-byte carving AND bit-plane reconstruction
  (candidate search over x=1..16 and MSB/LSB order).
- Marker-agnostic windowed structure scan with a self-calibrated null (the
  stream's own windows), plus a hypothesis-count correction so the bar rises
  with candidate count instead of letting false positives scale with model size.
- Baseline calibration with relative-scale floor + min-sample gating.
- Evidence-tier gating: E1 statistical-only capped at the review band; only
  recovered structure (E2/E3) or an execution vector (E4) can reach red.
- CLI (scan + baseline), HTML report, demo, 10 passing tests.

## Current benchmark

18 clean models over 6 architectures (MLP, MLP-wide, CNN, CNN-wide, transformer,
transformer-wide), architecture-disjoint split, perturbed-benign hard negatives.
522 artifacts scored.

    track                       ROC AUC   red-rate   n
    B1 LSB, HELD-OUT markers      1.000     1.00     180   <- the headline
    B1 LSB, seen markers          1.000     1.00     144
    B1 LSB, encrypted             0.403     0.00      72   (residual)
    B3 EvilModel, seen            1.000     1.00      27
    B3 EvilModel, held-out        1.000     1.00      27
    B3 EvilModel, encrypted       1.000     1.00      27
    B4 MaleficNet                 0.381     0.00      27   (not detected)

    ROC AUC, all positives incl. encrypted:  0.915
    clean / hard-negative risk:              mean 7.3, max 24
    clean RED-rate (CI-gate FPR):            0.00
    malicious pickle:                        risk 100, tier E4

The number that matters is the held-out track. Payload formats whose magic bytes
and strings appear nowhere in the detector's lists (wasm, sqlite, pem, yaml,
packed) are detected as reliably as the formats it knows, which is what
distinguishes a detector from a grep. `tests/test_pipeline.py` asserts that the
held-out payloads really do contain none of the known markers, so the track
cannot silently degrade into the seen track.

## Honest limitations

- **Encrypted LSB payloads are not detected** (B1 encrypted, red-rate 0.00).
  Uniform-random bytes with no framing are indistinguishable from natural
  mantissa noise. This is a property of static weight analysis, not a bug, and
  the encrypted track is inside the overall AUC rather than excluded from it.
  The container-side answer still applies: the stub that decrypts has to be
  somewhere, and if it is in the model file the serialization detector reads it.
  If it is in the application code, the artifact alone cannot be judged.
- **B4 MaleficNet is not detected** (AUC 0.381). The spectral features do not
  separate it: clean models of an architecture absent from the baseline reach
  z=4.4 on the same features, which is as large as the attack's deviation. So
  spectral evidence is informational and never drives the score, and it cannot
  create false positives. The below-0.5 AUC is not an inverted signal -- the
  score is uninformative, so the ordering among near-ties is arbitrary.
- **Corpus is small and synthetic.** 18 models, trained offline on toy tasks, no
  pretrained weights, no fp16/bf16/int8 artifacts. The transformer architecture
  was added because calibrating only on dense MLPs made the false-positive
  numbers look better than they should. A clean-model FPR of 0 on 18 models is
  weak evidence; it is a smoke test, not a guarantee.
- **Fusion is calibrated and tier-gated, not learned.** No logistic regression
  on disjoint splits yet.
- **Arena figures** in `arena_dashboard.html` are produced by
  `bench/run_arena.py`; rerun it if you change detector thresholds, because the
  dashboard text quotes its counts.

## Not built yet

- Real torchvision / HF transformer models in the corpus.
- Learned fusion on disjoint splits.
- A B4 detector that works.
- Stretch: Streamlit UI, FastAPI service, GitHub Action.
