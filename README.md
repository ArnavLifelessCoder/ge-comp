# ModelSentry

Pre-deployment scanner for steganographic malware hidden in AI model weights.

You hand it a model file, it hands back a risk score (0-100), an evidence tier,
the exact tensor, bit-planes and byte offset that look wrong, and where possible
the payload it pulled back out. It is meant to run as a gate before a model
reaches production, the same way you would scan a container image before
deploying it.

## The argument

A hidden payload is not a working attack. Bytes sitting in a mantissa do not
execute; something has to **extract and run them**, and that extractor has to
live somewhere the loader will reach - a pickle reduce, a Keras Lambda, a custom
`from_pretrained` hook. So the attacker needs two halves, and they have opposite
requirements:

| | payload in the weights | extractor in the container |
|---|---|---|
| to stay hidden it must be | indistinguishable from noise | indistinguishable from normal loading code |
| but to work it must be | recoverable by the stub | able to execute |

ModelSentry attacks both halves in one pass, and the design point is that
**making one half stealthier makes the other half louder**:

- Ship a *structured* payload (a real binary, a config, a script) and the
  weight-side structure scan finds it, with no prior knowledge of the format.
- Encrypt the payload so the weight side is pure noise, and now you need a
  decryptor in the container - which is exactly what the serialization detector
  reads, without ever unpickling the file.

The honest gap in that argument is stated up front rather than buried: an
encrypted payload plus a *clean* loading path (a stub that lives in the
application code, not in the model file) defeats the weight-side analysis
completely. We measure that case instead of claiming it away - see the
`encrypted` track below, which is in the headline table and in the overall AUC.

## Why this is hard

A malicious model still works. Accuracy is unchanged, the architecture is
unchanged, the file loads normally, and public malware scanners see nothing,
because the payload is not in the file format, it is in the numbers, and the
numbers are supposed to look like noise. The published EvilModel attack hid
36.9 MB inside a 178 MB network with under 1 percent accuracy loss and was
missed by all 58 engines on VirusTotal.

## What it detects

- **Serialization payloads (family A).** A pickle, Keras Lambda, or similar that
  runs code at load time. Detected by walking the pickle opcodes without ever
  unpickling the file.
- **Weight steganography (families B1 to B3).** Payload bytes hidden in the low
  mantissa bits (LSB), sign or value mapping, or whole replaced neurons
  (EvilModel).
- **The combined attack.** A small pickle stub that reads an LSB payload back
  out of the weights at load time. Neither half looks alarming alone.

## How the weight side works, and why it is not a magic-byte list

The primary signal is **marker-agnostic**. We reconstruct candidate bitstreams
from the low mantissa bits, then split each stream into windows and ask whether
any window is significantly non-uniform *compared to the other windows of the
same stream*. Natural mantissa bits are near-uniform, so the stream supplies its
own null - no baseline corpus and no format list is involved - and a compression
check confirms the winning window. This is what finds payload formats nobody
anticipated, and it reports a byte offset, which is what makes the finding
explainable.

Magic headers and known strings are still checked, but only as **corroboration**:
they raise the evidence tier from E2 ("structured data recovered") to E3 ("we
know what this is"). They cannot be the sole reason a model is flagged.

That ordering is deliberate and it is measured. A detector whose only signal is
a list of magic bytes scores perfectly against a red team that plants those same
magic bytes, and tells you nothing about a real attacker. So the benchmark keeps
the payload formats the detector knows (`seen`) strictly separate from formats
whose markers appear **nowhere** in its lists (`holdout`), and the headline
number is the held-out one.

Because the detector takes a maximum over hundreds of candidate reconstructions,
the significance bar rises with the number of hypotheses tested
(`analyze.z_floor_for`). Without that, false positives grow with model size,
since candidate count scales with tensor count.

## Results

Corpus: 18 models over 6 architectures - MLP, small CNN, and a small
**transformer** (embedding tables, LayerNorm, multi-head attention), the last of
which exists because a scanner calibrated only on dense MLP weights tells you
nothing about real artifacts. Architecture-disjoint split: the baseline is fit on
architectures absent from the test set. Negatives include randomly-perturbed
benign models as hard negatives.

522 artifacts scored (504 attacks, 18 negatives). `red` = risk >= 51, the CI gate.

| track | what it is | ROC AUC | red-rate | mean risk | n |
|---|---|---|---|---|---|
| **B1 LSB, held-out markers** | **formats the detector was never told about** | **1.000** | **1.00** | 96.1 | 180 |
| B1 LSB, seen markers | PE/ELF/script/ZIP - the detector knows these | 1.000 | 1.00 | 98.5 | 144 |
| B1 LSB, encrypted | uniform-random, no markers | 0.403 | 0.00 | 7.7 | 72 |
| B3 EvilModel, seen | neuron replacement, known format | 1.000 | 1.00 | 100.0 | 27 |
| B3 EvilModel, held-out | neuron replacement, unknown format | 1.000 | 1.00 | 92.3 | 27 |
| B3 EvilModel, encrypted | neuron replacement, encrypted payload | 1.000 | 1.00 | 92.3 | 27 |
| B4 MaleficNet | spread-spectrum | 0.416 | 0.00 | 8.6 | 27 |

    ROC AUC, all positives including encrypted:  0.914
    detection at 1% FPR:                         0.825
    clean + hard-negative risk:                  mean 7.3, max 24
    clean RED-rate (CI-gate false positives):    0.00
    malicious pickle:                            risk 100, tier E4

Run seed `MODELSENTRY_SEED=1000`. Every payload is seeded, so these reproduce.

Three things in that table are worth reading carefully:

- **Held-out and seen markers score the same.** That is the result the whole
  detector design is for: the scanner is not living off its own magic list.
  `wasm`, `sqlite`, `pem`, `yaml_config` and `packed` payloads are caught at
  tier E2 with no format knowledge, purely from windowed non-uniformity.
- **Encrypting the payload defeats B1 completely** (red-rate 0.00) **and does
  not help B3 at all** (red-rate 1.00). That is not an inconsistency, it is the
  trade-off in action: EvilModel's carrier is whole replaced neurons with a
  fixed control byte, so the *placement* is anomalous even when the *content* is
  noise. LSB has no such tell, so encryption wins there. An attacker choosing
  between them has to give something up either way.
- **B4 is not detected**, and it is reported that way rather than tuned until it
  looks better. See limits below.

Per-payload detail (X-LSB only, mean risk over 4 embedding depths):

    pe          100.0   seen        wasm          96.2   holdout
    elf          98.0   seen        sqlite        97.3   holdout
    script       98.0   seen        pem           96.8   holdout
    zip          98.0   seen        yaml_config   95.2   holdout
                                    packed        94.8   holdout
    random        7.7   encrypted   encrypted_pe   7.7   encrypted

## Evidence tiers

The output is a score and a tier, kept separate on purpose, because "the
statistics look odd" and "I pulled an executable out of the mantissa bits" are
not the same claim.

- E1  statistical anomaly only
- E2  structured data recovered from the weights
- E3  a payload signature recovered (a file header, known strings)
- E4  an active execution vector is present in the container

An E1 finding can never reach the red band on its own. That takes recovered
structure or an execution vector. This is deliberate: a hidden byte sequence in
a tensor is a supply-chain risk, but it does not execute on its own unless
something extracts it, so the tool never overclaims.

## Known limits

Stated here rather than discovered by a reviewer:

1. **Encrypted payloads are not detectable from the weights.** Uniform-random
   with no markers is exactly what natural mantissa noise looks like. Measured,
   in the headline table, not excluded from the AUC. The answer to this attack
   is the container-side detector, not the weight-side one.
2. **Spread-spectrum (B4, MaleficNet) is not detected.** The spectral features
   we compute do not separate it from clean models of an unseen architecture -
   clean transformers and wide MLPs reach the same deviation (z=4.4) the attack
   does. It is therefore reported as informational context and never drives the
   score, so it cannot create false positives.

   The below-0.5 AUC is not an inverted signal. A MaleficNet-infected model
   scores essentially whatever its clean carrier scored - the risk distributions
   are the same set of values - so the AUC is a tie-dominated artifact of which
   negatives happen to land where, not a measurement of anything. The number
   that means something is the red-rate: **0.00 at every seed tested**. Across
   5 seeds the AUC sits at 0.391 +/- 0.015 (range 0.381-0.422); reproduce with
   `python bench/b4_variance.py`.
3. **The corpus is small and offline.** 18 models, no pretrained weights, no
   fp16/int8 artifacts. The false-positive claim is only as strong as that
   corpus, and the transformer arch was added precisely because MLP-only
   calibration was flattering.
4. **Fusion is calibrated and tier-gated, not learned.** The plan's
   logistic-regression fusion on disjoint splits is not built.

## Install

    pip install -e .            # needs numpy, safetensors, scipy
    pip install -e ".[torch]"   # optional, for loading .pt/.pth and building demo models

## Use

    # scan a model
    modelsentry scan model.safetensors

    # scan with an HTML report, and use a clean-corpus baseline
    modelsentry scan model.pt --baseline baseline.json --report out.html

    # fit a baseline from a folder of known-clean models
    modelsentry baseline ./clean_models -o baseline.json

Exit code is 2 when risk is above 50, so it drops straight into CI as a gate.

## Layout

    modelsentry/
      core/        loaders, TensorGraph, evidence contract, baseline, scanner+fusion
      detectors/   bitplane (informational), recovery, distribution, spectral, serialization
      recovery/    bit-plane reconstruction, windowed structure scan, payload analysis
      report/      self-contained HTML report
    attacks/       red-team: LSB, EvilModel, MaleficNet, seen/holdout/encrypted payloads
    bench/         clean-model generator (MLP, CNN, transformer), benchmark harness
    tests/         smoke and correctness tests, incl. the held-out-marker test
    demo.py        clean model vs seen / held-out / encrypted infected look-alikes

## Benchmark

    python bench/build_models.py data/clean 3
    python bench/run_benchmark.py                 # detection tracks
    python bench/run_arena.py                     # capacity-stealth frontier
    python bench/b4_variance.py data 1000,2000,3000,4000,5000

Every payload is seeded, so a run reproduces exactly; `MODELSENTRY_SEED` picks
the run. The numbers above are `MODELSENTRY_SEED=1000`. This matters because the
encrypted and B4 tracks are driven by random payload content, and quoting one
unreproducible draw of those is how a benchmark flatters itself.

## Safety note

This is a defensive tool. The red-team payloads under `attacks/` are inert
stand-ins (a harmless DOS header string, a hand-built WASM section table, random
bytes) that carry the same statistical and structural fingerprints a real payload
would, without being runnable malware.
