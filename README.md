# ModelSentry

Pre-deployment scanner for steganographic malware hidden in AI model weights.

You hand it a model file, it hands back a risk score (0-100), an evidence tier,
the exact tensor and bits that look wrong, and where possible the payload it
pulled back out. It is meant to run as a gate before a model reaches production,
the same way you would scan a container image before deploying it.

## Why this is hard

A malicious model still works. Accuracy is unchanged, the architecture is
unchanged, the file loads normally, and public malware scanners see nothing,
because the payload is not in the file format, it is in the numbers, and the
numbers are supposed to look like noise. The published EvilModel attack hid
36.9 MB inside a 178 MB network with under 1 percent accuracy loss and was
missed by all 58 engines on VirusTotal.

## What it detects

ModelSentry covers three attack families in one pass:

- Serialization payloads (family A). A pickle, Keras Lambda, or similar that
  runs code at load time. Detected by walking the pickle opcodes without ever
  unpickling the file.
- Weight steganography (families B1 to B3). Payload bytes hidden in the low
  mantissa bits (LSB), sign or value mapping, or whole replaced neurons
  (EvilModel).
- The combined attack. A small pickle stub that reads an LSB payload back out
  of the weights at load time. Neither half looks alarming alone. ModelSentry is
  built to see both.

## Results so far

On a small corpus with an architecture-disjoint split (baseline fit on
architectures absent from the test set) and randomly-perturbed-benign hard
negatives:

    Family                  ROC AUC   red-rate
    B1 LSB (structured)       1.000     1.00
    B3 EvilModel              1.000     1.00
    B4 MaleficNet             0.454     0.00   (hard case)

Clean and hard-negative models max out at risk 16, and the CI-gate false-positive
rate (clean models scored red) is 0. A malicious pickle scores 100 at tier E4.
B4 spread-spectrum is the honest hard case and is reported as informational only,
so it never creates a false positive. See STATUS.md for the full picture.

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

## The honest part

On fully trained float32 models the entire mantissa is already close to maximum
entropy, so simple bit-plane entropy cannot separate an LSB payload from natural
weights. That surprised us on day one and it matches the research literature.
The signal we rely on instead is recoverability: if the low bits decode to a
structured stream (a header, printable text, a plausible length frame) that is
close to proof. This means ModelSentry is very strong against realistic payloads
that carry headers or strings, and weakest against a payload that is pure
encrypted noise with no framing, which is the honest hard case for any static
scanner. We report that case separately rather than hiding it.

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
      detectors/   bitplane (informational), recovery, serialization
      recovery/    bit-plane reconstruction and payload/stream analysis
      report/      self-contained HTML report
    attacks/       red-team: LSB injection, inert payload stand-ins
    bench/         clean-model generator, benchmark harness with ROC/AUC
    tests/         smoke and correctness tests
    demo.py        builds a clean model and an infected look-alike, scans both

## Benchmark

`python bench/build_models.py data/clean 3` then `python bench/run_benchmark.py`.
The harness uses an architecture-disjoint split (the baseline is fit on
architectures that do not appear in the test set) and includes randomly perturbed
but benign models as hard negatives, so the detector has to key on payload
structure rather than just "these weights were modified".

## Safety note

This is a defensive tool. The red-team payloads under `attacks/` are inert
stand-ins (a harmless DOS header string, random bytes) that carry the same
statistical and structural fingerprints a real payload would, without being
runnable malware.
