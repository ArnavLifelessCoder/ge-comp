# ModelSentry
## Technical Plan v2 (revised after review)

**One line:** ModelSentry statically analyzes AI model artifacts for evidence of steganographic manipulation and embedded payloads, combining serialization inspection, statistical steganalysis, localization and payload reconstruction into an explainable pre-deployment risk score.

Timeline: 7 days. Scope deliberately cut from v1, see section 10.

### Changelog from v1
| # | Change | Why |
|---|---|---|
| 1 | Payload subsystem split into raw-byte carving and bit-plane reconstruction | v1 assumed payload bytes are visible in raw floats. False for X-LSB. |
| 2 | Entropy cliff is now baseline-relative, not a shape assumption | Bit statistics are architecture, layer and training dependent |
| 3 | Contiguity demoted from assumption to evidence, split into localized and distributed regimes | Contiguity is false for spread spectrum and for any adaptive attacker |
| 4 | EvilModel clamp range relabelled as a signature, separated from the anomaly score | v1 was fingerprinting one paper's implementation detail |
| 5 | "Nobody has shipped" reframed as product positioning, not a research claim | Not true in the broad research sense |
| 6 | New: evidence tier model E1 to E4, separating hidden data from executing malware | A PE header in a tensor does not execute on its own |
| 7 | New: randomly perturbed benign model as a mandatory hard negative | Otherwise the classifier learns "modified equals malicious" |
| 8 | Architecture-disjoint and attack-disjoint splits | Prevents leakage in the fusion model |
| 9 | Day 1 now ships a thin end to end slice with a real number | v1 built attacks on day 1 with no detector to measure against |
| 10 | Behavioral backdoors, API, GitHub Action and UI moved to stretch | Feasibility |

---

## 1. What the PS actually constrains

Three lines in the PS do most of the work:

1. **"without access to the original training pipeline."** No training data, no trusted reference weights, no retraining. Every "diff against the original" approach is dead.
2. **"anomaly detection, not reverse engineering."** They want flagging with evidence, not malware analysis.
3. **"Assign a Model Risk Score"** and **"indicate which layers, weights or characteristics appear suspicious."** Localization and explanation are graded requirements, not polish.

---

## 2. Threat model

**Attacker.** Publishes a model artifact to a public hub, a private registry, or an internal MLOps pipeline. Controls weights, metadata and the serialization container. Cannot run code on the defender machine before the scan. Cannot see the scanner.

**Defender.** Has the artifact only. No training data, no reference copy, no guaranteed GPU. Scan runs pre-deployment, in minutes.

### 2.1 The point that must be stated clearly: hidden data is not executing malware

A Windows PE header sitting inside `layer4.conv2.weight` does not execute when someone calls `load_state_dict`. Tensors are numbers. For hidden bytes to become malware there must be a mechanism:

- a **second stage extractor**, either shipped in the same artifact (a pickle stub, a custom loading script, a `trust_remote_code` module) or already resident on the target host,
- or a **deserialization vulnerability** that makes the model file itself an execution vector,
- or a **framework or application bug** reachable during load or inference.

So ModelSentry never claims "this model executes malware." It claims: **this artifact contains statistically anomalous encoded data consistent with a hidden payload, and therefore presents elevated supply chain risk.** Where an extractor is also present, it says so, loudly, and that is a different and much higher severity finding.

This is also why serialization scanning and weight steganalysis belong in one tool rather than two. **The realistic full attack chain is a tiny pickle loader that reads the payload back out of the mantissa bits at load time.** Each half looks unremarkable alone. Together they are a working dropper. No existing tool sees both halves.

### 2.2 Attack families

| Family | Technique | Payload lives in | Recoverable by |
|---|---|---|---|
| A | Serialization: pickle `__reduce__`, Keras Lambda, TF ops, joblib | The container | Opcode scan |
| B1 | X-LSB substitution: overwrite last X mantissa bits | Bits scattered across many floats | **Bit-plane reconstruction** |
| B2 | Sign and value mapping (StegoNet) | Sign bits or mapped values | Bit-plane reconstruction, variant |
| B3 | Whole-neuron replacement (EvilModel) | Raw bytes of replaced floats | **Raw byte carving** |
| B4 | Spread spectrum with ECC (MaleficNet) | Low amplitude across the layer | Not recoverable without the PN sequence, detection only |
| C | Behavioral backdoor, no payload bytes | Nowhere, it is in the function | Out of scope, see section 10 |

Reference points from the literature, treated as approximate and to be re-measured on our own benchmark, not quoted as guarantees: EvilModel reported 36.9MB embedded in a 178MB AlexNet at under 1 percent accuracy loss with no detections across 58 VirusTotal engines. Model X-Ray reported roughly 80 percent out of distribution accuracy on MaleficNet in one evaluation regime.

**Explicitly out of scope**, and say so on the slide: we do not execute untrusted models by default, we do not decrypt or fully analyze recovered payloads, and we make no claim against an adaptive attacker who has read our detector source.

---

## 3. Why detection is possible

A trained float32 weight is not a random 32 bit string. It is sign, 8 bit exponent, 23 bit mantissa, produced by gradient descent. Its mantissa bit statistics are not uniform, and not independent of the exponent or of neighbouring weights. Compressed or encrypted payload bits are close to perfectly uniform and independent of everything.

Four exploitable consequences. Note that all four are now stated as **baseline-relative deviations**, not as assumed properties of clean models.

**3.1 Bit-plane entropy deviation.** Compute entropy of each mantissa bit plane, bit 0 to bit 22. Compare the resulting curve to the empirical distribution of curves from a clean corpus of the same dtype and layer type. X-LSB injection produces a characteristic step where bits 0 to X-1 saturate and bit X onward returns to baseline. We detect **deviation from the measured baseline**, not departure from an assumed smooth shape, because quantization, pruning, mixed precision, normalization and optimizer choice all move that curve for entirely benign reasons.

**3.2 Exponent-conditioned structure.** In real networks, mantissa behaviour differs between small and large magnitude weights. Injection destroys that dependence. Conditioning the entropy estimate on exponent bucket is cheap and catches injections a flat test misses.

**3.3 Spatial regime, two detectors not one assumption.** Sequential injection produces a contiguous anomalous run in the flattened tensor. Spread spectrum and adaptive attackers deliberately do not. So we run both: a **localized-regime** detector (sliding window, coherent run length) and a **distributed-regime** detector (global residual variance inflation, kurtosis shift, cross-region correlation). Localization matters independently because the PS demands it, so this is evidence and output, never a precondition.

**3.4 Recoverability, the strongest evidence we can produce.** See section 5. Statistics are circumstantial. Reconstructing a bitstream that decodes to a valid file header is close to proof.

**B3 detection**, reformulated. Primary signal is per-neuron distribution distance against the rest of the layer (KS and Wasserstein), plus cross-neuron structure and magnitude clustering. The EvilModel 0.0078 to 0.0313 clamp is kept as a **named signature check**, reported separately from the anomaly score, because a cheap high-precision signature is worth shipping as long as it never carries a verdict alone. Signatures and heuristics side by side is how real security tooling works.

**B4 detection.** Residual energy after a smoothing or low rank fit, variance inflation, kurtosis change. We expect this to be our weakest detector. Report the number we actually measure.

---

## 4. Evidence tiers

The output is not one number. It is a score plus a tier, because "the statistics look odd" and "I pulled an ELF header out of the mantissa bits" are not the same finding and should never collapse into the same 0 to 100 value.

| Tier | Meaning | Example |
|---|---|---|
| **E1** | Statistical anomaly only | Bit-plane entropy deviates 4.2 sigma from baseline |
| **E2** | Structured data recovered | Reconstructed bitstream has 0.91 printable ratio and low compression ratio |
| **E3** | Payload signature recovered | Recovered stream begins `MZ`, or a YARA rule fires |
| **E4** | Active execution vector present | Serialization layer contains a `__reduce__` that reads tensor data |

Risk score is reported per tier. An E1-only finding tops out in the "review" band. E3 or E4 goes red. This directly answers the "hidden data is not malware" problem: the tier *is* the claim.

---

## 5. The payload subsystem, both modes

This is the part v1 got wrong.

**Mode 1, raw byte carving.** For B3 and any attack that writes payload bytes directly into float storage. Take the raw bytes of the suspicious region, scan for magic headers (`MZ`, `\x7fELF`, Mach-O, `PK\x03\x04`), base64 blobs, URLs, shell strings, YARA rules, and compression ratio.

**Mode 2, bit-plane reconstruction.** For B1 and B2. The payload is one bit per weight, so the raw bytes contain nothing. Harvest bit *b* from each weight in traversal order and repack into bytes.

The search space, spelled out because this is the actual engineering:

- bit depth X in 1..8
- bit ordering within the byte, MSB first and LSB first
- traversal order, stored tensor order, plus row-major and column-major for 2D

That is roughly 32 to 48 candidate reconstructions per tensor, which is cheap. Score each candidate on magic header hit, printable string density, and compression ratio (a real payload compresses badly if encrypted, well if plaintext, but either way differs sharply from repacked noise). Keep the best candidate and report the exact parameters that recovered it, since "recovered at X=4, MSB first, stored order, offset 12288" is itself a piece of the explanation.

For B4 we do not attempt recovery. Without the PN sequence it is not recoverable, and pretending otherwise would be dishonest.

---

## 6. Architecture

```
modelsentry/
  core/
    loaders/          safetensors, .pt/.pth, .bin, .h5, .npz   (ONNX, GGUF stretch)
                      -> TensorGraph(name, dtype, shape, raw_bytes, role)
    detectors/
      serialization.py   pickle opcode scan, no unpickling; Keras Lambda / TF op scan
      bitplane.py        per-plane entropy vs baseline, exponent-conditioned
      spatial.py         localized-regime and distributed-regime detectors
      distribution.py    per-neuron KS / Wasserstein, magnitude clustering
      signatures.py      named attack fingerprints, reported separately
      spectral.py        residual energy, kurtosis, spread-spectrum indicators
    recovery/
      carve.py           mode 1, raw byte carving
      reconstruct.py     mode 2, bit-plane reconstruction and candidate search
      analyze.py         magic headers, YARA, strings, entropy, compression ratio
    baseline.py        clean-corpus null distributions, raw stat -> p-value -> z
    fusion.py          evidence -> tier + Model Risk Score
    explain.py         evidence -> readable narrative
  report/              HTML: entropy curves vs baseline band, localization heatmap, hexdump
  attacks/             RED TEAM: lsb, signmap, evilmodel, maleficnet, pickle_extractor
  bench/               dataset builder, architecture-disjoint splits, ROC harness
  cli.py
```

**Detector contract.** Every detector returns evidence, never a verdict:

```python
@dataclass
class DetectionEvidence:
    detector: str
    tensor: str
    score: float            # calibrated z or p
    confidence: float
    location: Optional[Slice]   # for localization + report heatmap
    features: dict[str, float]  # everything, for audit
    explanation: str
    tier_hint: Literal["E1","E2","E3","E4"]
```

Fusion consumes a list of these. Report generation becomes nearly trivial, and every alert is auditable back to the number that caused it.

**Never deserialize to inspect.** Pickle opcodes and safetensors headers are parsed directly. A scanner that must load the model to scan it is itself the attack surface.

---

## 7. Scoring and calibration

1. Clean corpus, 50 to 100 real models across CNN and transformer, float32 and float16.
2. Fit null distributions per detector, per dtype, per layer type.
3. At scan time convert each raw statistic to a p-value then a z-score, so all detectors speak one language.
4. Fuse with a small logistic regression into a 0 to 100 score, bounded by evidence tier.
5. **Report FPR at the operating threshold.** A scanner without an FPR number is not a scanner.

**Leakage control, mandatory.** Architecture-disjoint train and test split: no architecture appears in both. Attack-parameter-disjoint too, so we train on X in {1,3,12} and test on X in {6,23}. Otherwise the AUC is fiction.

---

## 8. Evaluation

Hard negatives are worth more than positives here. If we flag a legitimately quantized checkpoint the tool is useless in production.

**Positives:** B1 at X in {1,3,6,12,23} across payload types {random, zip, PE stub, base64 script} and targets {last FC, all conv, single layer}; B2 sign and value mapping; B3 neuron replacement at 10/25/50 percent; B4 spread spectrum at two amplitudes; A pickle variants; **and the combined chain, pickle extractor plus LSB payload.**

**Hard negatives:**
- quantized int8, pruned 50 percent, LoRA merged, distilled
- float16 and bfloat16 checkpoints
- weight normalized or rescaled models
- same architecture at different training seeds and different checkpoints
- **randomly perturbed but benign model**, the most important one, without it the detector just learns that modified weights are malicious

**Report:** ROC AUC by family and by embedding rate, detection rate at 1 percent FPR, per-negative-class false positive counts, recovery success rate for B1 and B3, and scan time by model size.

---

## 9. Seven day plan

Every day ends with the full benchmark run. Debug as you build, do not stack five days of untested detectors.

| Day | Work | Done when |
|---|---|---|
| 1 | Repo, loaders, TensorGraph, LSB attack, 10 clean + 10 infected + 5 hard negatives, one crude entropy detector, ROC harness | **A first AUC number exists, end to end** |
| 2 | Bit-plane entropy vs baseline, exponent-conditioned, spatial regimes, baseline fitting | AUC on B1 measurably beats day 1 |
| 3 | Recovery subsystem, both modes, plus serialization scanner | The demo prints an `MZ` header reconstructed out of mantissa bits |
| 4 | B3 neuron detector plus signatures, B2, matching attacks, build the combined pickle-extractor chain | All shipped families have a matching detector and a matching attack |
| 5 | Full clean corpus, disjoint splits, fusion, tiers, risk score, all hard negatives | ROC tables and an FPR number exist, quantized models pass |
| 6 | HTML report, entropy curve with baseline band, localization heatmap, hexdump, CLI polish | A non-technical person can read the output |
| 7 | Demo script, README, threat model doc, limitations, two full dry runs | Nothing is discovered live on stage |

---

## 10. Scope, cut deliberately

**Shipping:** serialization scan (A), B1, B3, plus B2 and B4 as detection-only if days allow. CLI plus HTML report. Calibration, tiers, recovery, hard negatives.

**Cut to stretch:** behavioral backdoor detection (family C, a research problem not a week), FastAPI service, GitHub Action, Streamlit UI, ONNX and GGUF loaders.

The reason for the cut, stated plainly: **three detectors that produce scientifically credible evidence, reconstruct an actual hidden payload, quantify false positives on nasty benign models and explain every alert, beat fifteen detectors that do none of that.** Scope is the most likely thing to kill this project.

---

## 11. Demo, about 7 minutes

1. Two safetensors files, same architecture, accuracy equal to three decimals, 0.2 percent of weights differ. Ask which is malicious.
2. `modelsentry scan clean.safetensors` gives 4/100, tier none, green.
3. `modelsentry scan vendor_model.safetensors` gives 91/100, **tier E3**, red. Names the layer, shows the entropy curve against the clean baseline band, shows the localization heatmap, then prints the reconstructed hexdump with `MZ` and `This program cannot be run in DOS mode`, along with the recovery parameters that found it.
4. **The chain.** Show that the same file also carries a pickle stub whose `__reduce__` reads that exact tensor. Tier E4. Neither half is alarming alone. Together they are a dropper.
5. Open the HTML report, walk the explanation.
6. ROC table by family, and the hard negative table showing quantized and randomly perturbed benign models correctly passing.
7. Limitations, stated honestly.

---

## 12. Positioning

Not "nobody has done this." The accurate and still strong version:

> Existing research demonstrates individual steganalysis techniques, and existing tools cover serialization attacks. ModelSentry integrates multiple attack-family detectors, static serialization inspection, localization, payload reconstruction, calibrated risk scoring and deployment-oriented reporting into a single pre-deployment scanner, and it is the only one that sees both halves of a combined extractor-plus-payload attack.

## 13. When they ask why this is hard

Because the model still works. Accuracy is unchanged, architecture is unchanged, the file loads normally, and 58 antivirus engines saw nothing. The payload is not in the file format, it is in the numbers, and the numbers are supposed to look like noise. The only thing separating trained noise from encrypted payload noise is a statistical fingerprint, and if you find it you still have to prove it by pulling the payload back out.
