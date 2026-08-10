# Detection of Steganographic Malware Hidden in AI Model Weights
## Technical Plan

Working name: **ModelSentry** (alternatives: WeightWatch, StegoScope, Aegis-ML)

Timeline assumed: 7 days. Target: a real pre-deployment security scanner, not a demo script.

---

## 1. Read of the problem statement

The PS is asking for a **pre-deployment scanner for AI model artifacts**. Three things in it are easy to misread, and getting them right is most of the score:

1. It says "without access to the original training pipeline." That means no training data, no clean reference copy of the model, no ability to retrain. Any detector that needs "compare against the original" is out. This is the hardest constraint and it kills most naive approaches.

2. It says "anomaly detection, not reverse engineering." They do not want us decompiling payloads or building a malware analysis lab. They want statistical and ML-based flagging with explanations.

3. It says "Assign a Model Risk Score" and "explain why a model is flagged." Explainability is a named criterion, weighted equally with detection. A black box classifier that outputs 0.94 will lose to a system that says "layer4.1.conv2.weight, bits 0 through 5, entropy 0.998 against a clean-model baseline of 0.71, contiguous run of 48KB, carved bytes begin with an MZ header."

So the product is: **give it a model file, get back a risk score, a list of suspicious tensors, and a readable reason for each.**

---

## 2. Threat model (write this up front, judges score it directly)

**Attacker.** Can produce and publish a model artifact. Uploads it to a public hub, a private registry, or slips it into an internal MLOps pipeline. Can modify weights, metadata, and the serialization container. Cannot execute code on the defender machine before the scan runs. Cannot observe the scanner.

**Defender (us).** Has only the artifact: weight file plus metadata. No training data. No trusted reference weights. No GPU guaranteed. Scan must finish in minutes and run in CI as a gate before the model reaches production.

**Attacker goals we defend against.**

| Family | What it is | Existing coverage |
|---|---|---|
| A. Serialization payload | Code runs at load time: pickle `__reduce__`, Keras Lambda layers, TF SavedModel ops, joblib, numpy `allow_pickle` | Covered by modelscan / picklescan. We include it for completeness. |
| B. Steganographic payload in weights | Malware bytes hidden inside the parameter values themselves | **Almost nothing covers this. This is our differentiator.** |
| C. Behavioral backdoor | No hidden bytes, but the model misbehaves on a trigger input | Partially covered by research tools, none practical without data |

Family B splits into four techniques we must handle separately, because they leave completely different fingerprints:

- **B1. LSB substitution (X-LSB).** Overwrite the last X mantissa bits of each float32. Cheap, huge capacity, invisible to accuracy at low X.
- **B2. Sign and value mapping (StegoNet).** Encode bits in sign bits or by mapping payload bits onto chosen parameter values.
- **B3. Whole-neuron replacement (EvilModel).** Replace entire neurons in the fully connected layers near the output with crafted floats. The original paper keeps values inside 0.0078 to 0.0313 so magnitudes look plausible, and hides 36.9MB inside a 178MB AlexNet at 1 percent accuracy loss. VirusTotal caught nothing across 58 engines.
- **B4. Spread spectrum (MaleficNet).** Payload spread across many parameters at low amplitude with error correcting codes. Survives fine tuning, pruning and noise. This is the hard one and we should be honest about our numbers on it.

**Explicitly out of scope** (say this in the deck, it reads as maturity, not weakness): we do not execute untrusted models by default, we do not attempt full payload reconstruction and decryption, and we do not claim detection of a nation state adversary who has read our detector source and optimized against it.

---

## 3. Why detection is possible at all

This is the core scientific claim and it needs to be stated cleanly.

A trained float32 weight is not a random 32 bit string. It is a sign bit, an 8 bit exponent, and a 23 bit mantissa produced by gradient descent. The bit statistics of that mantissa are **not uniform**, and more importantly they are **not independent of the exponent** and **not independent of neighbouring weights**.

Hidden payload bytes are the opposite. Malware is usually compressed or encrypted, so its bits are near perfectly uniform, independent of magnitude, and independent of position.

Four exploitable consequences:

**3.1 The entropy cliff.** Compute Shannon entropy of each mantissa bit plane, bit 0 through bit 22, across a tensor. In a clean model this rises smoothly with bit index and never quite saturates. In an X-LSB infected model, bits 0 through X-1 sit at exactly 1.0 and bit X onward drops back to the natural curve. That step is a sharp, visual, explainable signature. It is also the single best demo asset in the whole project: two line plots side by side and a non-technical judge instantly understands.

**3.2 Contiguity.** A payload is a byte stream, so it occupies a **contiguous run** in the flattened tensor. Natural weight anomalies are scattered. So we do not just measure "is this tensor weird," we run a sliding window and look for a long unbroken anomalous run. This massively cuts false positives and it also gives us localization for free, which feeds the explanation.

**3.3 Exponent-conditioned structure.** In a real network, small magnitude weights and large magnitude weights have different mantissa behaviour. Injection destroys that relationship. Conditioning the entropy estimate on the exponent bucket is a cheap and strong second signal, and it catches injections that a flat entropy test misses.

**3.4 Payload bytes are still payload bytes.** Once we have localized a suspicious region, we can **carve the candidate bytes out and inspect them as bytes**. PE `MZ` headers, ELF `\x7fELF`, Mach-O magic, ZIP `PK\x03\x04`, base64 blobs, URLs, shell command strings, YARA rule hits. This converts the project from "a statistics report" into "a security tool," and it is the moment in the demo where the room sits up. Not "this tensor is unusual" but "I pulled 48KB out of layer4.1.conv2.weight and it starts with a Windows executable header."

For B3 the signal is different: replaced neurons have a per-neuron value distribution that does not match the rest of the layer. A per-neuron Kolmogorov-Smirnov or Wasserstein distance against the layer distribution finds them directly, and the crafted 0.0078 to 0.0313 clamping is an obvious tell.

For B4 the signal is different again: spread spectrum raises the residual noise floor uniformly. We take the weight residual after a smoothing or low rank fit, and look at variance inflation and kurtosis change versus the clean baseline. We should expect roughly 80 percent detection here based on published results, and we should say so rather than overclaim.

---

## 4. What already exists, and where the gap is

| Tool / paper | Covers | Gap |
|---|---|---|
| modelscan (Protect AI) | Serialization attacks only, static byte scan for unsafe opcodes | Blind to anything in the weight values |
| picklescan | Pickle opcodes | Same |
| Hugging Face scanner | Pickle plus known signatures | Same |
| Model X-Ray (2024) | LSB detection via few shot learning on byte plane images, 85 to 100 percent on 8 to 18 LSB, about 80 percent on MaleficNet | Research code, CNN and float32 only, no payload carving, no risk score, no explanation layer, no product |
| Steganalysis of AI Models LSB Attacks | Supervised classifier, needs about 40k training models | Not practical to deploy |
| NeuPerm (2025) | Mitigation by permuting neurons | Mitigation, not detection, and it does not tell you a model was infected |

**The gap we fill:** nobody has shipped a single tool that covers serialization plus weight steganography plus payload carving, calibrated against a clean baseline, producing a scored and explained report that can run as a CI gate. That sentence is the pitch.

---

## 5. System architecture

```
modelsentry/
  core/
    loaders/          safetensors, .pt/.pth, .bin, ONNX, .h5, GGUF, .npz
                      -> unified TensorGraph (name, dtype, shape, raw bytes, role)
    detectors/
      serialization.py   pickle opcode scan, no unpickling, plus Keras Lambda / TF op scan
      bitplane.py        entropy vs bit index, entropy cliff, exponent-conditioned entropy
      contiguity.py      sliding window anomaly runs, localization map
      distribution.py    per-tensor and per-neuron KS / Wasserstein, magnitude clamping test
      spectral.py        residual energy, kurtosis, FFT/DCT for spread spectrum
      payload.py         byte carving, magic headers, strings, base64, YARA, compression ratio
      behavioral.py      optional, needs a runnable model: activation entropy, STRIP-style check
    baseline.py        clean-corpus null distributions, raw stat -> p-value -> z-score
    fusion.py          calibrated fusion -> Model Risk Score 0 to 100
    explain.py         detector evidence -> readable narrative
  report/              HTML report: bit plane images, entropy curves, heatmap, carved hexdump
  attacks/             RED TEAM: lsb.py, evilmodel.py, maleficnet.py, pickle_payload.py
  bench/               dataset builder, ROC/AUC harness, per-attack tables
  cli.py               modelsentry scan <path>
  api.py               optional FastAPI, and a GitHub Action wrapper
```

Two design decisions worth defending out loud:

**Never deserialize to inspect.** We parse pickle opcodes and safetensors headers directly. A scanner that has to load the model to scan it is itself the attack surface. This is a one line point in the demo that lands very well with security judges.

**Detectors are independent and each returns evidence, not a verdict.** Fusion happens once, at the end, with calibrated weights. This is what makes the explanation honest and makes adding a detector cheap.

---

## 6. Scoring and calibration

Hand tuned thresholds are the single most common way a project like this looks amateur. Do it properly:

1. Build a clean corpus of 50 to 100 real models (torchvision resnet18/34/50, mobilenet, vgg11, efficientnet, plus a few small HF transformers).
2. Run every detector over the clean corpus, fit a null distribution per detector per layer type.
3. At scan time, convert each raw statistic to a p-value and then a z-score against that null. Now every detector speaks the same language.
4. Fuse with a small logistic regression trained on clean versus synthetically attacked models. Output a 0 to 100 Model Risk Score with bands: 0-20 clean, 21-50 review, 51-80 suspicious, 81-100 likely compromised.
5. **Report the false positive rate at the chosen threshold.** A scanner without an FPR number is not a scanner.

**Hard negatives matter more than positives.** Quantized models, pruned models, LoRA merged models, distilled models and models trained with different optimizers all have unusual bit statistics for entirely innocent reasons. If we flag a legitimately quantized checkpoint we are useless in production. A slide showing "we correctly pass all 12 quantized and pruned models" is worth more than another point of AUC.

---

## 7. Evaluation

Build the attack module **first**, on day one. Without ground truth there is no science, only vibes.

Dataset matrix:
- Clean: 60 models across CNN and transformer, float32 and float16.
- Hard negatives: quantized int8, pruned 50 percent, LoRA merged, distilled.
- B1 LSB: embedding at 1, 3, 6, 12, 23 bits, payload types random / zip / PE stub / base64 script, targeting last FC only versus all conv versus one layer.
- B2 sign and value mapping.
- B3 EvilModel neuron replacement at 10, 25, 50 percent of a layer.
- B4 MaleficNet style spread spectrum at low and high amplitude.
- A. Pickle payload variants.

Report a table of ROC AUC by attack family and by embedding rate, plus detection rate at 1 percent FPR, plus scan time per model size. The interesting and honest cell in that table is low rate MaleficNet, where we will be weakest.

---

## 8. Seven day plan

| Day | Work | Done when |
|---|---|---|
| 1 | Repo, loaders, TensorGraph, red team attack module (LSB + pickle), 20 model mini dataset | You can generate an infected model and load it back |
| 2 | Bit plane and exponent conditioned entropy detectors, contiguity scan, first AUC number | You have a real number, not a feeling |
| 3 | Payload carving, magic headers, YARA, strings, plus serialization scanner | The demo can print an MZ header out of a weight tensor |
| 4 | EvilModel neuron detector, spectral / MaleficNet detector, extend attack module to match | All four B families have a matching detector |
| 5 | Clean corpus, baseline fitting, fusion, risk score, full benchmark harness, hard negatives | ROC tables and an FPR number exist |
| 6 | HTML report with plots and hexdump, CLI polish, optional Streamlit upload UI, GitHub Action | A non technical person can read the output |
| 7 | Demo script, README, threat model doc, limitations, two full dry runs | Nothing is discovered live on stage |

Debug as you go, at the end of each day run the full benchmark. Do not stack five days of untested detectors and find out on day six that the loader mangles float16.

---

## 9. Demo flow, about 7 minutes

1. Two safetensors files. Same architecture, same accuracy to three decimals, 0.2 percent of weights differ. Ask the room which one is malicious.
2. `modelsentry scan clean.safetensors` gives risk 4 out of 100, green.
3. `modelsentry scan vendor_model.safetensors` gives risk 91 out of 100, red. It names the layer, shows the entropy cliff plot next to the clean one, and prints the carved hexdump with `MZ` and `This program cannot be run in DOS mode`.
4. Open the HTML report, walk through the localization heatmap and the plain English explanation.
5. Show the ROC table across all attack families including the hard negatives that we correctly passed.
6. Limitations slide, honestly stated.
7. Close on deployment: the same scanner as a GitHub Action blocking a model merge, and the CI log showing the block.

---

## 10. Risks and how we handle them

**Scope creep into behavioral backdoors.** Family C is a research problem, not a week. Ship a lightweight version at most, and be explicit that it is exploratory.

**Overclaiming on MaleficNet.** Published state of the art is around 80 percent. If our slide says 99 percent, a knowledgeable judge will not believe any of our other numbers either.

**False positives on quantized models.** Test this on day five, not day seven.

**Large model scan time.** Sample tensors rather than scanning every byte of a 70B checkpoint, and report the sampling rate.

**Demo fragility.** Pre-generate all models, pin versions, run offline, rehearse twice.

---

## 11. What to say when they ask "why is this hard"

Because the model still works. Accuracy is unchanged, architecture is unchanged, the file loads normally, and 58 antivirus engines see nothing. The payload is not in the file format, it is in the numbers, and the numbers are supposed to look like noise. The only thing that separates trained noise from encrypted payload noise is a statistical fingerprint, and this tool is built around finding it.
