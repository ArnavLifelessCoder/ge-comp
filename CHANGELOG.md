# Changelog

All notable changes to ModelSentry are documented here.

---

## [91d37d2] — 2026-08-19

### Kill detector circularity, add held-out/encrypted benchmark tracks

The headline change: the detector's primary signal was circular — it grepped for
magic bytes (`MZ`, `\x7fELF`, `subprocess`, etc.) while the red-team payloads
were built from exactly those constants. AUC 1.000 was string search against a
planted answer key. This release replaces the signal, rebuilds the benchmark to
measure the right thing, and re-runs everything from scratch.

#### Detector — marker-agnostic primary signal

- **`modelsentry/recovery/analyze.py` rewritten.** The primary signal is now a
  windowed chi-square test: split each recovered bitstream into 1024-byte
  windows, score each window's non-uniformity against the other windows of the
  same stream (self-calibrated null), and confirm via a compression ratio check.
  This finds payload structure without knowing the format.
- **Magic bytes/strings demoted to corroboration only.** They can lift the
  evidence tier from E2 ("structured data recovered") to E3 ("we know what this
  is"), but they can never be the sole reason a model is flagged.
- **Multiple-comparison correction** (`z_floor_for(n)`): raises the significance
  bar by `0.7 · log₁₀(n_candidates)`. Without this, false positives scale with
  model size because candidate count scales with tensor count.
- **Full-capacity evasion closed.** A payload that fills the entire LSB capacity
  makes every window look alike, so the relative null goes blind. Added an
  absolute uniform-null test (`global_z`) to catch this case.
- **Raw-byte carving gated off uniform null.** Raw float bytes are never uniform
  (exponent bias, sign-bit skew), so running the structure scan on them without
  the gate flagged every clean model at risk 92. The gate prevents this.
- **`modelsentry/detectors/recovery_detector.py`** now counts hypotheses up
  front and passes the count to `analyze`, and reports the payload byte offset
  (localization) plus hex dump at that offset.

#### Benchmark — held-out and encrypted tracks

- **`attacks/payloads.py`**: added `HOLDOUT_KINDS` (wasm, sqlite, pem,
  yaml_config, packed — zero marker overlap with the detector's lists) and
  `ENCRYPTED_KINDS` (uniform-random, keystream-XOR'd PE).
- **`bench/run_benchmark.py`**: rewritten to report by marker regime (seen /
  held-out / encrypted), with encrypted inside the overall AUC instead of
  excluded. Dead `fam_tracks`, the no-op `"random" not in nm` filter, and the
  lexicographic tier compare are all removed.
- **`bench/build_models.py`**: added `TinyTransformer` (embedding tables,
  LayerNorm, multi-head attention). Corpus is now 6 architectures / 18 models.
- **`bench/run_arena.py`** (new): produces the arena dashboard's figures
  reproducibly, replacing the old hardcoded counts.

#### Tests — 5 → 10, all passing

| Test | What it checks |
|---|---|
| `test_clean_scores_low` | clean model risk stays in the safe band |
| `test_structured_payload_detected` | B1 LSB with a known-format payload is flagged |
| `test_roundtrip_recovery` | injected bits can be recovered intact |
| `test_holdout_payload_detected_without_markers` | held-out payloads contain none of the detector's known markers **and** are still detected |
| `test_capacity_saturating_payload_detected` | full-capacity payload (the evasion that was closed) is caught |
| `test_structure_scan_quiet_on_uniform_noise` | uniform-random bytes do not trigger the structure scan |
| `test_z_floor_rises_with_hypothesis_count` | significance bar grows with candidate count |
| `test_encrypted_lsb_is_reported_as_missed_not_hidden` | encrypted LSB is honest about being undetectable |
| `test_malicious_pickle_flagged` | serialization detector catches pickle reduce → E4 |
| `test_evilmodel_encrypted_flagged` | EvilModel with encrypted payload is still caught (placement anomaly) |

#### Results (post-saturation-fix benchmark run)

522 artifacts scored (504 attacks, 18 negatives).

| Track | ROC AUC | Red-rate | Mean risk | n |
|---|---|---|---|---|
| **B1 LSB, held-out markers** | **1.000** | **1.00** | **96.1** | **180** |
| B1 LSB, seen markers | 1.000 | 1.00 | 98.5 | 144 |
| B1 LSB, encrypted | 0.403 | 0.00 | 7.8 | 72 |
| B3 EvilModel, seen | 1.000 | 1.00 | 100.0 | 27 |
| B3 EvilModel, held-out | 1.000 | 1.00 | 92.3 | 27 |
| B3 EvilModel, encrypted | 1.000 | 1.00 | 92.3 | 27 |
| B4 MaleficNet | 0.381 | 0.00 | 8.0 | 27 |

```
ROC AUC, all positives including encrypted:  0.915
clean + hard-negative risk:                  mean 7.3, max 24
clean RED-rate (CI-gate false positives):    0.00
malicious pickle:                            risk 100, tier E4
```

Per-payload detail (X-LSB, mean risk over 4 embedding depths):

```
pe          100.0   seen        wasm          96.2   holdout
elf          98.0   seen        sqlite        97.3   holdout
script       98.0   seen        pem           96.8   holdout
zip          98.0   seen        yaml_config   95.2   holdout
                                packed        94.9   holdout
random        7.7   encrypted   encrypted_pe   7.9   encrypted
```

#### Arena results

198 attacks generated and scored by `bench/run_arena.py`:

| Metric | Count |
|---|---|
| Attacks generated & scored | 198 |
| Caught at the CI gate (risk ≥ 51) | 165 |
| Stealthy + usable weapon (danger corner) | **0** |
| Stealthy inert-data smuggles (disclosed residual) | 33 |

The danger corner remains empty: to evade the CI gate, the attacker must strip
the execution vector, leaving inert data that needs an out-of-band loader the
defender never sees.

#### Housekeeping

- `.gitignore`: `data/`, `results.json`, `arena.json` excluded (generated, not
  committed). Rebuild with `python bench/build_models.py data/clean 3` then
  `python bench/run_benchmark.py`.
- `arena_dashboard.html`: KPI figures updated from 140/0/90/50 to 198/0/165/33;
  subtitle now references `bench/run_arena.py`.
- `README.md` and `STATUS.md`: all benchmark numbers refreshed.
- No `Co-Authored-By` trailer on the commit.

---

## [267dff0] — 2026-08-11

### Add kill chain page, revise arena dashboard copy and figures

- Added `killchain.html`: a self-contained visual showing the attacker's kill
  chain (payload → embedding → container → extraction → execution) and where
  ModelSentry intercepts each stage.
- Revised `arena_dashboard.html` copy and SVG scatter for clarity.

---

## [35f8bed] — 2026-08-10

### Add ModelSentry Arena dashboard

- Added `arena_dashboard.html`: interactive capacity-stealth frontier
  visualization. Shows the trade-off between detection risk and payload
  usability, proving the danger corner is empty.

---

## [0a56cdf] — 2026-08-10

### Initial commit

- Full ModelSentry scanner: loaders (safetensors, npz, pytorch), TensorGraph,
  Evidence contract, central fusion, CLI (`scan` + `baseline`), HTML report.
- Detectors: bitplane (informational), recovery (bit-plane reconstruction +
  raw-byte carving), distribution (per-neuron outliers), spectral
  (informational), serialization (pickle opcode scan).
- Attack implementations: LSB steganography, EvilModel neuron replacement,
  MaleficNet spread-spectrum.
- Benchmark harness and clean-model generator (MLP, CNN).
- Demo script, `pyproject.toml`, initial tests.
