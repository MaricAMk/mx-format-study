# Findings

A short summary of what the project has built and measured so far. Method,
full tables and reproduction steps are in [`reports/`](reports/) (see
[Where the details are](#where-the-details-are)). This page only states the
results.

---

## In one paragraph

The project has a block-scaled quantization emulator that matches the OCP MX
specification and Microsoft's `microxcaling` library bit for bit, and uses it
to ask one question on Qwen2.5-1.5B: **does the direction in which values are
grouped matter as much as the number format?** For activations it often does:
grouping along tokens instead of the hidden dimension removes about half of
MXFP4's accuracy loss and most of BFP4's, and a control experiment shows this
is not caused by future tokens leaking through the group scale. For weights,
direction matters little. For NVFP4, the way the scale is rounded matters more
than the grouping direction.

---

## 1. The emulator is correct

| Check | Result |
|---|---|
| Every code of E2M1, E2M3, E3M2, E4M3, E5M2, E8M0 (912 codes) decoded from the specification | All values round-trip; midpoints round to even; overflow saturates or becomes NaN as configured |
| Agreement with `microxcaling` (MXFP4, MXINT8/4/2, group sizes 8–128, both axes) | Bit-exact |
| Fast (vectorized) implementation vs. reference implementation | Bit-exact on outputs and all diagnostics, across every scale × element format and rounding mode |
| Test suite | 345 tests, run in CI on every push |

Two errors were found and fixed by these checks along the way: the E8M0 scale
rule (the MX rule subtracts only the exponent of the element maximum), and the
maxima of formats with reserved Inf/NaN codes (E4M3 is 448, not 480).

---

## 2. Grouping direction (experiment 2)

**Setup.** Qwen2.5-1.5B in FP32, WikiText-2 test, 8 windows of 2048 tokens.
Either the weights or the inputs of all linear layers are quantized (never
both). FP32 perplexity is **8.098**. Numbers below are the *increase* in
perplexity; lower is better. Differences under about **0.02** are noise.

| Format | Group | Scale | Elements |
|---|---|---|---|
| MXFP4 | 32 | E8M0 | E2M1 |
| BFP4 | 32 | E8M0 | 4-bit integer |
| NVFP4 | 16 | E4M3 (FLOOR) | E2M1 |
| NVFP4-nearest | 16 | E4M3 (round to nearest) | E2M1 |
| BFP8 | 32 | E8M0 | 8-bit integer |

### Activations: grouping along tokens helps, a lot for some formats

| Format | Along hidden (standard) | Along tokens | Change |
|---|---|---|---|
| MXFP4 | +1.86 | +1.00 | **−46 %** |
| BFP4 | +2.95 | +1.16 | **−61 %** |
| NVFP4 | +0.63 | +0.54 | −14 % (within noise, see below) |
| NVFP4-nearest | +0.54 | +0.62 | +16 % (worse) |
| BFP8 | +0.02 | −0.00 | no effect |

With token grouping, the simple BFP4 format comes within 0.16 of MXFP4 (the
gap is 1.08 with standard grouping). This fits the known pattern of
activation outliers sitting in a few channels: token groups keep each channel
separate, hidden groups mix an outlier channel with ordinary ones.

### Weights: direction barely matters

| Format | Along input (standard) | Along output channels |
|---|---|---|
| MXFP4 | +1.03 | +1.01 |
| BFP4 | +1.70 | +1.49 |
| NVFP4 | +1.34 | +2.18 |
| NVFP4-nearest | +0.64 | +0.75 |
| BFP8 | +0.01 | +0.01 |

The large NVFP4 penalty disappears almost entirely once its scale is rounded
to nearest. Error size (MSE, clipping, zeroing) is nearly the same in both
directions, so direction changes the *structure* of the weight error, not its
size.

### NVFP4: scale rounding matters as much as direction

Rounding the E4M3 scale to nearest instead of down (FLOOR, the MX rule for
power-of-two scales) **halves NVFP4's weight loss** (+1.34 → +0.64), cuts
clipped weights from 9.4 % to 3.7 %, and puts NVFP4 ahead of MXFP4.

NVFP4 here uses only its per-group scale, not the extra per-tensor scale of
the deployed format, so these numbers are compared only within this study.

---

## 3. The token-grouping benefit is not leakage

**The concern.** With token grouping, a token's group scale is computed from
later tokens in the same group, which a language model should not see.

**The control.** The last token of each group depends on no later token. Its
prediction is made without any future information, with the format
unchanged. Comparing the two directions on only those positions shows how
much of the benefit survives without leakage.

| Format | Reduction, all positions | Reduction, last-in-group only | Windows where tokens wins |
|---|---|---|---|
| MXFP4 | 46 % | 45 % | 6 of 8 |
| BFP4 | 61 % | 76 % | 8 of 8 |
| NVFP4 | 14 % | 8 % (not significant) | 5 of 8 |

For MXFP4 and BFP4 the benefit is the same or larger on positions that cannot
see the future, so **it is real and not leakage**. A second check (does the
benefit shrink toward the end of a group?) shows no consistent trend either.
For NVFP4 the benefit is too small to measure on these positions; there is
no sign of leakage, and grouping direction has no material effect on its
activations at this sample size.

A simpler "causal" variant (scale from the running maximum) was also tried
and rejected as a measure of leakage: it changes the quantization in other
ways, improving BFP4 and worsening MXFP4.

---

## How the reports are prepared

Each experiment goes through four documents, from raw to readable. Each one
is written only from the one above it.

```
results/<experiment>_<date>.json   raw numbers, written by the script
          │
working report (not published)     full analysis: every value, caveat and rerun
          │
reports/<topic>.md                 published report: method, results, interpretation
          │
FINDINGS.md                        this page: headline result of each report
```

**1. Raw results (`results/`).** Every experiment is a script in `scripts/`.
It writes a JSON file after each configuration finishes, so a run that is
interrupted keeps what it completed. The file records the full configuration
of each run, the FP32 baseline, perplexity, weight diagnostics and runtime;
the leakage control also writes the loss at every predicted position
(`_losses.json`) so any position-based statistic can be recomputed. Result
files are **never edited**. A rerun (for example after a bug fix) gets its
own file and the old one stays.

**2. Working report.** Written while the experiment runs and kept alongside
a dated progress log. It records every value with the file it came from,
superseded values with the reason they were replaced, interrupted runs, and
hypotheses that were considered. Before a difference is called an effect, it
is compared with the noise level (BFP8 lies within ±0.015 of FP32) and, where
per-position data exist, with a paired bootstrap interval and a per-window
count. Untested explanations are labelled as such. This document is complete
but long, and is not published.

**3. Published report (`reports/`).** A short rewrite of the working report,
one page per experiment: question and answer, setup, one results table,
conclusions, data files. Only current values are shown, and the history of
how the result was reached is left out.

**4. This page.** Only the headline of each published report, rounded to two
decimals. If this page and a report disagree, the report is correct.

---

## Where the details are

| Topic | Document |
|---|---|
| Emulator validation: method, results, errors found | [`reports/emulator_validation.md`](reports/emulator_validation.md) |
| Grouping direction and leakage control | [`reports/grouping_direction.md`](reports/grouping_direction.md) |
| How the quantizer works, code layout, how to run | [`README.md`](README.md) |
| Raw numbers | [`results/`](results/) |
