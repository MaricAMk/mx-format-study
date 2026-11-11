# mx-format-study

An emulator and evaluation harness for block-scaled low-precision number
formats (OCP MX, NVFP4, classic block floating point) in PyTorch.

Block-scaled formats store a group of values as low-precision *elements* plus
one shared *scale* per group. Published formats such as MXFP4 and NVFP4 differ
in several design choices at once, which makes it hard to attribute accuracy
differences to any single one. This project treats those choices as
independent parameters, so that each can be varied while the others are held
fixed:

| Axis | Values supported |
|---|---|
| Group size *k* | any size that divides the grouped dimension (study uses 8–128) |
| Scale format | E8M0 (power of two), E4M3, FP16, FP32 |
| Element format | pure mantissa (2/4/6/8 bits, i.e. BFP), minifloat E2M1, E2M3, E3M2, E4M3, E5M2 |
| Grouping axis | any tensor dimension |
| Rounding and overflow | scale rounding (floor/ceil/nearest), element rounding (RNE/toward zero/stochastic), overflow (saturate/NaN) |

The emulator models **representation error only**. A tensor is quantized and
immediately dequantized back to floating point; matrix multiplication and
accumulation run in FP32. Accuracy differences can therefore be attributed to
the number format and not to arithmetic hardware.

The repository covers the numerics workstream of a larger project; the full
proposal and plans are in `docs/` (see [Documents](#documents)).

---

## Quick start

```bash
git clone --recurse-submodules <repo-url>
cd mx-format-study
python -m venv .venv
source .venv/Scripts/activate            # Windows Git Bash; PowerShell: .\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
pip install -e external/microxcaling --no-deps
pip install -e . --no-deps
pytest tests/
```

`--no-deps` is required: `microxcaling` pins `torch==2.2.0`, and letting pip
resolve it would downgrade the installed torch. The library works correctly
with torch 2.5.1.

To run the model experiments, additionally install `transformers`:

```bash
pip install "transformers>=4.45,<5"
python scripts/grouping_direction.py --max-windows 8
```

`pip install -e .` makes `mx_format_study` importable from scripts. pytest
also adds `src/` to the import path through `pyproject.toml`, so the tests run
without it (this is what CI relies on). `ruff format` (configured in
`pyproject.toml`, line length 100) is the code formatter; it is optional and
not installed by `requirements.txt`.

---

## Using the quantizer

```python
import torch
from mx_format_study.formats import (
    QuantConfig, E8M0, E4M3_SCALE, E2M1, MANTISSA4,
    quantize_dequantize, quantize_dequantize_fast,
)

mxfp4 = QuantConfig(group_size=32, scale_format=E8M0, element_format=E2M1)
nvfp4 = QuantConfig(group_size=16, scale_format=E4M3_SCALE, element_format=E2M1)
bfp4  = QuantConfig(group_size=32, scale_format=E8M0, element_format=MANTISSA4)

w = torch.randn(4096, 4096)
result = quantize_dequantize_fast(w, mxfp4)   # groups along the last dimension
result.output              # same shape and dtype as w
result.scale               # one scale per group
result.clipped, result.zeroed, result.magnitude_spread, result.scale_utilization

# Group along dimension 0 instead:
from dataclasses import replace
result_t = quantize_dequantize_fast(w, replace(mxfp4, axis=0))
```

`quantize_dequantize` (the reference implementation) has the same signature and
returns identical values; it is a per-element Python loop and is only practical
for small tensors.

---

## How quantization is computed

For each group of `k` values along `config.axis`:

1. **Special cases.** If any value is NaN or Inf, the whole group becomes NaN
   (the shared-scale computation reads every element). An all-zero group stays
   zero.
2. **Scale.** Let `amax` be the group's largest magnitude and `M` the element
   format's largest representable magnitude.
   - Power-of-two scales (E8M0) follow the MX shared-exponent rule:
     `scale = 2^(r(log2(amax)) - floor(log2(M)))`, where `r` is the scale
     rounding mode (`ScaleRound`: floor by default, or ceil or nearest). Only
     the exponent of `M` is subtracted; the element format's mantissa headroom
     absorbs the rest.
   - Scales with mantissa bits (E4M3, FP16, FP32) round `amax / M` to the scale
     format's grid, in the same rounding mode.
   - The scale saturates at the scale format's largest finite value. A scale
     that underflows to zero flushes the group to zero.
3. **Elements.** Each value is divided by the scale and rounded to the element
   grid (`ElementRound`). Pure-mantissa elements have a uniform step of 1 and a
   maximum of `2^(bits-1) - 1`. Minifloat elements have a step that depends on
   the value's binade, with subnormals below the smallest normal. Values beyond
   the maximum saturate or become NaN (`OverflowPolicy`).
4. **Dequantize.** Multiply back by the scale.

Format maxima follow the specifications, including reserved codes: E4M3 has a
maximum of 448 (OCP FP8 reserves `S.1111.111` for NaN), E5M2 has 57344 (IEEE-style
Inf/NaN exponent), FP16 has 65504. FP4 and FP6 formats have no reserved codes.

Per-group diagnostics are returned with every result:

| Diagnostic | Meaning |
|---|---|
| `scale` | the scale chosen for the group |
| `magnitude_spread` | largest / smallest nonzero magnitude in the group |
| `clipped` | elements whose scaled magnitude exceeded the element maximum |
| `zeroed` | nonzero inputs that quantized to zero |
| `scale_utilization` | `amax / (scale * M)`: how much of the element range the group uses |
| `poisoned` | whether a NaN/Inf input invalidated the group |

---

## Code overview

### `src/mx_format_study/formats/` — the quantizer

| File | Contents |
|---|---|
| `config.py` | The design space. `ScaleFormat` and `ElementFormat` describe number formats; predefined instances (`E8M0`, `E4M3_SCALE`, `E2M1`, `MANTISSA4`, …) and lookup tables `SCALE_FORMATS` / `ELEMENT_FORMATS`. `QuantConfig` combines one choice per axis. The rounding and overflow enums are defined here with the project defaults (scale FLOOR, element RNE, SATURATE). |
| `minifloat.py` | Scalar rounding primitives: `round_scale`, `round_element`, `max_magnitude`, and the shared `_step_and_max` that gives the quantization step and maximum for any minifloat at a given magnitude. |
| `reference.py` | `quantize_dequantize`: the reference implementation. Loops over groups and elements in plain Python, calling `minifloat.py`. Returns a `QuantResult` with a list of `GroupDiagnostics`. |
| `fast.py` | `quantize_dequantize_fast`: the same arithmetic as whole-tensor torch operations in float64, running on CPU or GPU. Returns a `FastQuantResult` with diagnostics as tensors. Stochastic rounding is not supported here, since its random stream could not be validated against the reference. Also `quantize_dequantize_prefix_fast`, a causal variant in which each element's scale is computed from the running maximum of the elements up to it within its group; it is used as a control when grouping activations along tokens. |

The two implementations are kept separate on purpose: `reference.py` is short
enough to check against the specification by reading, and `fast.py` is tested
for exact agreement with it (`tests/test_fast_quantizer.py`).

### `src/mx_format_study/eval/` — model evaluation

| File | Contents |
|---|---|
| `quantize_model.py` | `target_linears` selects every `nn.Linear` except the LM head. `quantize_weights` replaces their weights with quantized copies in place and returns aggregate diagnostics (`WeightStats`). `ActivationQuantizer` is a context manager that installs forward pre-hooks quantizing each linear layer's input; `causal=True` switches to the causal variant. |
| `perplexity.py` | Perplexity over non-overlapping windows of a token sequence. `position_losses` returns the loss of every predicted token (`[windows, seq_len - 1]`); `last_in_group` selects the last position of each token group of size `k`, used by the position-restricted leakage control. |
| `data.py` | Loads the WikiText-2 (raw) test split, joined with `"\n\n"` as in the standard Hugging Face recipe, and caches it in `data/`. It uses the Hugging Face datasets-server JSON API instead of the `datasets` library (see [Environment notes](#environment-notes)). |

Grouping axis conventions when applied to a model:

| Tensor | Shape | `axis=-1` | Other axis |
|---|---|---|---|
| Linear weight | `[out, in]` | along the input (reduction) dimension, the MX convention | `axis=0`: along output channels |
| Linear input activation | `[batch, seq, hidden]` | along the hidden (reduction) dimension | `axis=-2`: along tokens, per channel |

### `scripts/`

| File | Purpose |
|---|---|
| `grouping_direction.py` | Experiment 2: compares perplexity of Qwen2.5-1.5B with groups formed along the reduction dimension against the other dimension, separately for weights and activations. Formats: MXFP4, BFP4, NVFP4, BFP8, and NVFP4 with round-to-nearest scales (`NVFP4-nearest`). Arms: `w-red`, `w-out`, `a-red`, `a-tok`, and the causal control `a-tok-causal`. By default runs the first four formats × first four arms; `--runs FORMAT:ARM ...` selects specific pairs. Every run also records `last_in_group_ppl`, the perplexity over the last position of each token group of the format's group size (the position-restricted leakage control), and the per-position losses go to `<out>_losses.json` for paired analysis. Writes both files after every configuration and keeps Windows from sleeping while it runs. `--seq-len` must be a multiple of every group size in the plan. Other options: `--model`, `--max-windows`, `--seq-len`, `--quant-device`, `--out`. |
| `gen_ocp_conformance_vectors.py` | Regenerates `conformance_vectors/ocp_element_codes.txt` by decoding every code of each OCP format from the specification's bit layout. Independent of `formats/`. |

### `tests/`

Tests are organised by what they compare against:

| File | Compares | Against |
|---|---|---|
| `test_microxcaling_mxfp4_spec.py` | `microxcaling` MXFP4 | Behaviour stated in the OCP MX specification (special values, saturation, ties, monotonicity). Validates the external library used as the oracle by the other tests. |
| `test_reference_quantizer.py` | `reference.py` | The same properties, plus bit-exact agreement with `microxcaling` MXFP4 |
| `test_bfp_generic_path.py` | Pure-mantissa elements with E8M0 scale | `microxcaling` MXINT8/4/2, for group sizes 8–128 and both axes |
| `test_ocp_conformance.py` | Element and scale grids of both implementations | `conformance_vectors/ocp_element_codes.txt` and `microxcaling` |
| `test_fast_quantizer.py` | `fast.py` | `reference.py`, outputs and diagnostics, across every scale × element format and rounding mode. The causal variant is checked against the standard fast path (exact agreement on truncated groups, causality along the token axis). |
| `test_quantize_model.py` | `eval/quantize_model.py` | Direct calls to the fast quantizer (standard and causal), on a small toy model |
| `test_perplexity.py` | `eval/perplexity.py` | Per-position indexing, and the premise of the leakage control on a toy causal model: under token grouping, group-end positions are unaffected by later tokens and positions inside an unfinished group are affected |

None of the tests require a GPU or a model download. CI
(`.github/workflows/tests.yml`) runs the whole directory with CPU-only torch on
every push and pull request.

### `conformance_vectors/`

`ocp_element_codes.txt` lists every code of E2M1, E2M3, E3M2, E4M3, E5M2 and
E8M0 with its decoded value, one per line: `<format> <code hex> <value>`,
where the value is written with `float.hex()` (exact) or as `nan`/`inf`.
Generated by `scripts/gen_ocp_conformance_vectors.py`; do not edit by hand.

### `external/microxcaling/`

Microsoft's MX emulation library, included as a git submodule pinned at
v1.1.0 and not modified. It serves as the reference implementation for every
format it can express (MXFP4/6/8, MXINT8/4/2). Only its plain-PyTorch path is
used; its `custom_cuda` extension is not required. See
`docs/decisions/forking_microxcaling_strategy.ipynb` for the reasoning.

### `results/`

Raw experiment outputs. Final results are JSON files (one per experiment run,
named `<experiment>_<date>.json`) and are tracked by git; everything else in
this folder, such as run logs, is ignored. Test-run printouts are not kept
here, since CI records every test run. Written interpretations of these files
go in `docs/reports/`, under the same name.

---

## Validation

Correctness is established in three layers, following
`docs/plan/workstream-1-emulator-and-accuracy.md` §5:

1. **Specification.** Every code of every OCP format is decoded independently
   and the quantizer's grid is checked against it: each representable value
   round-trips unchanged, each midpoint rounds to the even code, and values
   beyond the maximum saturate or become NaN.
2. **Upstream agreement.** For every configuration `microxcaling` can express,
   the output must match it bit for bit.
3. **Internal consistency.** The fast implementation must match the reference
   on outputs and diagnostics across the full design space, including directed
   edge cases (NaN/Inf, underflow, saturation, ties, values just below powers
   of two).

---

## Environment notes

- Development machine: Windows 11, Python 3.11.9, torch 2.5.1+cu121, GTX 1650
  (4 GB), 16 GB RAM. Qwen2.5-1.5B does not fit on the GPU in FP32, so model
  evaluation runs on CPU; one 2048-token window takes about 55 seconds.
- On this machine Windows Application Control blocks the compiled pandas and
  pyarrow libraries, so `datasets` cannot be imported. `eval/data.py` avoids
  that dependency.
- `requirements.txt` contains only what the tests and CI need. `transformers`
  is needed only for `scripts/grouping_direction.py`.
- `pip install -e external/microxcaling` generates an untracked `_version.py`
  inside the submodule. It is harmless and can be deleted.

---

## Repository layout

```
mx-format-study/
├── src/mx_format_study/
│   ├── formats/            quantizer: config, scalar rounding, reference and fast implementations
│   └── eval/               model-level quantization, perplexity, dataset loading
├── scripts/                experiments and generators
├── tests/                  pytest suite (runs in CI)
├── conformance_vectors/    OCP code tables used by the tests
├── external/microxcaling/  git submodule, pinned at v1.1.0
├── docs/
│   ├── plan/               proposal, workstream plan, scope document
│   ├── decisions/          design decisions (how microxcaling is included)
│   ├── reports/            write-ups interpreting the raw outputs in results/
│   └── progress-log.md     dated record of work and findings
├── results/                raw experiment outputs (JSON), interpreted in docs/reports/
├── FINDINGS.md             short summary of all results
├── reports/                published experiment reports
├── data/                   downloaded datasets (gitignored)
├── .github/workflows/      CI: runs tests/ on every push and pull request
├── pyproject.toml          package metadata, pytest and ruff configuration
└── requirements.txt        dependencies for tests and CI
```

---

## Documents

`docs/plan/` holds the plans the work follows, `docs/decisions/` records
design decisions with their reasoning, `docs/reports/` interprets the raw
outputs in `results/`, and
`docs/progress-log.md` records what was done and when.

| File | Contents |
|---|---|
| `FINDINGS.md` | **Start here.** Short summary of all results so far, and how the reports are prepared. |
| `reports/emulator_validation.md` | Validation of the quantizer against the OCP specifications and `microxcaling`. |
| `reports/grouping_direction.md` | Experiment 2: grouping direction and the leakage control. |
| `docs/plan/option-A-number-formats-proposal.md` | Full project proposal: background, the three axes, experiments, confounds, 16-week plan. |
| `docs/plan/workstream-1-emulator-and-accuracy.md` | Build plan and weekly schedule for this repository. |
| `docs/plan/workstream-1-scope-document.md` | The design space and pinned defaults, frozen at the end of week 2. |
| `docs/decisions/forking_microxcaling_strategy.ipynb` | How `microxcaling` is included and why. |
| `docs/reports/grouping_direction_2026-09-22.md` | Experiment 2 (grouping direction): setup, results of the main and follow-up runs, weight diagnostics, observations, validity questions, and the position-restricted leakage control (§7, partial results from 2026-10-05). |
| `docs/reports/mx_directed_vectors_2026-09-13.ipynb` | Results and interpretation of the first conformance test suite; the raw test output is `mx_directed_vectors_2026-09-13.txt` in the same folder. |
| `docs/progress-log.md` | Dated record of work, decisions and findings. Read the latest entry for the current state. |
