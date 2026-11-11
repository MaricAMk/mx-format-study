# Emulator validation

**Question.** Does the quantizer produce exactly the values the OCP
specifications define?

**Answer.** Yes. It matches the specification code tables and Microsoft's
`microxcaling` bit for bit, and the fast implementation matches the
reference bit for bit (stochastic rounding excluded). 345 tests run in CI
on every push.

## Three layers

| Layer | Compared against | Result |
|---|---|---|
| Specification | All 912 codes of E2M1, E2M3, E3M2, E4M3, E5M2, E8M0, decoded independently from the spec bit layouts | Every value round-trips, midpoints round to even, overflow saturates or becomes NaN |
| Upstream | `microxcaling` v1.1.0: MXFP4, MXINT8/4/2 at k = 8–128, both axes | Bit-exact (also 0 mismatches in a one-off check of 200 random blocks spanning 1e-30 to 1e30) |
| Internal | Vectorized `fast.py` vs. scalar `reference.py`, every format, deterministic rounding mode, overflow policy and axis | Bit-exact on outputs and all diagnostics |

One spec behaviour worth knowing: a single NaN or Inf in an MX block turns
the **whole block** into NaN, because the shared scale reads every element.

## Errors caught

| Error | Fix |
|---|---|
| E8M0 scale computed as `amax / M` rounded to a power of two | MX rule: subtract only the exponent of `M`, i.e. `2^(floor(log2 amax) − floor(log2 M))` |
| Reserved Inf/NaN codes ignored (E4M3 max was 480) | E4M3 max 448, E5M2 57344, FP16 65504 |

All results use the corrected code, except the NVFP4 FLOOR weight values,
which were run with a maximum of 480 but are unaffected: the scale maximum
only matters for groups above 448 × 6 = 2688, and the largest weight is 3.06.

## Reproduce

```bash
pytest tests/
```
