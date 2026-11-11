# Grouping direction

**Question.** Does the dimension along which a block-scaled format groups
values matter as much as the format itself?

**Answer (Qwen2.5-1.5B).** For activations, yes: grouping along tokens
instead of the hidden dimension cuts the loss by 46 % (MXFP4) and 61 %
(BFP4), and this is not caused by future-token leakage. For weights, barely.
For NVFP4, scale rounding matters more than direction.

## Setup

Qwen2.5-1.5B in FP32, WikiText-2 test, 8 × 2048-token windows (16,376
predictions). Either the weights or the input activations of every linear
layer except the LM head are quantized → dequantized; everything else,
including all arithmetic, stays FP32. FP32 perplexity: **8.098**. Noise
level ≈ ±0.015 (spread of BFP8 around FP32).

| Format | k | Scale | Element |
|---|---|---|---|
| MXFP4 | 32 | E8M0, FLOOR | E2M1 |
| BFP4 | 32 | E8M0, FLOOR | INT4 |
| NVFP4 | 16 | E4M3, FLOOR | E2M1 |
| NVFP4-nearest | 16 | E4M3, NEAREST | E2M1 |
| BFP8 | 32 | E8M0, FLOOR | INT8 |

NVFP4 omits its per-tensor FP32 scale, so it is comparable only within this
study.

## Results

Perplexity increase over FP32. Standard direction = along the reduction
dimension (`in` for weights `[out, in]`, `hidden` for activations).

| Format | Weights: `in` | Weights: `out` | Activations: `hidden` | Activations: tokens |
|---|---|---|---|---|
| MXFP4 | +1.032 | +1.009 | +1.863 | **+1.002** |
| BFP4 | +1.696 | +1.491 | +2.947 | **+1.162** |
| NVFP4 | +1.344 | +2.184 | +0.629 | +0.542 |
| NVFP4-nearest | +0.636 | +0.746 | +0.539 | +0.623 |
| BFP8 | +0.011 | +0.009 | +0.015 | −0.002 |

Weight MSE, clipping and zeroing are nearly identical in both directions for
every format.

## Leakage control

With token grouping, a token's group scale depends on later tokens. The
**last token of each group** depends on none, so its prediction is
leakage-free. Comparing directions on those positions only:

| Format | Reduction of ppl increase, all positions | Same, last-in-group only | ΔNLL last-in-group (nats), 95 % CI | Windows where tokens wins, last-in-group |
|---|---|---|---|---|
| MXFP4 | 46 % | 45 % | −0.102 [−0.166, −0.040] | 6 / 8 |
| BFP4 | 61 % | 76 % | −0.251 [−0.339, −0.167] | 8 / 8 |
| NVFP4 | 14 % | 8 % | −0.006 [−0.037, +0.025] | 5 / 8 |

Leakage would also make the gain larger early in a group than late. BFP4
and NVFP4 show no such pattern; for MXFP4 at most ≈ 0.02 of its 0.09-nat gain
could be leakage, and the trend is not consistent across windows.

## Conclusions

1. **Activation outliers are per-channel.** Hidden-direction groups mix
   outlier channels with ordinary ones; token groups stay within one channel.
   With token grouping BFP4 comes within 0.16 of MXFP4 (gap 1.08 otherwise).
2. **The gain is real** for MXFP4 and BFP4. For NVFP4 direction has no
   material effect on activations (−0.010 nats, better in only 4 / 8
   windows).
3. **FLOOR rounding hurts E4M3 scales.** NEAREST halves NVFP4's weight loss
   (+1.344 → +0.636, clipping 9.4 % → 3.7 %) and removes most of its
   output-channel penalty (+0.840 → +0.110).
4. **NVFP4-nearest is the exception:** token grouping makes it worse.

**Limitations.** One model, 16k tokens; weights and activations never
quantized together; hardware cost of non-reduction grouping not evaluated.

## Data

`results/grouping_direction_2026-09-22.json` (main),
`results/grouping_direction_followup_2026-09-23.json` (NVFP4 activations,
NVFP4-nearest), `results/leakage_control_2026-10-*.json` (+ `_losses.json`
per-position losses).

```bash
python scripts/grouping_direction.py --runs MXFP4:a-red MXFP4:a-tok --out results/run.json
```

Main arms: `w-red`, `w-out`, `a-red`, `a-tok`.
