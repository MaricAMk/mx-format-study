from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mx_format_study.eval.data import wikitext2_test_text
from mx_format_study.eval.perplexity import (
    last_in_group,
    perplexity_of,
    position_losses,
    tokenize,
)
from mx_format_study.eval.quantize_model import (
    ActivationQuantizer,
    quantize_weights,
    target_linears,
)
from mx_format_study.formats import (
    E2M1,
    E4M3_SCALE,
    E8M0,
    MANTISSA4,
    MANTISSA8,
    QuantConfig,
    ScaleRound,
)

MODEL = "Qwen/Qwen2.5-1.5B"
ROOT = Path(__file__).resolve().parents[1]

FORMATS = {
    "MXFP4": QuantConfig(group_size=32, scale_format=E8M0, element_format=E2M1),
    "BFP4": QuantConfig(group_size=32, scale_format=E8M0, element_format=MANTISSA4),
    "NVFP4": QuantConfig(group_size=16, scale_format=E4M3_SCALE, element_format=E2M1),
    "BFP8": QuantConfig(group_size=32, scale_format=E8M0, element_format=MANTISSA8),
    # NVFP4 with round-to-nearest scales instead of the pinned FLOOR default.
    "NVFP4-nearest": QuantConfig(
        group_size=16, scale_format=E4M3_SCALE, element_format=E2M1, scale_round=ScaleRound.NEAREST
    ),
}

# key: (target, direction label, axis on the tensor as it reaches the matmul, causal)
ARMS = {
    "w-red": ("weights", "reduction (in)", -1, False),
    "w-out": ("weights", "output channels", 0, False),
    "a-red": ("activations", "reduction (hidden)", -1, False),
    "a-tok": ("activations", "tokens", -2, False),
    "a-tok-causal": ("activations", "tokens (causal)", -2, True),
}
DEFAULT_ARMS = ["w-red", "w-out", "a-red", "a-tok"]


def config_dict(c: QuantConfig) -> dict:
    d = asdict(c)
    return {k: (v.value if hasattr(v, "value") else v) for k, v in d.items()}


@contextlib.contextmanager
def keep_awake():
    """Keep Windows from sleeping during the run. No-op elsewhere."""
    if sys.platform != "win32":
        yield
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


def rounded(losses: torch.Tensor) -> list[list[float]]:
    return [[round(v, 6) for v in row] for row in losses.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--formats", nargs="*", default=["MXFP4", "BFP4", "NVFP4", "BFP8"])
    ap.add_argument(
        "--runs",
        nargs="*",
        help="explicit FORMAT:ARM pairs (arms: " + ", ".join(ARMS) + "); overrides --formats",
    )
    ap.add_argument("--quant-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results" / f"grouping_direction_{date.today()}.json"
    )
    args = ap.parse_args()

    if args.runs:
        plan = [tuple(r.split(":", 1)) for r in args.runs]
        if any(len(p) != 2 for p in plan):
            raise SystemExit("--runs takes FORMAT:ARM pairs")
    else:
        plan = [(f, a) for f in args.formats for a in DEFAULT_ARMS]
    for fmt, arm in plan:
        if fmt not in FORMATS or arm not in ARMS:
            raise SystemExit(f"unknown run {fmt}:{arm}")
        if args.seq_len % FORMATS[fmt].group_size:
            raise SystemExit(f"--seq-len must be a multiple of {fmt}'s group size")

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    ids = tokenize(tokenizer, wikitext2_test_text())

    # The checkpoint is bf16, so a bf16 copy restores the weights exactly.
    originals = {
        n: m.weight.data.to(torch.bfloat16).clone() for n, m in target_linears(model).items()
    }

    def restore() -> None:
        for n, m in target_linears(model).items():
            m.weight.data.copy_(originals[n].to(torch.float32))

    def losses() -> torch.Tensor:
        return position_losses(model, ids, args.seq_len, args.max_windows)

    group_sizes = sorted({FORMATS[f].group_size for f, _ in plan})
    losses_out = args.out.with_name(args.out.stem + "_losses.json")
    all_losses = {"positions": "column t: loss of the prediction made at position t", "runs": []}

    results = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1 test",
        "seq_len": args.seq_len,
        "windows": min(ids.numel() // args.seq_len, args.max_windows),
        # An E4M3 scale saturates only for group maxima above 448 * 6.
        "max_abs_weight": max(float(w.abs().max()) for w in originals.values()),
        "runs": [],
    }

    t0 = time.time()
    base = losses()
    results["fp32_ppl"] = perplexity_of(base)
    results["fp32_last_in_group_ppl"] = {
        str(k): perplexity_of(last_in_group(base, k)) for k in group_sizes
    }
    all_losses["fp32"] = rounded(base)
    print(f"FP32 baseline: {results['fp32_ppl']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    for fmt, arm in plan:
        target, direction, axis, causal = ARMS[arm]
        cfg = replace(FORMATS[fmt], axis=axis)
        t0 = time.time()
        run = {
            "format": fmt,
            "arm": arm,
            "target": target,
            "direction": direction,
            "causal": causal,
            "config": config_dict(cfg),
        }
        if target == "weights":
            stats = quantize_weights(model, cfg, device=args.quant_device)
            run_losses = losses()
            spread = torch.cat(stats.spread_log2)
            run["weight_diagnostics"] = {
                "relative_mse": stats.relative_mse,
                "clipped_frac": stats.clipped / stats.elements,
                "zeroed_frac": stats.zeroed / stats.elements,
                "median_log2_spread": float(spread.median()),
                "p99_log2_spread": float(spread.kthvalue(int(0.99 * spread.numel())).values),
            }
            restore()
        else:
            with ActivationQuantizer(model, cfg, causal=causal):
                run_losses = losses()
        run["ppl"] = perplexity_of(run_losses)
        run["last_in_group_ppl"] = perplexity_of(last_in_group(run_losses, cfg.group_size))
        run["seconds"] = time.time() - t0
        results["runs"].append(run)
        all_losses["runs"].append({"format": fmt, "arm": arm, "losses": rounded(run_losses)})
        print(
            f"{fmt:13s} {target:11s} {direction:20s} ppl={run['ppl']:.4f}"
            f"  last-in-group={run['last_in_group_ppl']:.4f}  ({run['seconds']:.0f}s)",
            flush=True,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        losses_out.write_text(json.dumps(all_losses))


if __name__ == "__main__":
    with keep_awake():
        main()
