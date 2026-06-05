#!/usr/bin/env python3
"""
Launch efficiency benchmarks via `run.sh --task eff-benchmark`.

Paper mapping (ICLR 2026, arXiv:2502.00340):
  - Table 2:  regular vs Centrifuge (--attn_filter), 50% filter, per model setup
  - Figure 5a: context-length sweep (1K–4K), regular vs Centrifuge
  - Figure 5b: filtering-ratio sweep (20%–60%), Centrifuge vs regular baseline

Modes:
  regular    — no token dropping, no graph acceleration (paper "Regular Training")
  loss_only  — fixed drop mask from eff-benchmark fake ref_loss, no --attn_filter
               (ablation: "filter loss only", Lin et al. style)
  centrifuge — loss_only + --attn_filter (paper "CENTRIFUGE" row)

Each run uses synthetic data in train.py (eff_bench_iters=12: 2 warmup + 10 timed steps).
Timing breakdown is written to batch_time*.json under the run output directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class BenchmarkConfig:
    cuda_devices: str
    model: str
    context_len: int
    mic_batch_size: int = 1
    batch_size: int = 512
    drop_rate: float = 0.5
    learning_rate: str = "5e-5"
    use_lora: bool = False


# Paper Table 2 (ZeRO-1 DP=4 unless noted)
TABLE2_CONFIGS = [
    BenchmarkConfig("0,1,2,3", "TinyLlama/TinyLlama_v1.1", context_len=4096),
    BenchmarkConfig("0,1,2,3", "Qwen/Qwen2.5-1.5B", context_len=2048),
    BenchmarkConfig("0", "meta-llama/Llama-3.2-3B", context_len=2048, use_lora=True),
]

FIG5A_CONTEXT_LENGTHS = [1024, 2048, 3072, 4096]
FIG5B_DROP_RATES = [0.2, 0.3, 0.4, 0.5, 0.6]


def run_efficiency_benchmark(
    cfg: BenchmarkConfig,
    mode: str,
    output_dir_suffix: str = "",
    extra_output_tag: str = "",
) -> int:
    """Build and execute one `run.sh --task eff-benchmark` job."""

    if mode not in ("regular", "loss_only", "centrifuge"):
        raise ValueError(f"Unknown mode: {mode}")

    num_procs = cfg.cuda_devices.count(",") + 1
    model_name = cfg.model.replace("/", "_")
    tag = output_dir_suffix or mode
    if extra_output_tag:
        tag = f"{tag}_{extra_output_tag}"

    output_dir = (
        f"effben_{model_name}_{num_procs}_{cfg.context_len}_"
        f"{cfg.batch_size}_{cfg.mic_batch_size}_{cfg.drop_rate}_{tag}"
    )

    cmd_parts = [
        "bash run.sh --task eff-benchmark",
        f"--model_path {cfg.model}",
        f"--output_dir {output_dir}",
        f"--hosts localhost:{cfg.cuda_devices}",
        "--master_addr localhost",
        "--node_rank 0",
        f"--learning_rate {cfg.learning_rate}",
        f"--context_length {cfg.context_len}",
        f"--mic_batch_size {cfg.mic_batch_size}",
        f"--batch_size {cfg.batch_size}",
        "--ref_model_backend none",
    ]

    if mode != "regular":
        cmd_parts.extend([
            "--pre_compute_ref",
            "--add_eos_token",
            "--packing_samples",
            "--dropping_strategy fixed",
            f"--drop_rate {cfg.drop_rate}",
        ])
        if mode == "centrifuge":
            cmd_parts.append("--attn_filter")

    if cfg.use_lora:
        cmd_parts.append("--use_lora")

    cmd = " ".join(cmd_parts)
    print("=" * 72)
    print(
        f"[{mode}] {cfg.model} | ctx={cfg.context_len} | "
        f"drop={cfg.drop_rate} | GPUs={cfg.cuda_devices}"
    )
    print(cmd)
    print("=" * 72)
    return os.system(cmd)


def run_modes(cfg: BenchmarkConfig, modes: Iterable[str], stop_on_error: bool = True) -> int:
    for mode in modes:
        rc = run_efficiency_benchmark(cfg, mode=mode)
        if rc != 0 and stop_on_error:
            print(f"Stopped: mode={mode} failed with exit code {rc}")
            return rc
    return 0


def suite_table2(stop_on_error: bool = True) -> int:
    """Paper Table 2: regular vs Centrifuge at 50% filtering."""
    for cfg in TABLE2_CONFIGS:
        rc = run_modes(cfg, ("regular", "centrifuge"), stop_on_error=stop_on_error)
        if rc != 0:
            return rc
    return 0


def suite_fig5a(
    cuda_devices: str = "0,1,2,3",
    model: str = "TinyLlama/TinyLlama_v1.1",
    batch_size: int = 512,
    mic_batch_size: int = 1,
    stop_on_error: bool = True,
) -> int:
    """Paper Figure 5a: throughput vs context length."""
    for ctx in FIG5A_CONTEXT_LENGTHS:
        cfg = BenchmarkConfig(
            cuda_devices,
            model,
            context_len=ctx,
            batch_size=batch_size,
            mic_batch_size=mic_batch_size,
            drop_rate=0.5,
        )
        rc = run_modes(cfg, ("regular", "centrifuge"), stop_on_error=stop_on_error)
        if rc != 0:
            return rc
    return 0


def suite_fig5b(
    cuda_devices: str = "0,1,2,3",
    model: str = "TinyLlama/TinyLlama_v1.1",
    context_len: int = 2048,
    batch_size: int = 512,
    mic_batch_size: int = 1,
    stop_on_error: bool = True,
) -> int:
    """Paper Figure 5b: speedup vs filtering ratio."""
    baseline_cfg = BenchmarkConfig(
        cuda_devices,
        model,
        context_len=context_len,
        batch_size=batch_size,
        mic_batch_size=mic_batch_size,
        drop_rate=0.5,
    )
    rc = run_efficiency_benchmark(baseline_cfg, mode="regular", output_dir_suffix="fig5b_baseline")
    if rc != 0 and stop_on_error:
        return rc

    for drop_rate in FIG5B_DROP_RATES:
        cfg = BenchmarkConfig(
            cuda_devices,
            model,
            context_len=context_len,
            batch_size=batch_size,
            mic_batch_size=mic_batch_size,
            drop_rate=drop_rate,
        )
        rc = run_efficiency_benchmark(
            cfg,
            mode="centrifuge",
            output_dir_suffix="fig5b",
            extra_output_tag=f"dr{int(drop_rate * 100)}",
        )
        if rc != 0 and stop_on_error:
            return rc
    return 0


def suite_quick(cfg: BenchmarkConfig, stop_on_error: bool = True) -> int:
    """Fast local check: regular + loss_only + centrifuge for one config."""
    return run_modes(cfg, ("regular", "loss_only", "centrifuge"), stop_on_error=stop_on_error)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned efficiency benchmarks (eff-benchmark task)."
    )
    parser.add_argument(
        "--suite",
        choices=["table2", "fig5a", "fig5b", "quick", "single"],
        default="quick",
        help="Predefined experiment suite (default: quick)",
    )
    parser.add_argument(
        "--mode",
        choices=["regular", "loss_only", "centrifuge"],
        help="Required when --suite single",
    )
    parser.add_argument("--cuda-devices", default="0,1,2,3", help="CUDA device list, e.g. 0,1,2,3")
    parser.add_argument("--model", default="TinyLlama/TinyLlama_v1.1", help="HF model id")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--mic-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drop-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", default="5e-5")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining jobs even if one fails",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stop_on_error = not args.continue_on_error

    cfg = BenchmarkConfig(
        cuda_devices=args.cuda_devices,
        model=args.model,
        context_len=args.context_length,
        mic_batch_size=args.mic_batch_size,
        batch_size=args.batch_size,
        drop_rate=args.drop_rate,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
    )

    if args.suite == "table2":
        return suite_table2(stop_on_error=stop_on_error)
    if args.suite == "fig5a":
        return suite_fig5a(
            cuda_devices=args.cuda_devices,
            model=args.model,
            batch_size=args.batch_size,
            mic_batch_size=args.mic_batch_size,
            stop_on_error=stop_on_error,
        )
    if args.suite == "fig5b":
        return suite_fig5b(
            cuda_devices=args.cuda_devices,
            model=args.model,
            context_len=args.context_length,
            batch_size=args.batch_size,
            mic_batch_size=args.mic_batch_size,
            stop_on_error=stop_on_error,
        )
    if args.suite == "quick":
        return suite_quick(cfg, stop_on_error=stop_on_error)
    if args.suite == "single":
        if not args.mode:
            print(
                "error: --suite single requires --mode regular|loss_only|centrifuge",
                file=sys.stderr,
            )
            return 2
        return run_efficiency_benchmark(cfg, mode=args.mode)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
