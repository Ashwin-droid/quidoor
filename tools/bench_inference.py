#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.config import Config
from trainer.model import Inferencer, QuoridorNet, input_planes


def parse_batch_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QuoridorNet inference")
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=parse_batch_sizes("1,8,32,128,512,1024"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--extra-planes", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    else:
        print("warning: CUDA is unavailable; reporting CPU throughput")

    print()
    print(f"{'batch':>8} {'latency_ms':>12} {'pos/s':>12} {'compiled':>10} {'dtype':>8}")
    print("-" * 58)

    for batch_size in args.batch_sizes:
        cfg = Config()
        cfg.net.extra_planes = bool(args.extra_planes)
        cfg.net.fixed_batch_size = batch_size
        cfg.net.compile = not args.no_compile
        cfg.net.bf16 = not args.fp32
        cfg.net.blocks = 8
        cfg.net.channels = 128

        model = QuoridorNet(cfg).to(device)
        inferencer = Inferencer(model, cfg)
        rng = np.random.default_rng(1234)
        planes = rng.standard_normal(
            (batch_size, input_planes(cfg), cfg.rules.board, cfg.rules.board),
            dtype=np.float32,
        )
        masks = np.ones((batch_size, cfg.rules.num_actions), dtype=np.bool_)

        for _ in range(args.warmup):
            inferencer.infer(planes, masks)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            inferencer.infer(planes, masks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latency_ms = elapsed / args.iters * 1000.0
        positions_per_second = batch_size * args.iters / elapsed
        dtype = "bf16" if cfg.net.bf16 and device.type == "cuda" else "fp32"
        print(
            f"{batch_size:8d} {latency_ms:12.3f} {positions_per_second:12.0f} "
            f"{str(inferencer.compiled):>10} {dtype:>8}"
        )


if __name__ == "__main__":
    main()
