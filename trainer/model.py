from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config


def input_planes(config: Config) -> int:
    return 8 if config.net.extra_planes else 6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class QuoridorNet(nn.Module):
    """Small AlphaZero-style ResNet for canonical Quoridor positions."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        board = config.rules.board
        channels = config.net.channels
        in_channels = input_planes(config)
        num_actions = config.rules.num_actions

        self.board = board
        self.num_actions = num_actions
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(config.net.blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board * board, num_actions),
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(board * board, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self, planes: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(self.stem(planes))
        policy_logits = self.policy_head(x)
        if legal_mask is not None:
            policy_logits = mask_illegal_logits(policy_logits, legal_mask)
        value = self.value_fc(self.value_conv(x)).squeeze(-1)
        return policy_logits, value


def mask_illegal_logits(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    legal = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    return policy_logits.masked_fill(~legal, float("-inf"))


def load_model(ckpt_dir: Path, config: Config, device: torch.device | str) -> QuoridorNet:
    ckpt_dir = Path(ckpt_dir)
    state_path = ckpt_dir / "state.pt"
    try:
        state: dict[str, Any] = torch.load(
            state_path, map_location=device, weights_only=False
        )
    except TypeError:  # pragma: no cover - for older torch versions
        state = torch.load(state_path, map_location=device)
    model_state = state.get("model_state", state)

    model = QuoridorNet(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


class Inferencer:
    """GPU-oriented bounded-shape inference wrapper.

    The Rust engine consumes raw logits and applies softmax internally, so this
    class only masks illegal actions to -inf and never normalizes policies.
    Partial batches are padded to a small bucket ladder rather than always to
    the maximum self-play batch size, which keeps torch.compile's shape cache
    bounded without making tiny arena/analysis calls pay for a 1024-row forward.
    """

    def __init__(self, model: QuoridorNet, config: Config) -> None:
        self.config = config
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.board = config.rules.board
        self.num_actions = config.rules.num_actions
        self.in_channels = input_planes(config)
        fixed = config.net.fixed_batch_size or config.selfplay.leaf_batch_size
        self.fixed_batch_size = max(1, int(fixed))
        self.batch_buckets = self._build_batch_buckets(self.fixed_batch_size)
        self.compiled = False
        self.compile_error: Exception | None = None

        self._model_for_infer: nn.Module = self.model
        running_under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        if (
            config.net.compile
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
            and not running_under_pytest
        ):
            try:
                self._model_for_infer = torch.compile(  # type: ignore[assignment]
                    self.model, mode=config.net.compile_mode
                )
                self.compiled = True
            except Exception as exc:  # pragma: no cover - depends on local torch stack
                self.compile_error = exc
                self._model_for_infer = self.model

    def infer(
        self, planes: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        planes_np = np.asarray(planes, dtype=np.float32)
        masks_np = np.asarray(masks).astype(np.bool_, copy=False)
        self._validate_inputs(planes_np, masks_np)

        batch = planes_np.shape[0]
        if batch == 0:
            return (
                np.empty((0, self.num_actions), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        policy_out = np.empty((batch, self.num_actions), dtype=np.float32)
        value_out = np.empty((batch,), dtype=np.float32)

        with torch.inference_mode():
            for start in range(0, batch, self.fixed_batch_size):
                stop = min(start + self.fixed_batch_size, batch)
                actual = stop - start
                padded = self._bucket_for_batch(actual)

                planes_chunk = np.zeros(
                    (
                        padded,
                        self.in_channels,
                        self.board,
                        self.board,
                    ),
                    dtype=np.float32,
                )
                masks_chunk = np.zeros(
                    (padded, self.num_actions), dtype=np.bool_
                )
                planes_chunk[:actual] = planes_np[start:stop]
                masks_chunk[:actual] = masks_np[start:stop]

                planes_t = torch.from_numpy(planes_chunk).to(
                    self.device, non_blocking=True
                )
                masks_t = torch.from_numpy(masks_chunk).to(
                    self.device, non_blocking=True
                )

                autocast_enabled = self.config.net.bf16 and self.device.type == "cuda"
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    logits_t, values_t = self._model_for_infer(planes_t)

                logits_t = mask_illegal_logits(logits_t.float(), masks_t)
                values_t = values_t.float()
                policy_out[start:stop] = logits_t[:actual].cpu().numpy()
                value_out[start:stop] = values_t[:actual].cpu().numpy()

        return policy_out, value_out

    @staticmethod
    def _build_batch_buckets(max_batch_size: int) -> tuple[int, ...]:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        first = min(32, max_batch_size)
        buckets: list[int] = []
        size = first
        while size < max_batch_size:
            buckets.append(size)
            size *= 2
        buckets.append(max_batch_size)
        return tuple(sorted(set(buckets)))

    def _bucket_for_batch(self, actual: int) -> int:
        if actual <= 0:
            raise ValueError("actual batch size must be positive")
        for bucket in self.batch_buckets:
            if actual <= bucket:
                return bucket
        return self.fixed_batch_size

    def _validate_inputs(self, planes: np.ndarray, masks: np.ndarray) -> None:
        expected_planes = (self.in_channels, self.board, self.board)
        if planes.ndim != 4 or planes.shape[1:] != expected_planes:
            raise ValueError(
                f"planes must have shape [B,{expected_planes[0]},"
                f"{expected_planes[1]},{expected_planes[2]}], got {planes.shape}"
            )
        if masks.ndim != 2 or masks.shape != (planes.shape[0], self.num_actions):
            raise ValueError(
                f"masks must have shape [B,{self.num_actions}], got {masks.shape}"
            )
