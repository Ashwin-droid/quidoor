from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import Config
from .lifecycle import (
    ShutdownController,
    load_checkpoint,
    resolve_checkpoint_dir,
    restore_rng_state,
    save_checkpoint_atomic,
    validate_resume_config,
)
from .logging_utils import (
    StatusWriter,
    StructuredLogger,
    SystemMetricsSampler,
    flatten_numeric,
    log_scalar_group,
    make_summary_writer,
    read_json_if_present,
)
from .model import Inferencer, QuoridorNet, load_model_state
from .replay_buffer import ReplayBuffer
from .selfplay_driver import (
    SelfPlayMetrics,
    inferencer_evaluator,
    run_selfplay_phase,
    stub_evaluator,
)


@dataclass(slots=True)
class RunResult:
    exit_code: int
    run_dir: Path
    counters: dict[str, int]
    last_checkpoint: Path | None
    overfit_trend: list[float] | None = None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.overfit_smoke:
        config = Config()
        apply_cli_overrides(config, args)
        device = select_device(args.device)
        trend = run_overfit_smoke(
            config,
            device=device,
            seed=args.seed,
            positions=args.overfit_positions,
            steps=args.overfit_steps,
            threshold=args.overfit_threshold,
        )
        print(
            "overfit-smoke loss trend: "
            + " -> ".join(f"{value:.6f}" for value in trend),
            flush=True,
        )
        return 0

    result = run_from_args(args)
    return result.exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quoridor AlphaZero training loop")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-phases", type=int, default=None)
    parser.add_argument("--status-interval-seconds", type=float, default=30.0)
    parser.add_argument("--stub-selfplay", action="store_true")
    parser.add_argument("--skip-hooks", action="store_true")

    parser.add_argument("--phase-games", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--planned-train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--replay-shard-size", type=int, default=None)
    parser.add_argument("--sims-per-move", type=int, default=None)
    parser.add_argument("--fast-sims-per-move", type=int, default=None)
    parser.add_argument("--concurrent-games", type=int, default=None)
    parser.add_argument("--leaf-batch-size", type=int, default=None)
    parser.add_argument("--collect-timeout-ms", type=int, default=None)
    parser.add_argument("--ply-cap", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--extra-planes", action="store_true", default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adamw-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--autosave-minutes", type=float, default=None)
    parser.add_argument("--keep-last", type=int, default=None)
    parser.add_argument("--keep-every", type=int, default=None)

    parser.add_argument("--overfit-smoke", action="store_true")
    parser.add_argument("--overfit-positions", type=int, default=1000)
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--overfit-threshold", type=float, default=0.08)
    return parser


def run_from_args(args: argparse.Namespace) -> RunResult:
    device = select_device(args.device)
    loaded: dict[str, Any] | None = None
    if args.resume is not None:
        loaded = load_checkpoint(args.resume, device=device, seed=args.seed)
        config: Config = loaded["config"]
        saved_config: Config = loaded["saved_config"]
        apply_cli_overrides(config, args)
        warnings = validate_resume_config(saved_config, config)
        if warnings:
            print(
                "resume config changed benign fields: " + ", ".join(sorted(warnings)),
                flush=True,
            )
        loaded = load_checkpoint(args.resume, config, device=device, seed=args.seed)
        run_dir = infer_run_dir_from_resume(args.resume)
    else:
        config = Config()
        apply_cli_overrides(config, args)
        run_dir = args.run_dir or default_run_dir(args.runs_root)

    return run_training(
        config,
        run_dir=run_dir,
        device=device,
        seed=args.seed,
        max_phases=args.max_phases,
        loaded_checkpoint=loaded,
        init_from=args.init_from,
        use_stub_selfplay=args.stub_selfplay,
        run_hooks=not args.skip_hooks,
        status_interval_seconds=args.status_interval_seconds,
    )


def run_training(
    config: Config,
    *,
    run_dir: Path,
    device: torch.device,
    seed: int = 1234,
    max_phases: int | None = None,
    loaded_checkpoint: dict[str, Any] | None = None,
    init_from: Path | None = None,
    use_stub_selfplay: bool = False,
    run_hooks: bool = True,
    status_interval_seconds: float = 30.0,
    console: bool = True,
) -> RunResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = StructuredLogger(run_dir, console=console)
    writer = make_summary_writer(run_dir)
    status = StatusWriter(run_dir, interval_seconds=status_interval_seconds)
    system_sampler = SystemMetricsSampler()
    shutdown = ShutdownController(run_dir, logger=logger)
    shutdown.install()
    last_checkpoint: Path | None = None
    last_autosave = time.monotonic()

    try:
        if loaded_checkpoint is not None:
            config = loaded_checkpoint["config"]
            buffer: ReplayBuffer = loaded_checkpoint["buffer"]
            counters = dict(loaded_checkpoint["counters"])
            last_checkpoint = Path(loaded_checkpoint["ckpt_dir"])
            restore_rng_state(loaded_checkpoint["state"].get("rng"))
            logger.event(
                "resume_loaded",
                checkpoint=loaded_checkpoint["ckpt_dir"],
                counters=counters,
                buffer_size=len(buffer),
            )
        else:
            buffer = ReplayBuffer(config, seed=seed)
            counters = {"games": 0, "steps": 0, "positions": 0}
            logger.event("run_started", run_dir=run_dir, device=str(device), seed=seed)

        model = QuoridorNet(config).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        if loaded_checkpoint is not None:
            state = loaded_checkpoint["state"]
            if not load_model_state(model, state["model_state"]):
                logger.event("score_head_initialized_fresh", reason="checkpoint predates score head")
            load_optimizer_state(optimizer, state.get("optimizer_state"), logger)
            load_scheduler_state(scheduler, state.get("scheduler_state"), logger)
        elif init_from is not None:
            ckpt_dir = resolve_checkpoint_dir(init_from)
            state = torch.load(ckpt_dir / "state.pt", map_location=device, weights_only=False)
            if not load_model_state(model, state["model_state"]):
                logger.event("score_head_initialized_fresh", reason="checkpoint predates score head")
            logger.event("init_from_loaded", checkpoint=ckpt_dir)

        inferencer = None
        if not use_stub_selfplay:
            inferencer = Inferencer(model, config)
            if inferencer.compile_error is not None:
                logger.event(
                    "inferencer_compile_skipped",
                    error=repr(inferencer.compile_error),
                )
        evaluator = stub_evaluator() if use_stub_selfplay else inferencer_evaluator(inferencer)

        status.write(
            build_status(
                phase="initializing",
                counters=counters,
                buffer=buffer,
                shutdown=shutdown,
                last_checkpoint=last_checkpoint,
            ),
            force=True,
        )

        def checkpoint(
            *,
            force: bool = False,
            fast: bool = False,
            counters_override: dict[str, int] | None = None,
        ) -> Path | None:
            nonlocal last_autosave, last_checkpoint
            if not force and config.checkpoint.autosave_minutes <= 0:
                return None
            due = (
                config.checkpoint.autosave_minutes > 0
                and time.monotonic() - last_autosave
                >= float(config.checkpoint.autosave_minutes) * 60.0
            )
            if not force and not due:
                return None
            active_counters = counters_override or counters
            last_checkpoint = save_checkpoint_atomic(
                run_dir=run_dir,
                step=active_counters["steps"],
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                buffer=buffer,
                counters=active_counters,
                logger=logger,
                fast=fast,
            )
            last_autosave = time.monotonic()
            return last_checkpoint

        def maybe_exit_fast() -> RunResult | None:
            if not shutdown.fast_abort_requested:
                return None
            checkpoint(force=True, fast=True)
            status.write(
                build_status(
                    phase="fast_abort",
                    counters=counters,
                    buffer=buffer,
                    shutdown=shutdown,
                    last_checkpoint=last_checkpoint,
                ),
                force=True,
            )
            return RunResult(130, run_dir, counters, last_checkpoint)

        phase_idx = 0
        while max_phases is None or phase_idx < max_phases:
            shutdown.poll_stop_file()
            fast_result = maybe_exit_fast()
            if fast_result is not None:
                return fast_result
            if shutdown.draining:
                break

            logger.event("phase_transition", phase="selfplay", phase_idx=phase_idx)
            selfplay_base = counters.copy()

            def selfplay_tick(metrics: SelfPlayMetrics) -> None:
                active_counters = {
                    "games": selfplay_base["games"] + metrics.games,
                    "steps": selfplay_base["steps"],
                    "positions": selfplay_base["positions"] + metrics.positions,
                }
                status.write(
                    build_status(
                        phase="selfplay",
                        counters=active_counters,
                        buffer=buffer,
                        shutdown=shutdown,
                        last_checkpoint=last_checkpoint,
                        phase_metrics=metrics.as_dict(),
                    )
                )
                checkpoint(counters_override=active_counters)

            sp_metrics = run_selfplay_phase(
                config=config,
                buffer=buffer,
                evaluator=evaluator,
                num_games=config.selfplay.phase_games,
                seed=seed + phase_idx * 1_000_003 + counters["games"],
                shutdown=shutdown,
                tick_callback=selfplay_tick,
                logger=logger,
            )
            counters["games"] += sp_metrics.games
            counters["positions"] += sp_metrics.positions
            log_scalar_group(writer, "self-play", sp_metrics.as_dict(), counters["steps"])
            log_system_and_strength(
                writer, run_dir, system_sampler, counters["steps"], logger=logger
            )
            writer.flush()

            fast_result = maybe_exit_fast()
            if fast_result is not None:
                return fast_result
            if shutdown.draining:
                break

            logger.event("phase_transition", phase="training", phase_idx=phase_idx)
            status.write(
                build_status(
                    phase="training",
                    counters=counters,
                    buffer=buffer,
                    shutdown=shutdown,
                    last_checkpoint=last_checkpoint,
                ),
                force=True,
            )
            for local_step in range(config.training.train_steps_per_phase):
                shutdown.poll_stop_file()
                fast_result = maybe_exit_fast()
                if fast_result is not None:
                    return fast_result
                if len(buffer) == 0:
                    logger.event("anomaly", kind="empty_replay_buffer")
                    break

                metrics = train_one_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    buffer=buffer,
                    config=config,
                    device=device,
                )
                counters["steps"] += 1
                logger.event(
                    "train_step",
                    step=counters["steps"],
                    local_step=local_step,
                    **metrics,
                )
                log_scalar_group(writer, "training", metrics, counters["steps"])
                if counters["steps"] % 10 == 0:
                    log_system_and_strength(
                        writer, run_dir, system_sampler, counters["steps"], logger=None
                    )
                status.write(
                    build_status(
                        phase="training",
                        counters=counters,
                        buffer=buffer,
                        shutdown=shutdown,
                        last_checkpoint=last_checkpoint,
                        phase_metrics=metrics,
                    )
                )
                checkpoint()
                if shutdown.draining:
                    break

            fast_result = maybe_exit_fast()
            if fast_result is not None:
                return fast_result

            last_checkpoint = checkpoint(force=True)
            if not shutdown.draining:
                run_checkpoint_hooks(
                    run_dir=run_dir,
                    ckpt_step=counters["steps"],
                    config=config,
                    logger=logger,
                    writer=writer,
                    run_hooks=run_hooks,
                )
            phase_idx += 1
            if shutdown.draining:
                break

        if shutdown.draining:
            logger.event("shutdown_drain_complete")
            checkpoint(force=True)
            shutdown.mark_stopped_clean()
            final_phase = "stopped_clean"
            exit_code = 0
        else:
            shutdown.shutdown_state = "completed"
            final_phase = "completed"
            exit_code = 0

        status.write(
            build_status(
                phase=final_phase,
                counters=counters,
                buffer=buffer,
                shutdown=shutdown,
                last_checkpoint=last_checkpoint,
            ),
            force=True,
        )
        return RunResult(exit_code, run_dir, counters, last_checkpoint)
    finally:
        writer.flush()
        writer.close()
        shutdown.restore()
        logger.close()


def train_one_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    buffer: ReplayBuffer,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    batch = buffer.sample_batch(config.training.batch_size)
    planes = torch.from_numpy(batch["planes"]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(batch["mask"]).to(device=device, dtype=torch.bool)
    policy_target = torch.from_numpy(batch["policy"]).to(device=device, dtype=torch.float32)
    z = torch.from_numpy(batch["z"]).to(device=device, dtype=torch.float32)
    score_target = torch.from_numpy(batch["eval_score"]).to(device=device, dtype=torch.float32)
    policy_target = normalize_policy_targets(policy_target, mask)

    optimizer.zero_grad(set_to_none=True)
    logits, values, scores = model(planes, mask)
    policy_loss, policy_entropy = soft_policy_loss_and_entropy(logits, policy_target, mask)
    value_loss = F.mse_loss(values, z)
    score_weight = float(config.training.eval_score_weight)
    # Scores span roughly +/-40, so scale down before the Huber loss to keep
    # this term comparable to the policy/value losses.
    score_loss = F.smooth_l1_loss(scores / 10.0, score_target / 10.0)
    loss = policy_loss + value_loss + score_weight * score_loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
    optimizer.step()
    scheduler.step()

    metrics = {
        "total_loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "eval_score_loss": float(score_loss.detach().cpu()),
        "eval_score_mae": float((scores.detach() - score_target).abs().mean().cpu()),
        "policy_entropy": float(policy_entropy.detach().cpu()),
        "value_target_correlation": value_target_correlation(values.detach(), z.detach()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
        "buffer_size": float(len(buffer)),
    }
    return metrics


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    decay, no_decay = split_weight_decay_params(model)
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.training.adamw_learning_rate,
        )
    if optimizer_name != "sgd":
        raise ValueError(f"unknown optimizer {config.training.optimizer!r}")
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
    )


def split_weight_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
    )
    seen: set[int] = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if name.endswith("bias") or isinstance(module, norm_types):
                no_decay.append(param)
            else:
                decay.append(param)
    return decay, no_decay


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> Any:
    if config.training.lr_schedule.lower() != "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config.training.planned_train_steps)),
        eta_min=0.0,
    )


def normalize_policy_targets(policy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    policy = policy * mask.float()
    sums = policy.sum(dim=1, keepdim=True)
    legal = mask.float()
    legal_counts = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform = legal / legal_counts
    normalized = policy / sums.clamp_min(1.0e-12)
    return torch.where(sums > 1.0e-8, normalized, uniform)


def soft_policy_loss_and_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=1)
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.exp(log_probs) * mask.float()
    policy_loss = -(target * log_probs).sum(dim=1).mean()
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return policy_loss, entropy


def value_target_correlation(values: torch.Tensor, target: torch.Tensor) -> float:
    values_f = values.float()
    target_f = target.float()
    if values_f.numel() < 2:
        return 0.0
    values_centered = values_f - values_f.mean()
    target_centered = target_f - target_f.mean()
    denom = values_centered.norm() * target_centered.norm()
    if float(denom) <= 1.0e-12:
        return 0.0
    corr = (values_centered * target_centered).sum() / denom
    return float(corr.detach().cpu())


def run_checkpoint_hooks(
    *,
    run_dir: Path,
    ckpt_step: int,
    config: Config,
    logger: StructuredLogger,
    writer: Any,
    run_hooks: bool = True,
) -> None:
    if not run_hooks:
        logger.event("checkpoint_hooks_skipped", reason="disabled")
        return
    hooks = [
        ("trainer.arena", "run_arena", "arena"),
        ("trainer.analyze", "run_probe_suite", "analyze"),
        ("trainer.viz", "generate_checkpoint_artifacts", "viz"),
    ]
    for module_name, func_name, label in hooks:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            result = func(Path(run_dir), int(ckpt_step), config)
            logger.event("checkpoint_hook_finished", hook=label, result=result)
            if isinstance(result, dict):
                log_scalar_group(writer, f"strength/{label}", flatten_numeric("", result), ckpt_step)
        except ImportError as exc:
            logger.event("checkpoint_hook_skipped", hook=label, reason=repr(exc))
        except Exception as exc:
            logger.event("checkpoint_hook_failed", hook=label, error=repr(exc))
    elo = read_json_if_present(Path(run_dir) / "elo.json")
    if elo is not None:
        log_scalar_group(writer, "strength/elo_json", flatten_numeric("", elo), ckpt_step)
    writer.flush()


def log_system_and_strength(
    writer: Any,
    run_dir: Path,
    sampler: SystemMetricsSampler,
    step: int,
    *,
    logger: StructuredLogger | None,
) -> None:
    metrics = sampler.sample()
    log_scalar_group(writer, "system", metrics, step)
    if logger is not None:
        logger.event("system_metrics", **metrics)
    elo = read_json_if_present(Path(run_dir) / "elo.json")
    if elo is not None:
        log_scalar_group(writer, "strength", flatten_numeric("", elo), step)


def build_status(
    *,
    phase: str,
    counters: dict[str, int],
    buffer: ReplayBuffer,
    shutdown: ShutdownController,
    last_checkpoint: Path | None,
    phase_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = buffer.stats()
    return {
        "state": shutdown.shutdown_state,
        "phase": phase,
        "games_completed": int(counters["games"]),
        "training_steps": int(counters["steps"]),
        "positions_generated": int(counters["positions"]),
        "buffer_fill": int(len(buffer)),
        "buffer_capacity": int(stats["capacity"]),
        "buffer_total_written": int(stats["total_written"]),
        "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
        "shutdown_state": shutdown.shutdown_state,
        "shutdown_reason": shutdown.stop_reason,
        "phase_metrics": phase_metrics or {},
    }


def run_overfit_smoke(
    config: Config,
    *,
    device: torch.device,
    seed: int,
    positions: int = 1000,
    steps: int = 200,
    threshold: float = 0.08,
) -> list[float]:
    config.training.replay_buffer_size = max(positions, config.training.batch_size)
    config.training.mirror_probability = 0.0
    config.net.compile = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = QuoridorNet(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    buffer = ReplayBuffer(config, seed=seed)
    planes, policy, mask, z = overfit_dataset(config, positions)
    buffer.add(planes, policy, mask, z)

    trend: list[float] = []
    checkpoints = {0, max(0, steps // 2), max(0, steps - 1)}
    for step in range(max(1, steps)):
        metrics = train_one_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            buffer=buffer,
            config=config,
            device=device,
        )
        if step in checkpoints:
            trend.append(metrics["total_loss"])
    if trend[-1] > threshold:
        raise AssertionError(
            f"overfit smoke final loss {trend[-1]:.6f} exceeds threshold {threshold:.6f}"
        )
    return trend


def overfit_dataset(
    config: Config, positions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import quoridor_engine as qe

        state = qe.GameState()
        base_planes = np.asarray(state.encode(config.net.extra_planes), dtype=np.float32)
        base_mask = np.asarray(state.legal_mask()).astype(np.bool_, copy=False)
    except Exception:
        base_planes = np.zeros(
            (
                8 if config.net.extra_planes else 6,
                config.rules.board,
                config.rules.board,
            ),
            dtype=np.float32,
        )
        base_planes[0, 0, 4] = 1.0
        base_planes[1, 8, 4] = 1.0
        base_mask = np.ones((config.rules.num_actions,), dtype=np.bool_)
    action = int(np.flatnonzero(base_mask)[0])
    planes = np.repeat(base_planes[None, ...], positions, axis=0)
    policy = np.zeros((positions, config.rules.num_actions), dtype=np.float32)
    policy[:, action] = 1.0
    mask = np.repeat(base_mask[None, ...], positions, axis=0)
    z = np.full((positions,), 0.25, dtype=np.float32)
    return planes, policy, mask, z


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.phase_games is not None:
        config.selfplay.phase_games = args.phase_games
    if args.train_steps is not None:
        config.training.train_steps_per_phase = args.train_steps
    if args.planned_train_steps is not None:
        config.training.planned_train_steps = args.planned_train_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.replay_buffer_size is not None:
        config.training.replay_buffer_size = args.replay_buffer_size
    if args.replay_shard_size is not None:
        config.training.replay_shard_size = args.replay_shard_size
    if args.sims_per_move is not None:
        config.mcts.sims_per_move = args.sims_per_move
    if args.fast_sims_per_move is not None:
        config.mcts.fast_sims_per_move = args.fast_sims_per_move
    if args.concurrent_games is not None:
        config.selfplay.concurrent_games = args.concurrent_games
    if args.leaf_batch_size is not None:
        config.selfplay.leaf_batch_size = args.leaf_batch_size
        config.net.fixed_batch_size = args.leaf_batch_size
    if args.collect_timeout_ms is not None:
        config.selfplay.collect_timeout_ms = args.collect_timeout_ms
    if args.ply_cap is not None:
        config.rules.ply_cap = args.ply_cap
    if args.blocks is not None:
        config.net.blocks = args.blocks
    if args.channels is not None:
        config.net.channels = args.channels
    if args.extra_planes:
        config.net.extra_planes = True
    if args.no_bf16:
        config.net.bf16 = False
    if args.no_compile:
        config.net.compile = False
    if args.optimizer is not None:
        config.training.optimizer = args.optimizer
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.adamw_learning_rate is not None:
        config.training.adamw_learning_rate = args.adamw_learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.grad_clip is not None:
        config.training.grad_clip = args.grad_clip
    if args.autosave_minutes is not None:
        config.checkpoint.autosave_minutes = args.autosave_minutes
    if args.keep_last is not None:
        config.checkpoint.keep_last = args.keep_last
    if args.keep_every is not None:
        config.checkpoint.keep_every = args.keep_every


def load_optimizer_state(
    optimizer: torch.optim.Optimizer, state: dict[str, Any] | None, logger: StructuredLogger
) -> None:
    if state is None:
        return
    try:
        optimizer.load_state_dict(state)
    except Exception as exc:
        logger.event("optimizer_state_skipped", error=repr(exc))


def load_scheduler_state(scheduler: Any, state: dict[str, Any] | None, logger: StructuredLogger) -> None:
    if state is None:
        return
    try:
        scheduler.load_state_dict(state)
    except Exception as exc:
        logger.event("scheduler_state_skipped", error=repr(exc))


def select_device(value: str | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_run_dir(runs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(runs_root) / f"{stamp}-{random.randrange(16**6):06x}"


def infer_run_dir_from_resume(path: Path) -> Path:
    path = Path(path)
    if (path / "state.pt").exists():
        return path.parents[1]
    return path


if __name__ == "__main__":
    raise SystemExit(main())
