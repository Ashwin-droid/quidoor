from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RulesConfig:
    board: int = 9
    num_actions: int = 140
    walls_per_player: int = 10
    ply_cap: int = 200


@dataclass(slots=True)
class MctsConfig:
    sims_per_move: int = 800
    fast_sims_per_move: int = 400
    c_puct: float = 1.5
    fpu_reduction: float = 0.2
    dirichlet_alpha: float = 0.08
    dirichlet_epsilon: float = 0.25
    virtual_loss: int = 3
    temperature_moves: int = 16
    temperature: float = 1.0
    temperature_final: float = 0.0


@dataclass(slots=True)
class SelfPlayConfig:
    concurrent_games: int = 256
    leaf_batch_size: int = 1024
    collect_timeout_ms: int = 2
    phase_games: int = 500


@dataclass(slots=True)
class NetConfig:
    blocks: int = 8
    channels: int = 128
    extra_planes: bool = False
    bf16: bool = True
    compile: bool = True
    compile_mode: str = "max-autotune"
    fixed_batch_size: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    replay_buffer_size: int = 1_000_000
    replay_shard_size: int = 100_000
    mirror_probability: float = 0.5
    batch_size: int = 512
    train_steps_per_phase: int = 1000
    planned_train_steps: int = 100_000
    optimizer: str = "sgd"
    learning_rate: float = 0.05
    adamw_learning_rate: float = 2e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"
    # Weight of the continuous eval-score head loss (smooth L1 on /10-scaled
    # scores). 0 disables the head's gradient without changing the model.
    eval_score_weight: float = 1.0


@dataclass(slots=True)
class EvalConfig:
    games_vs_previous: int = 200
    games_vs_baseline: int = 100
    arena_temperature_moves: int = 0
    opening_temperature_moves: int = 4
    checkpoint_every_phases: int = 1


@dataclass(slots=True)
class CheckpointConfig:
    autosave_minutes: int = 30
    keep_last: int = 5
    keep_every: int = 10


@dataclass(slots=True)
class Config:
    rules: RulesConfig = field(default_factory=RulesConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    net: NetConfig = field(default_factory=NetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_engine_kwargs(self) -> dict[str, int | float | bool]:
        """Return keyword arguments accepted by ``quoridor_engine.EngineConfig``."""
        return {
            "sims_per_move": self.mcts.sims_per_move,
            "fast_sims_per_move": self.mcts.fast_sims_per_move,
            "c_puct": self.mcts.c_puct,
            "fpu_reduction": self.mcts.fpu_reduction,
            "dirichlet_alpha": self.mcts.dirichlet_alpha,
            "dirichlet_epsilon": self.mcts.dirichlet_epsilon,
            "virtual_loss": self.mcts.virtual_loss,
            "temperature_moves": self.mcts.temperature_moves,
            "temperature": self.mcts.temperature,
            "temperature_final": self.mcts.temperature_final,
            "concurrent_games": self.selfplay.concurrent_games,
            "leaf_batch_size": self.selfplay.leaf_batch_size,
            "collect_timeout_ms": self.selfplay.collect_timeout_ms,
            "ply_cap": self.rules.ply_cap,
            "extra_planes": self.net.extra_planes,
        }

    def asdict(self) -> dict:
        return asdict(self)
