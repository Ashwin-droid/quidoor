from __future__ import annotations

import numpy as np
import torch

from trainer.config import Config
from trainer.model import Inferencer, QuoridorNet, load_model


def tiny_config() -> Config:
    cfg = Config()
    cfg.net.blocks = 1
    cfg.net.channels = 16
    cfg.net.bf16 = False
    cfg.net.compile = False
    cfg.net.fixed_batch_size = 4
    return cfg


def test_model_shapes_and_optional_masking() -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg).eval()
    planes = torch.randn(3, 6, 9, 9)

    logits, values = model(planes)
    assert logits.shape == (3, 140)
    assert values.shape == (3,)
    assert torch.all(values <= 1)
    assert torch.all(values >= -1)

    mask = torch.zeros(3, 140, dtype=torch.bool)
    mask[:, [0, 7]] = True
    masked_logits, _ = model(planes, mask)
    assert torch.isfinite(masked_logits[:, 0]).all()
    assert torch.isfinite(masked_logits[:, 7]).all()
    assert torch.isneginf(masked_logits[:, 1]).all()


def test_inferencer_pads_to_fixed_shape_and_returns_masked_logits() -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg).eval()
    inferencer = Inferencer(model, cfg)

    planes = np.random.default_rng(7).standard_normal((3, 6, 9, 9), dtype=np.float32)
    masks = np.zeros((3, 140), dtype=np.uint8)
    masks[:, 0] = 1
    masks[1, 13] = 1

    logits, values = inferencer.infer(planes, masks)
    assert logits.shape == (3, 140)
    assert values.shape == (3,)
    assert logits.dtype == np.float32
    assert values.dtype == np.float32
    assert np.isfinite(logits[:, 0]).all()
    assert np.isneginf(logits[0, 13])
    assert np.isfinite(logits[1, 13])


def test_inferencer_uses_bounded_bucket_shapes() -> None:
    cfg = tiny_config()
    cfg.net.fixed_batch_size = 64
    model = QuoridorNet(cfg).eval()
    inferencer = Inferencer(model, cfg)
    seen_batches: list[int] = []

    class Recorder(torch.nn.Module):
        def forward(self, planes: torch.Tensor):
            seen_batches.append(int(planes.shape[0]))
            batch = int(planes.shape[0])
            return (
                torch.zeros(batch, cfg.rules.num_actions, device=planes.device),
                torch.zeros(batch, device=planes.device),
            )

    inferencer._model_for_infer = Recorder()
    planes = np.zeros((70, 6, 9, 9), dtype=np.float32)
    masks = np.ones((70, 140), dtype=np.bool_)

    logits, values = inferencer.infer(planes, masks)

    assert logits.shape == (70, 140)
    assert values.shape == (70,)
    assert inferencer.batch_buckets == (32, 64)
    assert seen_batches == [64, 32]


def test_load_model_reads_checkpoint_state(tmp_path) -> None:
    cfg = tiny_config()
    model = QuoridorNet(cfg)
    torch.save({"model_state": model.state_dict()}, tmp_path / "state.pt")

    loaded = load_model(tmp_path, cfg, torch.device("cpu"))
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)
