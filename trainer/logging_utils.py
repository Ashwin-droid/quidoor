from __future__ import annotations

import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    fsync_dir(path.parent)


class StructuredLogger:
    def __init__(self, run_dir: Path, console: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.console = console
        self.path = self.run_dir / "log.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")

    def event(self, event: str, **fields: Any) -> dict[str, Any]:
        payload = {"ts": utc_now_iso(), "event": event, **_jsonable_dict(fields)}
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()
        if self.console:
            suffix = " ".join(
                f"{key}={_short_console_value(value)}"
                for key, value in payload.items()
                if key not in {"ts", "event"}
            )
            message = f"[{payload['ts']}] {event}"
            if suffix:
                message = f"{message} {suffix}"
            print(message, flush=True)
        return payload

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class NoOpSummaryWriter:
    def add_scalar(self, *_args: object, **_kwargs: object) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def make_summary_writer(run_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return NoOpSummaryWriter()
    tb_dir = Path(run_dir) / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


class StatusWriter:
    def __init__(self, run_dir: Path, interval_seconds: float = 30.0) -> None:
        self.path = Path(run_dir) / "status.json"
        self.interval_seconds = float(interval_seconds)
        self._last_write = 0.0

    def write(self, payload: dict[str, Any], *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_write < self.interval_seconds:
            return False
        status = {"updated_at": utc_now_iso(), **_jsonable_dict(payload)}
        write_json_atomic(self.path, status)
        self._last_write = now
        return True


class SystemMetricsSampler:
    def sample(self) -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {
            "cpu_load_1m_per_core": None,
            "gpu_util_percent": None,
            "gpu_vram_used_mb": None,
            "gpu_vram_total_mb": None,
        }
        cores = os.cpu_count() or 1
        try:
            metrics["cpu_load_1m_per_core"] = os.getloadavg()[0] / cores
        except OSError:
            pass

        gpu_util, vram_used, vram_total = _torch_cuda_metrics()
        if gpu_util is None:
            gpu_util, vram_used, vram_total = _nvidia_smi_metrics()
        metrics["gpu_util_percent"] = gpu_util
        metrics["gpu_vram_used_mb"] = vram_used
        metrics["gpu_vram_total_mb"] = vram_total
        return metrics


def log_scalar_group(
    writer: Any,
    group: str,
    metrics: dict[str, Any],
    step: int,
    *,
    walltime: float | None = None,
) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"{group}/{key}", float(value), step, walltime=walltime)


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    if isinstance(data, dict):
        return data
    return {"value": data}


def flatten_numeric(prefix: str, value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, bool):
        out[prefix] = float(int(value))
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            out.update(flatten_numeric(child_prefix, child))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            out.update(flatten_numeric(child_prefix, child))
    return out


def _torch_cuda_metrics() -> tuple[float | None, float | None, float | None]:
    try:
        import torch
    except Exception:
        return None, None, None
    if not torch.cuda.is_available():
        return None, None, None
    util = None
    try:
        util_value = torch.cuda.utilization()
        util = float(util_value)
    except Exception:
        util = None
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return util, used / (1024 * 1024), total / (1024 * 1024)
    except Exception:
        return util, None, None


def _nvidia_smi_metrics() -> tuple[float | None, float | None, float | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None, None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None, None
    first_line = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return None, None, None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None, None, None


def _jsonable_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in payload.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _short_console_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)[:160]
    if isinstance(value, list):
        return json.dumps(value)[:160]
    return str(value)
