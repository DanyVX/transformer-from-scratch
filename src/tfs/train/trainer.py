"""Reproducible CPU/GPU training with exact sampler and RNG resume."""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tfs.config import ExperimentConfig
from tfs.data.shards import WindowSampler
from tfs.model.gpt import GPT
from tfs.provenance import write_result
from tfs.train.checkpoint import (
    capture_rng,
    config_fingerprint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from tfs.train.optim import configure_adamw
from tfs.train.schedule import learning_rate


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """M5 trainer; config and sample stream are part of the resume contract."""

    def __init__(
        self, model: GPT, train_sampler: WindowSampler,
        config: ExperimentConfig, device: str = "cpu",
        log_path: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.config = config
        self.fingerprint = config_fingerprint(config.model_dump(mode="json"))
        self.model = model.to(self.device)
        self.sampler = train_sampler
        self.optimizer = configure_adamw(model, config.train.learning_rate)
        self.step = 0
        self.losses: list[float] = []
        self.log_path = log_path

    def train_steps(self, steps: int, *, checkpoint: Path | None = None) -> list[float]:
        if steps < 0 or self.step + steps > self.config.train.max_steps:
            raise ValueError("Steps exceed scheduled training")
        self.model.train()
        start = time.perf_counter()
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(
                self.step, self.config.train.max_steps, self.config.train.warmup_steps,
                self.config.train.learning_rate, self.config.train.min_learning_rate
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            total_loss = 0.0
            for _ in range(self.config.train.gradient_accumulation):
                x, y = self.sampler.next_batch()
                ids = torch.from_numpy(x).long().to(self.device)
                targets = torch.from_numpy(y).long().to(self.device)
                out = self.model(ids, targets)
                if out.loss is None or not torch.isfinite(out.loss).all():
                    if checkpoint is not None:
                        self.save(checkpoint)
                    raise FloatingPointError("Nonfinite training loss; last-good state saved")
                (out.loss / self.config.train.gradient_accumulation).backward()
                total_loss += out.loss.item() / self.config.train.gradient_accumulation
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                if checkpoint is not None:
                    self.save(checkpoint)
                raise FloatingPointError("Nonfinite gradients; last-good state saved")
            self.optimizer.step()
            self.step += 1
            self.losses.append(total_loss)
            if self.log_path is not None:
                self._append_log(total_loss, lr, float(grad_norm))
        elapsed = time.perf_counter() - start
        if self.log_path is not None and steps:
            write_result(
                self.log_path.with_suffix(".json"),
                {"steps_completed": self.step, "last_loss": self.losses[-1],
                 "tokens_per_second": steps * self.config.train.batch_size *
                 self.config.train.gradient_accumulation * self.config.model.block_size /
                 max(elapsed, 1e-9)},
                config=self.config.model_dump(mode="json"),
            )
        return self.losses

    def _append_log(self, loss: float, rate: float, norm: float) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["step", "loss", "learning_rate", "grad_norm"])
            writer.writerow([self.step, loss, rate, norm])

    def save(self, path: Path) -> None:
        save_checkpoint(path, {
            "fingerprint": self.fingerprint,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sampler": self.sampler.state_dict(),
            "rng": capture_rng(),
            "losses": self.losses.copy(),
        })

    def resume(self, path: Path) -> None:
        state = load_checkpoint(path, self.fingerprint)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.load_state_dict(state["sampler"])
        self.step = int(state["step"])
        self.losses = list(state["losses"])
        restore_rng(state["rng"])
