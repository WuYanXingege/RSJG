"""Standalone EBJD trainer with differentiable rollout and constrained updates."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import SceneBatch
from .model import EBJDModel
from .objectives import diffusion_loss, map_loss, relative_motion_loss, rollout_loss
from .optimizer import (
    ActualStepAdamW, ProjectionStats, clip_global_norm, gradients,
    project_two_halfspaces, zeros_for,
)
from .representation import cosine_vp
from .sampling import differentiable_sample


@dataclass
class TrainerOptions:
    epochs: int = 100
    accumulation: int = 4
    rollout_start_epoch: int = 21
    rollout_every: int = 4
    rollout_worlds: int = 4
    rollout_steps: int = 20
    use_rollout_loss: bool = True
    use_geometry_loss: bool = True
    use_actual_step_constraint: bool = True
    activation_checkpointing: bool = True


def rollout_weight(epoch: int) -> float:
    if epoch < 21:
        return 0.0
    if epoch >= 40:
        return 0.20
    return 0.01 + (0.20 - 0.01) * (epoch - 21) / 19


def softmin_temperature(epoch: int) -> float:
    if epoch <= 21:
        return 0.10
    if epoch >= 40:
        return 0.05
    return 0.10 + (0.05 - 0.10) * (epoch - 21) / 19


class EBJDTrainer:
    def __init__(self, model: EBJDModel, options: TrainerOptions | None = None) -> None:
        self.model = model
        self.options = options or TrainerOptions()
        self.optimizer = ActualStepAdamW(list(model.named_parameters()))
        self.parameters = self.optimizer.parameters
        self.update_index = 0
        self.use_bf16 = next(model.parameters()).is_cuda and torch.cuda.is_bf16_supported()

    def _schedule(self, epoch: int) -> None:
        if epoch <= 5:
            factor = epoch / 5
        else:
            progress = (epoch - 5) / max(self.options.epochs - 5, 1)
            factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
        self.optimizer.lr = 1e-4 * factor
        self.optimizer.unet_lr = 1e-5 * factor

    def _batch_gradients(
        self, batch: SceneBatch, epoch: int, do_rollout: bool
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
        autocast_device = "cuda" if batch.observed.is_cuda else "cpu"
        with torch.autocast(
            device_type=autocast_device, dtype=torch.bfloat16,
            enabled=self.use_bf16):
            context = self.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        latent_clean = self.model.representation.encode_target(
            batch.future.detach(), context.baseline)[:, None]
        time = torch.randint(1, 101, (batch.observed.shape[0],), device=batch.observed.device).float() / 100
        alpha, sigma = cosine_vp(time)
        alpha = alpha.to(latent_clean).view(-1, 1, 1, 1, 1)
        sigma = sigma.to(latent_clean).view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(latent_clean)
        latent_noisy = alpha * latent_clean + sigma * noise
        velocity_target = alpha * noise - sigma * latent_clean
        with torch.autocast(
            device_type=autocast_device, dtype=torch.bfloat16,
            enabled=self.use_bf16):
            final, coarse, coarse_trajectory, _ = self.model.denoiser(
                latent_noisy, time, context)
        final_clean = alpha * latent_noisy - sigma * final.float()
        final_trajectory, _ = self.model.representation.decode(final_clean, context.baseline)
        loss_diff = diffusion_loss(final, coarse, velocity_target, batch.valid)
        loss_geo = loss_diff.new_zeros(())
        if self.options.use_geometry_loss:
            loss_geo = alpha.square().mean() * 0.5 * (
                relative_motion_loss(coarse_trajectory, batch.future, context)
                + relative_motion_loss(final_trajectory, batch.future, context))
        loss_map = map_loss(batch.future, context)
        total = loss_diff + 0.05 * loss_geo + 0.10 * loss_map
        marginal_a = total.new_zeros(())
        marginal_f = total.new_zeros(())
        loss_roll = total.new_zeros(())
        if do_rollout:
            initial = torch.randn(
                batch.observed.shape[0], self.options.rollout_worlds,
                batch.observed.shape[1], 12, 2, device=batch.observed.device)
            with torch.autocast(
                device_type=autocast_device, dtype=torch.bfloat16,
                enabled=self.use_bf16):
                free, goals, _ = differentiable_sample(
                    self.model, context, initial, self.options.rollout_steps,
                    checkpoint_steps=self.options.activation_checkpointing)
            loss_roll, values = rollout_loss(
                free, goals, batch.future, batch.valid, softmin_temperature(epoch))
            marginal_a, marginal_f, _, _ = values
            total = total + 4 * rollout_weight(epoch) * loss_roll
            grad_a = gradients(marginal_a, self.parameters, retain_graph=True)
            grad_f = gradients(marginal_f, self.parameters, retain_graph=True)
        else:
            grad_a = zeros_for(self.parameters)
            grad_f = zeros_for(self.parameters)
        total_gradient = gradients(total, self.parameters)
        logs = {
            "loss": float(total.detach()), "diffusion": float(loss_diff.detach()),
            "geometry": float(loss_geo.detach()), "map": float(loss_map.detach()),
            "rollout": float(loss_roll.detach()),
            "soft_marginal_ade": float(marginal_a.detach()),
            "soft_marginal_fde": float(marginal_f.detach()),
        }
        return total_gradient, grad_a, grad_f, logs

    def train_step(self, microbatches: list[SceneBatch], epoch: int) -> dict[str, float]:
        if not microbatches:
            raise ValueError("train_step needs at least one microbatch")
        self._schedule(epoch)
        do_rollout = (
            self.options.use_rollout_loss and epoch >= self.options.rollout_start_epoch
            and self.update_index % self.options.rollout_every == 0)
        total_grad = zeros_for(self.parameters)
        grad_a = zeros_for(self.parameters)
        grad_f = zeros_for(self.parameters)
        aggregate: dict[str, float] = {}
        scale = 1 / len(microbatches)
        for batch in microbatches:
            gradients_total, gradients_a, gradients_f, logs = self._batch_gradients(
                batch, epoch, do_rollout)
            total_grad = [a + scale * b for a, b in zip(total_grad, gradients_total, strict=True)]
            grad_a = [a + scale * b for a, b in zip(grad_a, gradients_a, strict=True)]
            grad_f = [a + scale * b for a, b in zip(grad_f, gradients_f, strict=True)]
            for name, value in logs.items():
                aggregate[name] = aggregate.get(name, 0) + scale * value
        candidate, pending = self.optimizer.propose(clip_global_norm(total_grad, 5.0))
        stats = ProjectionStats(0, 0, 0, 0, False)
        if do_rollout and self.options.use_actual_step_constraint:
            candidate, stats = project_two_halfspaces(candidate, grad_a, grad_f)
        self.optimizer.apply(candidate, pending)
        self.update_index += 1
        aggregate.update({
            "rollout_update": float(do_rollout),
            "projection_active": float(stats.active_constraints),
            "projection_changed_fraction": stats.changed_fraction,
            "learning_rate": self.optimizer.lr,
        })
        return aggregate

    def save_checkpoint(self, path: str | Path, epoch: int, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "epoch": epoch, "update_index": self.update_index, "extra": extra or {},
        }, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=next(self.model.parameters()).device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload["update_index"])
        return payload
