"""Standalone EBJD trainer with differentiable rollout and constrained updates."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import SceneBatch
from .model import EBJDModel
from .objectives import (
    diffusion_loss, gated_geometry_mean, map_loss, relative_motion_loss,
    rollout_loss,
)
from .optimizer import (
    ActualStepAdamW, ProjectionStats, clip_global_norm, gradients, list_dot,
    list_norm, project_two_halfspaces, zeros_for,
)
from .representation import cosine_vp
from .sampling import differentiable_sample


@dataclass
class TrainerOptions:
    epochs: int = 100
    accumulation: int = 4
    new_module_lr: float = 1e-4
    unet_lr: float = 1e-5
    warmup_epochs: int = 5
    lr_final_fraction: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    noise_bins: int = 100
    coarse_v_weight: float = 0.25
    geometry_weight: float = 0.05
    map_weight: float = 0.10
    heatmap_sigma_m: float = 0.5
    supervision_edge_radius_m: float = 6.0
    supervision_edge_cpa_radius_m: float = 2.0
    supervision_edge_cpa_horizon_s: float = 4.8
    rollout_start_epoch: int = 21
    rollout_ramp_end_epoch: int = 40
    rollout_weight_start: float = 0.01
    rollout_weight_end: float = 0.20
    rollout_every: int = 4
    rollout_compensation: float = 4.0
    rollout_worlds: int = 4
    rollout_steps: int = 20
    softmin_temperature_start: float = 0.10
    softmin_temperature_end: float = 0.05
    use_rollout_loss: bool = True
    use_geometry_loss: bool = True
    use_actual_step_constraint: bool = True
    activation_checkpointing: bool = True
    precision: str = "bf16_with_fp32_geometry_loss_optimizer"


class EBJDTrainer:
    def __init__(
        self,
        model: EBJDModel,
        options: TrainerOptions | None = None,
        resolved_config: dict | None = None,
        run_identity: dict | None = None,
    ) -> None:
        self.model = model
        self.options = options or TrainerOptions()
        self.optimizer = ActualStepAdamW(
            list(model.named_parameters()), lr=self.options.new_module_lr,
            unet_lr=self.options.unet_lr, betas=self.options.betas,
            eps=self.options.optimizer_eps, weight_decay=self.options.weight_decay)
        self.parameters = self.optimizer.parameters
        self.update_index = 0
        self.resolved_config = resolved_config or {}
        self.run_identity = run_identity or {}
        wants_bf16 = self.options.precision == "bf16_with_fp32_geometry_loss_optimizer"
        self.use_bf16 = (
            wants_bf16 and next(model.parameters()).is_cuda
            and torch.cuda.is_bf16_supported())

    def _rollout_weight(self, epoch: int) -> float:
        if epoch < self.options.rollout_start_epoch:
            return 0.0
        if epoch >= self.options.rollout_ramp_end_epoch:
            return self.options.rollout_weight_end
        span = max(self.options.rollout_ramp_end_epoch - self.options.rollout_start_epoch, 1)
        fraction = (epoch - self.options.rollout_start_epoch) / span
        return self.options.rollout_weight_start + fraction * (
            self.options.rollout_weight_end - self.options.rollout_weight_start)

    def _temperature(self, epoch: int) -> float:
        if epoch <= self.options.rollout_start_epoch:
            return self.options.softmin_temperature_start
        if epoch >= self.options.rollout_ramp_end_epoch:
            return self.options.softmin_temperature_end
        span = max(self.options.rollout_ramp_end_epoch - self.options.rollout_start_epoch, 1)
        fraction = (epoch - self.options.rollout_start_epoch) / span
        return self.options.softmin_temperature_start + fraction * (
            self.options.softmin_temperature_end - self.options.softmin_temperature_start)

    def _schedule(self, epoch: int) -> None:
        if epoch <= self.options.warmup_epochs:
            factor = epoch / max(self.options.warmup_epochs, 1)
        else:
            progress = (epoch - self.options.warmup_epochs) / max(
                self.options.epochs - self.options.warmup_epochs, 1)
            final = self.options.lr_final_fraction
            factor = final + (1 - final) * 0.5 * (
                1 + math.cos(math.pi * min(progress, 1)))
        self.optimizer.lr = self.options.new_module_lr * factor
        self.optimizer.unet_lr = self.options.unet_lr * factor

    def _batch_gradients(
        self, batch: SceneBatch, epoch: int, do_rollout: bool
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
        autocast_device = "cuda" if batch.observed.is_cuda else "cpu"
        with torch.autocast(
            device_type=autocast_device, dtype=torch.bfloat16,
            enabled=self.use_bf16):
            context = self.model.encode_context(
                batch.observed, batch.semantic_maps, batch.valid)
        latent_clean = self.model.representation.encode_target(
            batch.future.detach(), context.baseline)[:, None]
        time = torch.randint(
            1, self.options.noise_bins + 1, (batch.observed.shape[0],),
            device=batch.observed.device).float() / self.options.noise_bins
        with torch.autocast(device_type=autocast_device, enabled=False):
            alpha, sigma = cosine_vp(time)
            alpha = alpha.to(latent_clean).float().view(-1, 1, 1, 1, 1)
            sigma = sigma.to(latent_clean).float().view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(latent_clean, dtype=torch.float32)
            latent_noisy = alpha * latent_clean.float() + sigma * noise
            velocity_target = alpha * noise - sigma * latent_clean.float()
        with torch.autocast(
            device_type=autocast_device, dtype=torch.bfloat16,
            enabled=self.use_bf16):
            final, coarse, coarse_trajectory, _ = self.model.denoiser(
                latent_noisy, time, context)
        with torch.autocast(device_type=autocast_device, enabled=False):
            final_clean = alpha * latent_noisy.float() - sigma * final.float()
            final_trajectory, _ = self.model.representation.decode(
                final_clean, context.baseline)
            loss_diff = diffusion_loss(
                final, coarse, velocity_target, batch.valid,
                coarse_weight=self.options.coarse_v_weight)
            loss_geo = loss_diff.new_zeros(())
            if self.options.use_geometry_loss:
                per_scene_geo = 0.5 * (
                    relative_motion_loss(
                        coarse_trajectory, batch.future, context,
                        self.options.supervision_edge_radius_m,
                        self.options.supervision_edge_cpa_radius_m,
                        self.options.supervision_edge_cpa_horizon_s)
                    + relative_motion_loss(
                        final_trajectory, batch.future, context,
                        self.options.supervision_edge_radius_m,
                        self.options.supervision_edge_cpa_radius_m,
                        self.options.supervision_edge_cpa_horizon_s))
                loss_geo = gated_geometry_mean(per_scene_geo, alpha)
            loss_map = map_loss(
                batch.future, context, self.options.heatmap_sigma_m)
            total = (
                loss_diff + self.options.geometry_weight * loss_geo
                + self.options.map_weight * loss_map)
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
                free, goals, batch.future, batch.valid, self._temperature(epoch))
            marginal_a, marginal_f, _, _ = values
            total = total + (
                self.options.rollout_compensation * self._rollout_weight(epoch)
                * loss_roll)
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
            "minimum_time": float(time.min()), "maximum_time": float(time.max()),
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
        candidate, pending = self.optimizer.propose(
            clip_global_norm(total_grad, self.options.gradient_clip_norm))
        candidate_dot_a = float(list_dot(grad_a, candidate)) if do_rollout else 0.0
        candidate_dot_f = float(list_dot(grad_f, candidate)) if do_rollout else 0.0
        stats = ProjectionStats(0, float(list_norm(candidate)), float(list_norm(candidate)), 0, False)
        decrement = candidate
        if do_rollout and self.options.use_actual_step_constraint:
            decrement, stats = project_two_halfspaces(candidate, grad_a, grad_f)
        applied_dot_a = float(list_dot(grad_a, decrement)) if do_rollout else 0.0
        applied_dot_f = float(list_dot(grad_f, decrement)) if do_rollout else 0.0
        parameter_norm_before = float(torch.sqrt(sum(
            parameter.detach().double().square().sum() for parameter in self.parameters)))
        self.optimizer.apply(decrement, pending)
        parameter_norm_after = float(torch.sqrt(sum(
            parameter.detach().double().square().sum() for parameter in self.parameters)))
        self.update_index += 1
        aggregate.update({
            "rollout_update": float(do_rollout),
            "projection_active": float(stats.active_constraints),
            "projection_changed_fraction": stats.changed_fraction,
            "candidate_dot_marginal_ade": candidate_dot_a,
            "candidate_dot_marginal_fde": candidate_dot_f,
            "applied_dot_marginal_ade": applied_dot_a,
            "applied_dot_marginal_fde": applied_dot_f,
            "candidate_decrement_norm": stats.candidate_norm,
            "applied_decrement_norm": stats.projected_norm,
            "parameter_norm_before": parameter_norm_before,
            "parameter_norm_after": parameter_norm_after,
            "learning_rate": self.optimizer.lr,
            "unet_learning_rate": self.optimizer.unet_lr,
        })
        return aggregate

    @staticmethod
    def _rng_state(loader_generator: torch.Generator | None = None) -> dict:
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state() if loader_generator else None,
        }

    @staticmethod
    def _restore_rng(state: dict, loader_generator: torch.Generator | None = None) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
        if loader_generator is not None and state.get("loader_generator") is not None:
            loader_generator.set_state(state["loader_generator"].cpu())

    def save_checkpoint(
        self, path: str | Path, epoch: int, selection_state: dict,
        loader_generator: torch.Generator | None = None, extra: dict | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "schema": "ebjd-checkpoint-v2", "epoch_boundary_only": True,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "epoch": epoch, "update_index": self.update_index,
            "resolved_config": self.resolved_config, "run_identity": self.run_identity,
            "selection_state": selection_state,
            "scales": {
                "representation_goal": float(
                    self.model.representation.goal_scale.detach().cpu().item()),
                "encoder_goal": float(self.model.encoder.goal_scale.detach().cpu().item()),
                "bridge": float(getattr(
                    self.model.representation, "bridge_scale",
                    torch.tensor(float("nan"))).detach().cpu().item()),
            },
            "rng": self._rng_state(loader_generator), "extra": extra or {},
        }, temporary)
        os.replace(temporary, path)

    def load_checkpoint(
        self, path: str | Path, loader_generator: torch.Generator | None = None,
    ) -> dict:
        payload = torch.load(
            path, map_location=next(self.model.parameters()).device, weights_only=False)
        if payload.get("schema") != "ebjd-checkpoint-v2":
            raise ValueError("resume requires an ebjd-checkpoint-v2 epoch-boundary checkpoint")
        if self.run_identity and payload.get("run_identity") != self.run_identity:
            raise ValueError("resume identity does not match checkpoint identity")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload["update_index"])
        self._restore_rng(payload["rng"], loader_generator)
        return payload
