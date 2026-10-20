"""Standalone EBJD trainer with differentiable rollout and constrained updates."""

from __future__ import annotations

import math
import os
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function

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
    goal_v_weight: float = 1.0
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
    # Zero preserves the legacy whole-logical-batch execution.  Formal OOM-safe
    # runs set this explicitly to one; it is an execution setting, not a change
    # to batch_scenes or gradient accumulation.
    scene_forward_chunk: int = 0
    # Execution-only speed controls.  Defaults reproduce the certified OOM
    # recovery path exactly; the optimized path opts into them explicitly.
    scene_compaction: bool = False
    physical_group_max_scenes: int = 1
    physical_group_max_agent_slots: int = 0
    physical_group_max_padding_ratio: float = 1.0
    rollout_block_checkpointing: bool = True
    # Execution-only threshold measured in padded physical agent slots.  Zero
    # preserves step checkpointing for every differentiable DDIM rollout.
    rollout_uncheckpointed_max_agent_slots: int = 0
    keep_weight: float = 0.0
    association_weight: float = 0.0
    rollout_keep_weight: float = 0.0
    pairing_seed_offset: int = 3_000_007
    lazy_off_projection_gradients: bool = False


@dataclass(frozen=True)
class BatchRandomDraws:
    """Random tensors drawn once in the original logical [B,Npad] layout."""

    time: torch.Tensor
    diffusion_noise: torch.Tensor
    rollout_initial: torch.Tensor | None

    def slice_scenes(self, start: int, stop: int) -> "BatchRandomDraws":
        return BatchRandomDraws(
            self.time[start:stop], self.diffusion_noise[start:stop],
            None if self.rollout_initial is None else self.rollout_initial[start:stop])
    def compact_scenes(
        self, valid: torch.Tensor, start: int, stop: int,
    ) -> "BatchRandomDraws":
        """Pack already-drawn tensors using the logical batch's valid mask."""
        batch = int(valid.shape[0])
        if not (valid.ndim == 2 and 0 <= start < stop <= batch):
            raise ValueError("invalid logical valid mask or scene range")
        positions = [
            valid[index].nonzero(as_tuple=False).flatten()
            for index in range(start, stop)
        ]
        if any(int(index.numel()) == 0 for index in positions):
            raise ValueError("each compacted scene must contain a valid agent")
        maximum = max(int(index.numel()) for index in positions)

        def pack(source: torch.Tensor) -> torch.Tensor:
            # Agent is dimension two for both [B,1,N,T,2] diffusion draws and
            # [B,P,N,T,2] rollout draws.
            packed = source.new_zeros(
                (stop - start, source.shape[1], maximum, *source.shape[3:]))
            for local, (scene, index) in enumerate(zip(
                range(start, stop), positions, strict=True,
            )):
                packed[local, :, :index.numel()] = source[scene].index_select(
                    1, index.to(source.device))
            return packed

        return BatchRandomDraws(
            self.time[start:stop], pack(self.diffusion_noise),
            None if self.rollout_initial is None else pack(self.rollout_initial))


def rollout_step_checkpoint_policy(
    activation_checkpointing: bool,
    physical_agent_slots: int,
    uncheckpointed_max_agent_slots: int,
) -> bool:
    """Return the immutable DDIM-step checkpoint policy for one graph."""
    if physical_agent_slots <= 0 or uncheckpointed_max_agent_slots < 0:
        raise ValueError("rollout checkpoint policy received an invalid slot budget")
    return bool(
        activation_checkpointing
        and not (
            uncheckpointed_max_agent_slots > 0
            and physical_agent_slots <= uncheckpointed_max_agent_slots))


def logical_scene_weights(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return n_s/A and 1/B weights for one legacy logical microbatch."""
    if valid.ndim != 2 or valid.shape[0] == 0:
        raise ValueError("valid must be a non-empty [B,Npad] tensor")
    counts = valid.sum(dim=1).to(dtype=torch.float64, device="cpu")
    if (counts <= 0).any():
        raise ValueError("each logical scene must contain at least one valid agent")
    return counts / counts.sum(), torch.full_like(counts, 1.0 / len(counts))


def contiguous_scene_groups(
    valid: torch.Tensor,
    max_scenes: int,
    max_agent_slots: int,
    max_padding_ratio: float,
) -> list[tuple[int, int]]:
    """Greedily form deterministic contiguous physical groups.

    A candidate group is accepted only if its locally re-padded ``B*N`` slots
    and padding ratio remain within the configured bounds.  A dense scene that
    cannot pair with its neighbour therefore falls back to a one-scene group;
    no OOM retry or data-dependent reordering is used.
    """
    if valid.ndim != 2 or valid.shape[0] == 0:
        raise ValueError("valid must be a non-empty [B,Npad] tensor")
    if max_scenes < 1 or max_agent_slots < 0 or max_padding_ratio < 1.0:
        raise ValueError("invalid physical grouping limits")
    counts = [int(value) for value in valid.sum(dim=1).tolist()]
    if any(value <= 0 for value in counts):
        raise ValueError("each logical scene must contain at least one valid agent")
    groups: list[tuple[int, int]] = []
    start = 0
    while start < len(counts):
        best_stop = start + 1
        candidate_stop = start + 2
        while candidate_stop <= len(counts) and candidate_stop - start <= max_scenes:
            candidate = counts[start:candidate_stop]
            slots = len(candidate) * max(candidate)
            ratio = slots / sum(candidate)
            if max_agent_slots and slots > max_agent_slots:
                break
            if ratio <= max_padding_ratio:
                best_stop = candidate_stop
            candidate_stop += 1
        groups.append((start, best_stop))
        start = best_stop
    return groups


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
        self._last_random_draw_sha256: str | None = None
        self._last_base_random_draw_sha256: str | None = None
        self._last_rollout_random_draw_sha256: str | None = None
        # Rollout draws are deliberately isolated from the base diffusion RNG.
        # This permits a paired ON/OFF intervention without shifting the next
        # update's time/noise draws when OFF skips the expensive rollout graph.
        self.rollout_generator = torch.Generator(device=self.device)
        self.rollout_generator.manual_seed(
            int(self.resolved_config.get("seed", 0)) + 2_000_003)
        self.pairing_generator = torch.Generator(device="cpu")
        self.pairing_generator.manual_seed(
            int(self.resolved_config.get("seed", 0))
            + int(self.options.pairing_seed_offset))

    @staticmethod
    def _random_draw_digest(draws: BatchRandomDraws) -> str:
        """Hash already-drawn tensors without consuming or changing RNG state."""
        digest = hashlib.sha256()
        for label, value in (
            ("time", draws.time),
            ("diffusion_noise", draws.diffusion_noise),
            ("rollout_initial", draws.rollout_initial),
        ):
            digest.update(label.encode("ascii"))
            if value is None:
                digest.update(b"<NONE>")
                continue
            cpu = value.detach().to(device="cpu").contiguous()
            digest.update(str(cpu.dtype).encode("ascii"))
            digest.update(str(tuple(cpu.shape)).encode("ascii"))
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _base_random_draw_digest(draws: BatchRandomDraws) -> str:
        return EBJDTrainer._random_draw_digest(BatchRandomDraws(
            draws.time, draws.diffusion_noise, None))

    @staticmethod
    def _rollout_random_draw_digest(draws: BatchRandomDraws) -> str:
        if draws.rollout_initial is None:
            return "NONE"
        digest = hashlib.sha256()
        value = draws.rollout_initial.detach().to(device="cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

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

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_batch_draws(
        self, batch: SceneBatch, do_rollout: bool,
    ) -> BatchRandomDraws:
        """Draw RNG once with the exact legacy logical B/Npad shapes and order."""
        batch_size, padded_agents = batch.observed.shape[:2]
        time = torch.randint(
            1, self.options.noise_bins + 1, (batch_size,),
            device=self.device).float() / self.options.noise_bins
        # The legacy tensor was a contiguous FP32 encode_target result with the
        # following shape.  randn_like on a matching contiguous prototype keeps
        # both random-number consumption and per-element mapping unchanged.
        prototype = torch.empty(
            batch_size, 1, padded_agents, 12, 2,
            device=self.device, dtype=torch.float32)
        diffusion_noise = torch.randn_like(prototype, dtype=torch.float32)
        rollout_initial = None
        if do_rollout:
            rollout_initial = torch.randn(
                batch_size, self.options.rollout_worlds, padded_agents, 12, 2,
                device=self.device, generator=self.rollout_generator)
        return BatchRandomDraws(time, diffusion_noise, rollout_initial)

    def _batch_gradients(
        self,
        batch: SceneBatch,
        epoch: int,
        do_rollout: bool,
        draws: BatchRandomDraws | None = None,
        agent_weight: float = 1.0,
        scene_weight: float = 1.0,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor] | None,
        list[torch.Tensor] | None, dict[str, float],
    ]:
        """Build one physical graph and return detached gradients/scalars.

        Whole-batch legacy execution uses unit weights because its objective
        functions already reduce over the logical batch.  Scene-at-a-time
        execution passes n_s/A for agent-reduced losses and 1/B for
        scene-reduced losses.
        """
        if batch.observed.device != self.device:
            raise ValueError("physical SceneBatch must be on the model device")
        autocast_device = "cuda" if batch.observed.is_cuda else "cpu"
        with record_function("ebjd.context_encoder"):
            with torch.autocast(
                device_type=autocast_device, dtype=torch.bfloat16,
                enabled=self.use_bf16):
                context = self.model.encode_context(
                    batch.observed, batch.semantic_maps, batch.valid,
                    batch.native_goals)
                base_context = context
                if getattr(self.model.representation, "anchor_residual", False):
                    base_context, _ = self.model.assign_training_anchors(
                        context, batch.future.detach())
        with record_function("ebjd.base_diffusion_and_losses"):
            latent_clean = self.model.representation.encode_target(
                batch.future.detach(), base_context.baseline,
                None if base_context.world_anchors is None
                else base_context.world_anchors[:, 0])[:, None]
            if draws is None:
                time = torch.randint(
                    1, self.options.noise_bins + 1, (batch.observed.shape[0],),
                    device=batch.observed.device).float() / self.options.noise_bins
                noise = torch.randn_like(latent_clean, dtype=torch.float32)
            else:
                expected_noise = (
                    batch.observed.shape[0], 1, batch.observed.shape[1], 12, 2)
                if tuple(draws.time.shape) != (batch.observed.shape[0],):
                    raise ValueError("prepared time draw does not match physical scene batch")
                if tuple(draws.diffusion_noise.shape) != expected_noise:
                    raise ValueError("prepared diffusion noise does not match physical scene batch")
                time = draws.time
                noise = draws.diffusion_noise
            with torch.autocast(device_type=autocast_device, enabled=False):
                alpha, sigma = cosine_vp(time)
                alpha = alpha.to(latent_clean).float().view(-1, 1, 1, 1, 1)
                sigma = sigma.to(latent_clean).float().view(-1, 1, 1, 1, 1)
                latent_noisy = alpha * latent_clean.float() + sigma * noise
                velocity_target = alpha * noise - sigma * latent_clean.float()
            with torch.autocast(
                device_type=autocast_device, dtype=torch.bfloat16,
                enabled=self.use_bf16):
                final, coarse, coarse_trajectory, _ = self.model.denoiser(
                    latent_noisy, time, base_context)
            with torch.autocast(device_type=autocast_device, enabled=False):
                final_clean = alpha * latent_noisy.float() - sigma * final.float()
                final_trajectory, _ = self.model.representation.decode(
                    final_clean, base_context.baseline,
                    base_context.world_anchors)
                loss_diff, diffusion_components = diffusion_loss(
                    final, coarse, velocity_target, batch.valid,
                    coarse_weight=self.options.coarse_v_weight,
                    goal_weight=self.options.goal_v_weight,
                    return_components=True)
                loss_geo = loss_diff.new_zeros(())
                if self.options.use_geometry_loss:
                    per_scene_geo = 0.5 * (
                        relative_motion_loss(
                            coarse_trajectory, batch.future, base_context,
                            self.options.supervision_edge_radius_m,
                            self.options.supervision_edge_cpa_radius_m,
                            self.options.supervision_edge_cpa_horizon_s)
                        + relative_motion_loss(
                            final_trajectory, batch.future, base_context,
                            self.options.supervision_edge_radius_m,
                            self.options.supervision_edge_cpa_radius_m,
                            self.options.supervision_edge_cpa_horizon_s))
                    loss_geo = gated_geometry_mean(per_scene_geo, alpha)
                loss_map = map_loss(
                    batch.future, base_context, self.options.heatmap_sigma_m)
                loss_keep = loss_diff.new_zeros(())
                loss_assoc = loss_diff.new_zeros(())
                if base_context.world_anchors is not None:
                    coarse_clean = alpha * latent_noisy.float() - sigma * coarse.float()
                    correction = torch.stack((
                        coarse_clean[..., 11, :], final_clean[..., 11, :]), dim=0)
                    keep_raw = F.smooth_l1_loss(
                        correction, torch.zeros_like(correction), reduction="none")
                    keep_agent = keep_raw.mean(dim=(0, 2, 4))
                    keep_scene = (
                        (keep_agent * batch.valid).sum(-1)
                        / batch.valid.sum(-1).clamp_min(1))
                    loss_keep = (
                        keep_scene * alpha[:, 0, 0, 0, 0].square()).mean()
                    if self.model.joint_goal_assembler is not None:
                        loss_assoc = self.model.joint_goal_assembler.association_loss(
                            context, batch.future.detach(), self.pairing_generator)
                total = (
                    float(agent_weight) * (
                        loss_diff + self.options.map_weight * loss_map)
                    + float(scene_weight) * (
                        self.options.geometry_weight * loss_geo
                        + self.options.keep_weight * loss_keep
                        + self.options.association_weight * loss_assoc))
        marginal_a = total.new_zeros(())
        marginal_f = total.new_zeros(())
        loss_roll = total.new_zeros(())
        checkpoint_steps = False
        if do_rollout:
            if draws is None:
                initial = torch.randn(
                    batch.observed.shape[0], self.options.rollout_worlds,
                    batch.observed.shape[1], 12, 2, device=batch.observed.device)
            else:
                initial = draws.rollout_initial
                expected_initial = (
                    batch.observed.shape[0], self.options.rollout_worlds,
                    batch.observed.shape[1], 12, 2)
                if initial is None or tuple(initial.shape) != expected_initial:
                    raise ValueError("prepared rollout noise does not match physical scene batch")
            with record_function("ebjd.rollout_forward"):
                rollout_context = context
                if self.model.joint_goal_assembler is not None:
                    rollout_context, _, _ = self.model.assemble_world_anchors(
                        context, batch.scene_ids, batch.agent_ids,
                        int(self.resolved_config.get("seed", 0)))
                    selections = torch.stack([
                        torch.randperm(20, generator=self.pairing_generator)[
                            :self.options.rollout_worlds]
                        for _ in range(batch.observed.shape[0])
                    ]).to(batch.observed.device)
                    rollout_context = self.model.select_world_anchors(
                        rollout_context, selections)
                physical_agent_slots = int(
                    batch.observed.shape[0] * batch.observed.shape[1])
                threshold = int(
                    self.options.rollout_uncheckpointed_max_agent_slots)
                checkpoint_steps = rollout_step_checkpoint_policy(
                    self.options.activation_checkpointing,
                    physical_agent_slots, threshold)
                with torch.autocast(
                    device_type=autocast_device, dtype=torch.bfloat16,
                    enabled=self.use_bf16):
                    free, goals, _ = differentiable_sample(
                        self.model, rollout_context, initial, self.options.rollout_steps,
                        checkpoint_steps=checkpoint_steps,
                        checkpoint_blocks=self.options.rollout_block_checkpointing)
                loss_roll, values = rollout_loss(
                    free, goals, batch.future, batch.valid, self._temperature(epoch))
                loss_roll_keep = loss_roll.new_zeros(())
                if rollout_context.world_anchors is not None:
                    correction = (
                        (goals.float() - rollout_context.world_anchors.float())
                        / self.model.representation.goal_scale.float().clamp_min(0.5)
                    )
                    loss_roll_keep = F.smooth_l1_loss(
                        correction, torch.zeros_like(correction))
                    loss_roll = loss_roll + (
                        self.options.rollout_keep_weight * loss_roll_keep)
            marginal_a, marginal_f, _, _ = values
            total = total + (
                float(scene_weight) * self.options.rollout_compensation
                * self._rollout_weight(epoch) * loss_roll)
            with record_function("ebjd.grad_marginal_ade"):
                grad_a = gradients(
                    float(scene_weight) * marginal_a, self.parameters,
                    retain_graph=True)
            with record_function("ebjd.grad_marginal_fde"):
                grad_f = gradients(
                    float(scene_weight) * marginal_f, self.parameters,
                    retain_graph=True)
        else:
            # OFF updates have no projection consumer.  Avoid allocating two
            # complete all-zero gradient trees for every physical group.
            grad_a = (
                None if self.options.lazy_off_projection_gradients
                else zeros_for(self.parameters))
            grad_f = (
                None if self.options.lazy_off_projection_gradients
                else zeros_for(self.parameters))
        with record_function("ebjd.grad_total"):
            total_gradient = gradients(total, self.parameters)
        logs = {
            "loss": float(total.detach()), "diffusion": float(loss_diff.detach()),
            "weighted_diffusion": float(loss_diff.detach()),
            "diffusion_goal_fine_unweighted": float(
                diffusion_components["goal_fine_unweighted"].detach()),
            "diffusion_bridge_fine_unweighted": float(
                diffusion_components["bridge_fine_unweighted"].detach()),
            "diffusion_goal_coarse_unweighted": float(
                diffusion_components["goal_coarse_unweighted"].detach()),
            "diffusion_bridge_coarse_unweighted": float(
                diffusion_components["bridge_coarse_unweighted"].detach()),
            "active_goal_v_weight": float(self.options.goal_v_weight),
            "active_rollout_weight": float(self._rollout_weight(epoch)),
            "rollout_frequency_compensation": float(
                self.options.rollout_compensation),
            "effective_rollout_coefficient": float(
                self.options.rollout_compensation * self._rollout_weight(epoch)),
            "geometry": float(loss_geo.detach()), "map": float(loss_map.detach()),
            "keep": float(loss_keep.detach()),
            "association": float(loss_assoc.detach()),
            "rollout": float(loss_roll.detach()),
            "rollout_keep": float(loss_roll_keep.detach()) if do_rollout else 0.0,
            "soft_marginal_ade": float(marginal_a.detach()),
            "soft_marginal_fde": float(marginal_f.detach()),
            "minimum_time": float(time.min()), "maximum_time": float(time.max()),
            "rollout_step_checkpointing": float(
                do_rollout and checkpoint_steps),
        }
        return total_gradient, grad_a, grad_f, logs

    @staticmethod
    def _add_gradients_(target: list[torch.Tensor], source: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for accumulated, value in zip(target, source, strict=True):
                accumulated.add_(value)

    def _scene_chunked_batch_gradients(
        self, logical_batch: SceneBatch, epoch: int, do_rollout: bool,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor] | None,
        list[torch.Tensor] | None, dict[str, float],
    ]:
        """Evaluate one logical batch as bounded contiguous physical groups."""
        if self.options.scene_forward_chunk != 1:
            raise ValueError(
                "the memory-safe implementation requires scene_forward_chunk=1")
        batch_size = int(logical_batch.observed.shape[0])
        agent_weights, scene_weights = logical_scene_weights(logical_batch.valid)
        draws = self._prepare_batch_draws(logical_batch, do_rollout)
        self._last_random_draw_sha256 = self._random_draw_digest(draws)
        self._last_base_random_draw_sha256 = self._base_random_draw_digest(draws)
        self._last_rollout_random_draw_sha256 = self._rollout_random_draw_digest(draws)
        if self.options.scene_compaction:
            groups = contiguous_scene_groups(
                logical_batch.valid,
                self.options.physical_group_max_scenes,
                self.options.physical_group_max_agent_slots,
                self.options.physical_group_max_padding_ratio)
        else:
            if self.options.physical_group_max_scenes != 1:
                raise ValueError("physical grouping requires scene_compaction=true")
            groups = [(index, index + 1) for index in range(batch_size)]
        total_grad = zeros_for(self.parameters)
        materialize_off = not self.options.lazy_off_projection_gradients
        grad_a = zeros_for(self.parameters) if do_rollout or materialize_off else None
        grad_f = zeros_for(self.parameters) if do_rollout or materialize_off else None
        aggregate = {
            "loss": 0.0, "diffusion": 0.0, "geometry": 0.0, "map": 0.0,
            "keep": 0.0, "association": 0.0, "rollout_keep": 0.0,
            "weighted_diffusion": 0.0,
            "diffusion_goal_fine_unweighted": 0.0,
            "diffusion_bridge_fine_unweighted": 0.0,
            "diffusion_goal_coarse_unweighted": 0.0,
            "diffusion_bridge_coarse_unweighted": 0.0,
            "active_goal_v_weight": 0.0,
            "active_rollout_weight": 0.0,
            "rollout_frequency_compensation": 0.0,
            "effective_rollout_coefficient": 0.0,
            "rollout": 0.0, "soft_marginal_ade": 0.0,
            "soft_marginal_fde": 0.0,
        }
        minimum_time = float("inf")
        maximum_time = -float("inf")
        maximum_group_scenes = 0
        maximum_group_slots = 0
        checkpointed_rollout_groups = 0
        for start, stop in groups:
            agent_weight = float(agent_weights[start:stop].sum())
            scene_weight = float(scene_weights[start:stop].sum())
            if self.options.scene_compaction:
                physical_batch = logical_batch.compact_scenes(start, stop).to(self.device)
                physical_draws = draws.compact_scenes(
                    logical_batch.valid, start, stop)
            else:
                physical_batch = logical_batch.slice_scenes(start, stop).to(self.device)
                physical_draws = draws.slice_scenes(start, stop)
            maximum_group_scenes = max(maximum_group_scenes, stop - start)
            maximum_group_slots = max(
                maximum_group_slots,
                int(physical_batch.valid.numel()))
            gradients_total, gradients_a, gradients_f, logs = self._batch_gradients(
                physical_batch, epoch, do_rollout, physical_draws,
                agent_weight=agent_weight, scene_weight=scene_weight)
            with record_function("ebjd.gradient_accumulation"):
                self._add_gradients_(total_grad, gradients_total)
                if do_rollout or materialize_off:
                    if grad_a is None or grad_f is None or \
                            gradients_a is None or gradients_f is None:
                        raise AssertionError("rollout gradients are missing")
                    self._add_gradients_(grad_a, gradients_a)
                    self._add_gradients_(grad_f, gradients_f)
            aggregate["loss"] += logs["loss"]
            for name in (
                "diffusion", "weighted_diffusion",
                "diffusion_goal_fine_unweighted",
                "diffusion_bridge_fine_unweighted",
                "diffusion_goal_coarse_unweighted",
                "diffusion_bridge_coarse_unweighted",
                "active_goal_v_weight",
                "active_rollout_weight",
                "rollout_frequency_compensation",
                "effective_rollout_coefficient",
            ):
                aggregate[name] += agent_weight * logs[name]
            aggregate["map"] += agent_weight * logs["map"]
            for name in (
                "geometry", "keep", "association", "rollout",
                "rollout_keep", "soft_marginal_ade",
                "soft_marginal_fde",
            ):
                aggregate[name] += scene_weight * logs[name]
            minimum_time = min(minimum_time, logs["minimum_time"])
            maximum_time = max(maximum_time, logs["maximum_time"])
            checkpointed_rollout_groups += int(
                logs["rollout_step_checkpointing"])
            del (
                physical_batch, physical_draws, gradients_total,
                gradients_a, gradients_f, logs)
        aggregate["minimum_time"] = minimum_time
        aggregate["maximum_time"] = maximum_time
        aggregate["physical_group_count"] = float(len(groups))
        aggregate["maximum_physical_group_scenes"] = float(maximum_group_scenes)
        aggregate["maximum_physical_agent_slots"] = float(maximum_group_slots)
        aggregate["rollout_step_checkpointed_group_fraction"] = (
            float(checkpointed_rollout_groups) / len(groups) if do_rollout else 0.0)
        return total_grad, grad_a, grad_f, aggregate

    def train_step(self, microbatches: list[SceneBatch], epoch: int) -> dict[str, float]:
        if not microbatches:
            raise ValueError("train_step needs at least one microbatch")
        self._schedule(epoch)
        do_rollout = (
            self.options.use_rollout_loss and epoch >= self.options.rollout_start_epoch
            and self.update_index % self.options.rollout_every == 0)
        total_grad = zeros_for(self.parameters)
        materialize_off = not self.options.lazy_off_projection_gradients
        grad_a = zeros_for(self.parameters) if do_rollout or materialize_off else None
        grad_f = zeros_for(self.parameters) if do_rollout or materialize_off else None
        aggregate: dict[str, float] = {}
        input_digest = hashlib.sha256()
        augmentation_digest = hashlib.sha256()
        random_draw_digest = hashlib.sha256()
        base_random_draw_digest = hashlib.sha256()
        rollout_random_draw_digest = hashlib.sha256()
        scale = 1 / len(microbatches)
        for batch in microbatches:
            input_digest.update("\0".join(batch.scene_ids).encode("utf-8"))
            for identities in batch.agent_ids:
                input_digest.update("\0".join(identities).encode("utf-8"))
            augmentation_digest.update(
                (batch.augmentation_sha256 or "NONE").encode("ascii"))
            if self.options.scene_forward_chunk:
                gradients_total, gradients_a, gradients_f, logs = (
                    self._scene_chunked_batch_gradients(batch, epoch, do_rollout))
                if self._last_random_draw_sha256 is None:
                    raise AssertionError("formal scene-chunked draw digest is missing")
                random_draw_digest.update(bytes.fromhex(
                    self._last_random_draw_sha256))
                if self._last_base_random_draw_sha256 is None or \
                        self._last_rollout_random_draw_sha256 is None:
                    raise AssertionError("paired draw digests are missing")
                base_random_draw_digest.update(bytes.fromhex(
                    self._last_base_random_draw_sha256))
                rollout_random_draw_digest.update(
                    b"NONE" if self._last_rollout_random_draw_sha256 == "NONE"
                    else bytes.fromhex(self._last_rollout_random_draw_sha256))
            else:
                physical_batch = (
                    batch if batch.observed.device == self.device else batch.to(self.device))
                physical_draws = self._prepare_batch_draws(
                    physical_batch, do_rollout)
                random_draw_digest.update(bytes.fromhex(
                    self._random_draw_digest(physical_draws)))
                base_random_draw_digest.update(bytes.fromhex(
                    self._base_random_draw_digest(physical_draws)))
                rollout_digest = self._rollout_random_draw_digest(physical_draws)
                rollout_random_draw_digest.update(
                    b"NONE" if rollout_digest == "NONE"
                    else bytes.fromhex(rollout_digest))
                gradients_total, gradients_a, gradients_f, logs = self._batch_gradients(
                    physical_batch, epoch, do_rollout, physical_draws)
            total_grad = [a + scale * b for a, b in zip(total_grad, gradients_total, strict=True)]
            if do_rollout or materialize_off:
                if grad_a is None or grad_f is None or \
                        gradients_a is None or gradients_f is None:
                    raise AssertionError("rollout gradients are missing")
                grad_a = [
                    a + scale * b
                    for a, b in zip(grad_a, gradients_a, strict=True)]
                grad_f = [
                    a + scale * b
                    for a, b in zip(grad_f, gradients_f, strict=True)]
            for name, value in logs.items():
                aggregate[name] = aggregate.get(name, 0) + scale * value
            del gradients_total, gradients_a, gradients_f, logs
        with record_function("ebjd.adamw_proposal"):
            total_gradient_norm_before_clip = float(list_norm(total_grad))
            gate_gradient = 0.0
            gate_name = "denoiser.native_goal_conditioner.gate"
            parameter_names = [name for name, _ in self.optimizer.named_parameters]
            if gate_name in parameter_names:
                gate_gradient = float(total_grad[parameter_names.index(gate_name)].float())
            candidate, pending = self.optimizer.propose(
                clip_global_norm(total_grad, self.options.gradient_clip_norm))
        if do_rollout and (grad_a is None or grad_f is None):
            raise AssertionError("rollout gradients are missing before projection")
        candidate_dot_a = float(list_dot(grad_a, candidate)) if do_rollout else 0.0
        candidate_dot_f = float(list_dot(grad_f, candidate)) if do_rollout else 0.0
        stats = ProjectionStats(0, float(list_norm(candidate)), float(list_norm(candidate)), 0, False)
        decrement = candidate
        if do_rollout and self.options.use_actual_step_constraint:
            with record_function("ebjd.fp64_qp_projection"):
                decrement, stats = project_two_halfspaces(candidate, grad_a, grad_f)
        applied_dot_a = float(list_dot(grad_a, decrement)) if do_rollout else 0.0
        applied_dot_f = float(list_dot(grad_f, decrement)) if do_rollout else 0.0
        parameter_norm_before = float(torch.sqrt(sum(
            parameter.detach().double().square().sum() for parameter in self.parameters)))
        with record_function("ebjd.optimizer_apply"):
            self.optimizer.apply(decrement, pending)
        parameter_norm_after = float(torch.sqrt(sum(
            parameter.detach().double().square().sum() for parameter in self.parameters)))
        self.update_index += 1
        aggregate.update({
            "rollout_update": float(do_rollout),
            "projection_executed": float(
                do_rollout and self.options.use_actual_step_constraint),
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
            "total_gradient_norm_before_clip": total_gradient_norm_before_clip,
            "logical_scene_id_sha256": input_digest.hexdigest(),
            "logical_scene_agent_identity_sha256": input_digest.hexdigest(),
            "augmentation_sha256": augmentation_digest.hexdigest(),
            "native_goal_gate_gradient": gate_gradient,
            "random_draw_sha256": random_draw_digest.hexdigest(),
            "base_random_draw_sha256": base_random_draw_digest.hexdigest(),
            "rollout_random_draw_sha256": rollout_random_draw_digest.hexdigest()
            if do_rollout else "NONE",
            "scene_forward_chunk": float(self.options.scene_forward_chunk),
            "scene_compaction": float(self.options.scene_compaction),
            "physical_group_max_scenes": float(
                self.options.physical_group_max_scenes),
            "rollout_block_checkpointing": float(
                self.options.rollout_block_checkpointing),
        })
        conditioner = self.model.denoiser.native_goal_conditioner
        aggregate["native_goal_gate"] = (
            float(conditioner.gate.detach().float().cpu())
            if conditioner is not None else 0.0)
        aggregate["native_goal_condition_norm_ratio"] = (
            float(conditioner.last_condition_norm_ratio or 0.0)
            if conditioner is not None else 0.0)
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
            "rollout_rng": self.rollout_generator.get_state().cpu(),
            "pairing_rng": self.pairing_generator.get_state().cpu(),
        }, temporary)
        os.replace(temporary, path)

    def load_checkpoint(
        self, path: str | Path, loader_generator: torch.Generator | None = None,
    ) -> dict:
        payload = torch.load(
            path, map_location="cpu", weights_only=False)
        self.restore_checkpoint_payload(
            payload, loader_generator=loader_generator, require_identity=True)
        return payload

    def restore_checkpoint_payload(
        self,
        payload: dict,
        loader_generator: torch.Generator | None = None,
        require_identity: bool = True,
    ) -> None:
        """Restore a validated v2 payload, optionally after controlled migration."""
        if payload.get("schema") != "ebjd-checkpoint-v2":
            raise ValueError("resume requires an ebjd-checkpoint-v2 epoch-boundary checkpoint")
        if payload.get("epoch_boundary_only") is not True:
            raise ValueError("resume requires an epoch-boundary checkpoint")
        if require_identity and self.run_identity and payload.get("run_identity") != self.run_identity:
            raise ValueError("resume identity does not match checkpoint identity")
        current_state = self.model.state_dict()
        saved_state = payload.get("model")
        if not isinstance(saved_state, dict) or list(saved_state) != list(current_state):
            raise ValueError("model state names/order differ from checkpoint")
        for name, current in current_state.items():
            saved = saved_state[name]
            if not torch.is_tensor(saved) or saved.shape != current.shape:
                raise ValueError(f"model state shape differs for {name}")
        parameter_names = [name for name, _ in self.model.named_parameters()]
        parameter_name_set = set(parameter_names)
        saved_parameter_order = [
            name for name in saved_state if name in parameter_name_set]
        if saved_parameter_order != parameter_names:
            raise ValueError("model parameter names/order differ from checkpoint")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_index = int(payload["update_index"])
        self._restore_rng(payload["rng"], loader_generator)
        if payload.get("rollout_rng") is not None:
            self.rollout_generator.set_state(payload["rollout_rng"].cpu())
        if payload.get("pairing_rng") is not None:
            self.pairing_generator.set_state(payload["pairing_rng"].cpu())

    def restore_model_extension_payload(
        self, payload: dict, loader_generator: torch.Generator | None = None,
        prefix: str = "denoiser.native_goal_conditioner.",
    ) -> dict:
        """Restore a legacy checkpoint by name, leaving only new-module state empty."""
        if payload.get("schema") != "ebjd-checkpoint-v2" or \
                payload.get("epoch_boundary_only") is not True:
            raise ValueError("model extension requires an epoch-boundary v2 checkpoint")
        current = self.model.state_dict()
        saved = payload.get("model")
        if not isinstance(saved, dict):
            raise ValueError("model extension checkpoint lacks model state")
        old_names = [name for name in current if not name.startswith(prefix)]
        new_names = [name for name in current if name.startswith(prefix)]
        if list(saved) != old_names or not new_names:
            raise ValueError("model extension differs by more than the new conditioner")
        for name in old_names:
            if saved[name].shape != current[name].shape:
                raise ValueError(f"model extension shape mismatch: {name}")
        missing, unexpected = self.model.load_state_dict(saved, strict=False)
        if list(missing) != new_names or unexpected:
            raise ValueError("model extension load produced unexpected state differences")

        optimizer = payload.get("optimizer", {})
        saved_names = list(optimizer.get("parameter_names", []))
        saved_states = list(optimizer.get("states", []))
        if len(saved_names) != len(saved_states):
            raise ValueError("origin optimizer name/state lengths differ")
        by_name = dict(zip(saved_names, saved_states, strict=True))
        current_names = [name for name, _ in self.optimizer.named_parameters]
        if [name for name in current_names if not name.startswith(prefix)] != saved_names:
            raise ValueError("origin optimizer parameters do not match legacy model")
        extended_optimizer = {
            "lr": optimizer["lr"], "unet_lr": optimizer["unet_lr"],
            "parameter_names": current_names,
            "states": [by_name.get(name, {}) for name in current_names],
        }
        self.optimizer.load_state_dict(extended_optimizer)
        self.update_index = int(payload["update_index"])
        self._restore_rng(payload["rng"], loader_generator)
        if payload.get("rollout_rng") is not None:
            self.rollout_generator.set_state(payload["rollout_rng"].cpu())
        if payload.get("pairing_rng") is not None:
            self.pairing_generator.set_state(payload["pairing_rng"].cpu())
        exact = all(torch.equal(
            self.model.state_dict()[name].detach().cpu(), saved[name].detach().cpu())
            for name in old_names)
        if not exact:
            raise ValueError("legacy tensors changed during model-extension migration")
        return {
            "old_model_state_exact": True,
            "old_optimizer_state_by_name_exact": True,
            "new_model_state_names": new_names,
            "new_optimizer_state_empty": all(
                extended_optimizer["states"][current_names.index(name)] == {}
                for name in current_names if name.startswith(prefix)),
            "rng_state_restored": True,
            "update_index_restored": self.update_index,
        }

    def restore_named_extension_payload(
        self, payload: dict, loader_generator: torch.Generator | None = None,
    ) -> dict:
        """Warm-start a declared V2 extension by exact shared state names.

        Shared parameters and optimizer moments are restored byte-for-byte;
        parameters absent from the parent receive empty optimizer state.  A
        missing or shape-changed parent tensor is never silently accepted.
        """
        if payload.get("schema") != "ebjd-checkpoint-v2" or \
                payload.get("epoch_boundary_only") is not True:
            raise ValueError("V2 extension requires an epoch-boundary v2 checkpoint")
        current = self.model.state_dict()
        saved = payload.get("model")
        if not isinstance(saved, dict) or not set(saved).issubset(current):
            raise ValueError("V2 extension parent state is not a subset of the model")
        for name, value in saved.items():
            if value.shape != current[name].shape:
                raise ValueError(f"V2 shared state shape mismatch: {name}")
        missing, unexpected = self.model.load_state_dict(saved, strict=False)
        expected_new = [name for name in current if name not in saved]
        if list(missing) != expected_new or unexpected:
            raise ValueError("V2 extension load produced undeclared state differences")

        optimizer = payload.get("optimizer", {})
        saved_names = list(optimizer.get("parameter_names", []))
        saved_states = list(optimizer.get("states", []))
        if len(saved_names) != len(saved_states):
            raise ValueError("V2 parent optimizer state is malformed")
        by_name = dict(zip(saved_names, saved_states, strict=True))
        current_names = [name for name, _ in self.optimizer.named_parameters]
        if [name for name in current_names if name in by_name] != saved_names:
            raise ValueError("V2 shared optimizer parameter ordering differs")
        unknown = set(saved_names) - set(current_names)
        if unknown:
            raise ValueError(f"V2 parent optimizer has unknown parameters: {sorted(unknown)}")
        extended = {
            "lr": optimizer["lr"], "unet_lr": optimizer["unet_lr"],
            "parameter_names": current_names,
            "states": [by_name.get(name, {}) for name in current_names],
        }
        self.optimizer.load_state_dict(extended)
        self.update_index = int(payload["update_index"])
        self._restore_rng(payload["rng"], loader_generator)
        if payload.get("rollout_rng") is not None:
            self.rollout_generator.set_state(payload["rollout_rng"].cpu())
        shared_exact = all(torch.equal(
            self.model.state_dict()[name].detach().cpu(), value.detach().cpu())
            for name, value in saved.items())
        if not shared_exact:
            raise ValueError("V2 shared tensors changed during named migration")
        new_parameters = [name for name in current_names if name not in by_name]
        return {
            "shared_model_state_exact": True,
            "shared_optimizer_state_by_name_exact": True,
            "new_model_state_names": expected_new,
            "new_parameter_names": new_parameters,
            "new_optimizer_state_empty": all(
                extended["states"][current_names.index(name)] == {}
                for name in new_parameters),
            "rng_state_restored": True,
            "pairing_rng_initialized_from_frozen_offset": True,
            "update_index_restored": self.update_index,
        }
