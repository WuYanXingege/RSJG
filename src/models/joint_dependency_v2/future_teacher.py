"""Training-only future teachers for scene and relation latent variables."""

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.models.joint_goal import canonicalize_scene_index


def future_pair_descriptor(
    future_position: torch.Tensor,
    last_position: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build the frozen six-value descriptor for canonical sparse edges.

    ``future_position`` is absolute world position ``[N,T,2]``.  Signed
    quantities use destination minus source.
    """
    if future_position.ndim != 3 or future_position.shape[-1] != 2:
        raise ValueError("future_position must have shape [N,T,2]")
    if last_position.shape != (future_position.shape[0], 2):
        raise ValueError("last_position must have shape [N,2]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return future_position.new_empty((0, 6), dtype=torch.float32)
    src, dst = edge_index.long()
    relative = future_position.float()[dst] - future_position.float()[src]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    minimum, minimum_index = distance.min(dim=1)
    time_denominator = max(future_position.shape[1] - 1, 1)
    normalized_time = minimum_index.float() / float(time_denominator)
    final_relative = relative[:, -1]
    initial_relative = last_position.float()[dst] - last_position.float()[src]
    formation_change = final_relative - initial_relative
    return torch.cat((
        minimum[:, None], normalized_time[:, None], final_relative,
        formation_change), dim=-1)


class SceneFutureTeacher(nn.Module):
    """Compute ``q(z|X,Y*)`` and the future relation posterior."""

    def __init__(self, num_scene_modes: int = 4,
                 num_relation_modes: int = 4) -> None:
        super().__init__()
        if num_scene_modes != 4 or num_relation_modes != 4:
            raise ValueError("JDV2 requires four scene/relation modes")
        self.future_projection = nn.Sequential(nn.Linear(4, 64), nn.SiLU())
        self.future_gru = nn.GRU(64, 128, batch_first=True)
        self.posterior_head = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 4))
        self.relation_teacher = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 4))

    @staticmethod
    def _scene_mean(
        features: torch.Tensor,
        compact_scene_index: torch.Tensor,
        num_scenes: int,
    ) -> torch.Tensor:
        pooled = features.new_zeros((num_scenes, features.shape[-1]))
        pooled.index_add_(0, compact_scene_index, features)
        counts = torch.bincount(
            compact_scene_index, minlength=num_scenes).to(
                device=features.device, dtype=features.dtype)
        return pooled / counts.clamp_min(1)[:, None]

    def scene_posterior(
        self,
        future_position: torch.Tensor,
        future_velocity: torch.Tensor,
        last_position: torch.Tensor,
        prior_logits: torch.Tensor,
        scene_index: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Encode exact future inputs and return posterior tensors ``[C,4]``."""
        if future_position.shape != future_velocity.shape or \
                future_position.ndim != 3 or future_position.shape[-1] != 2:
            raise ValueError("future positions/velocities must be [N,T,2]")
        if last_position.shape != (future_position.shape[0], 2):
            raise ValueError("last_position must be [N,2]")
        scene_ids, compact = canonicalize_scene_index(
            scene_index, future_position.shape[0], future_position.device)
        if prior_logits.shape != (scene_ids.numel(), 4):
            raise ValueError("prior_logits must have shape [C,4]")
        future_input = torch.cat((
            future_position - last_position[:, None], future_velocity), dim=-1)
        projected = self.future_projection(future_input)
        _, hidden = self.future_gru(projected)
        future_agent = hidden[-1]
        future_scene = self._scene_mean(
            future_agent, compact, scene_ids.numel())
        posterior = self.posterior_from_future_scene(
            future_scene, prior_logits)
        posterior.update({
            "future_agent": future_agent,
            "future_scene": future_scene,
        })
        return posterior

    def posterior_from_future_scene(
        self,
        future_scene: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Combine detached history prior with future-only evidence.

        This is the only route into ``q_phi``.  In particular, trainable
        history features are not concatenated into the evidence head, and
        posterior distillation cannot update the prior through this branch.
        """
        if future_scene.ndim != 2 or future_scene.shape[1] != 128:
            raise ValueError("future_scene must have shape [C,128]")
        if prior_logits.shape != (future_scene.shape[0], 4):
            raise ValueError("prior_logits must have shape [C,4]")
        evidence_logits = self.posterior_head(future_scene)
        centered_evidence_logits = evidence_logits - \
            evidence_logits.mean(dim=-1, keepdim=True)
        logits = prior_logits.detach().float() + \
            centered_evidence_logits.float()
        log_prob = F.log_softmax(logits.float(), dim=-1)
        return {
            "evidence_logits": evidence_logits,
            "centered_evidence_logits": centered_evidence_logits,
            "logits": logits,
            "log_prob": log_prob,
            "prob": log_prob.exp(),
        }

    def relation_posterior(
        self,
        descriptor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return endpoint-reversal-invariant ``q(r_ij|X,Y*)``.

        Reversal negates the four signed coordinate channels and leaves the
        distance/time channels intact.  Averaging shared-network logits makes
        the physical-edge posterior independent of canonical endpoint order.
        """
        if descriptor.ndim != 2 or descriptor.shape[-1] != 6:
            raise ValueError("descriptor must have shape [E,6]")
        if descriptor.shape[0] == 0:
            logits = descriptor.new_empty((0, 4))
        else:
            reverse = descriptor.clone()
            reverse[:, 2:] *= -1
            logits = 0.5 * (
                self.relation_teacher(descriptor) +
                self.relation_teacher(reverse))
        log_prob = F.log_softmax(logits.float(), dim=-1)
        return {"logits": logits, "log_prob": log_prob,
                "prob": log_prob.exp()}


__all__ = ["SceneFutureTeacher", "future_pair_descriptor"]
