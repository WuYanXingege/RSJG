"""Low-rank latent-mode goal distributions for groups of pedestrians.

The module in this file models

    q(G | X) = sum_z q(z | X) prod_i q(g_i | z, X)

without constructing a joint table over all ``K ** N`` goal assignments.  A
``scene_index`` vector associates every agent with its synchronous scene/window;
scene pooling is symmetric, so reordering agents does not change the model's
meaning.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def canonicalize_scene_index(
        scene_index: Optional[torch.Tensor],
        num_agents: int,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sorted scene ids and a compact agent-to-scene index.

    Parameters
    ----------
    scene_index:
        Agent-level scene/window ids with shape ``[N]``.  Ids may be sparse or
        non-contiguous.  ``None`` means that all agents belong to one scene.
    num_agents:
        Number of agents ``N``.
    device:
        Device on which the compact index should be returned.

    Returns
    -------
    scene_ids, compact_scene_index:
        ``scene_ids`` has shape ``[S]`` and ``compact_scene_index`` has shape
        ``[N]`` with values in ``[0, S)``.
    """
    if num_agents <= 0:
        raise ValueError("At least one agent is required.")

    if scene_index is None:
        scene_index = torch.zeros(num_agents, dtype=torch.long, device=device)
    else:
        if not torch.is_tensor(scene_index):
            scene_index = torch.as_tensor(scene_index, device=device)
        scene_index = scene_index.to(device=device)
        if scene_index.ndim != 1 or scene_index.numel() != num_agents:
            raise ValueError(
                "scene_index must have shape [N]; got {} for N={}.".format(
                    tuple(scene_index.shape), num_agents))
        if scene_index.dtype.is_floating_point:
            if not torch.equal(scene_index, scene_index.round()):
                raise ValueError("scene_index must contain integer-valued ids.")
        scene_index = scene_index.long()

    scene_ids, compact = torch.unique(
        scene_index, sorted=True, return_inverse=True)
    return scene_ids, compact


def sanitize_candidate_mask(
        candidate_mask: Optional[torch.Tensor],
        num_agents: int,
        num_candidates: int,
        device: torch.device,
) -> torch.Tensor:
    """Create a finite-safe ``[N, K]`` candidate mask.

    An all-invalid row is repaired by enabling its first candidate.  This gives
    malformed/no-valid-candidate inputs a deterministic finite fallback rather
    than propagating ``NaN`` from a softmax over only ``-inf`` values.
    """
    if num_candidates <= 0:
        raise ValueError("At least one goal candidate is required.")
    if candidate_mask is None:
        return torch.ones(
            num_agents, num_candidates, dtype=torch.bool, device=device)

    if not torch.is_tensor(candidate_mask):
        candidate_mask = torch.as_tensor(candidate_mask, device=device)
    candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
    if candidate_mask.shape != (num_agents, num_candidates):
        raise ValueError(
            "candidate_mask must have shape [N, K]; got {} instead of {}."
            .format(tuple(candidate_mask.shape),
                    (num_agents, num_candidates)))

    candidate_mask = candidate_mask.clone()
    empty_rows = ~candidate_mask.any(dim=-1)
    if empty_rows.any():
        candidate_mask[empty_rows, 0] = True
    return candidate_mask


def masked_log_softmax(
        logits: torch.Tensor,
        candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply log-softmax over candidates while assigning invalid goals zero mass.

    ``logits`` may have any leading dimensions as long as its first dimension
    is agents and its last dimension is candidates. ``candidate_mask`` is
    ``[N, K]`` and is broadcast across any latent-mode dimensions.
    """
    if logits.shape[0] != candidate_mask.shape[0] or \
            logits.shape[-1] != candidate_mask.shape[1]:
        raise ValueError("candidate_mask is incompatible with logits.")
    mask = candidate_mask
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    # If upstream candidate extraction supplied no finite score for a row/mode,
    # fall back to that agent's first valid candidate instead of producing NaN.
    has_finite_candidate = torch.isfinite(masked_logits).any(
        dim=-1, keepdim=True)
    if not bool(has_finite_candidate.all()):
        first_valid = candidate_mask.to(dtype=torch.long).argmax(
            dim=-1, keepdim=True)
        fallback = torch.zeros_like(candidate_mask)
        fallback.scatter_(1, first_valid, True)
        while fallback.ndim < logits.ndim:
            fallback = fallback.unsqueeze(1)
        repair = (~has_finite_candidate) & fallback.expand_as(logits)
        masked_logits = torch.where(
            repair, torch.zeros_like(masked_logits), masked_logits)
    return F.log_softmax(masked_logits, dim=-1)


class LowRankJointGoal(nn.Module):
    """Scene-mode mixture of conditionally independent agent goal marginals.

    Parameters
    ----------
    agent_dim:
        Dimension ``D`` of each social/trajectory agent feature.
    num_social_modes:
        Number of low-rank scene modes ``R``.
    hidden_dim:
        Internal feature dimension.
    goal_dim:
        Goal coordinate dimension, normally 2.
    dropout:
        Dropout used only in the conditional goal head.

    Notes
    -----
    With ``N`` agents and ``K`` candidates, the returned representation has
    ``S*R + N*R*K`` values.  It never creates the ``K ** N`` joint table.
    """

    def __init__(
            self,
            agent_dim: int,
            num_social_modes: int = 4,
            hidden_dim: int = 128,
            goal_dim: int = 2,
            dropout: float = 0.0,
            prior_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if agent_dim <= 0 or hidden_dim <= 0 or goal_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if num_social_modes <= 0:
            raise ValueError("num_social_modes must be positive.")
        if prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive.")

        self.agent_dim = int(agent_dim)
        self.num_social_modes = int(num_social_modes)
        self.hidden_dim = int(hidden_dim)
        self.goal_dim = int(goal_dim)
        self.prior_temperature = float(prior_temperature)

        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.mode_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_social_modes),
        )
        self.mode_embedding = nn.Parameter(
            torch.empty(self.num_social_modes, self.hidden_dim))
        nn.init.normal_(self.mode_embedding, mean=0.0,
                        std=self.hidden_dim ** -0.5)

        self.conditional_goal_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _scene_mean_pool(
            agent_features: torch.Tensor,
            compact_scene_index: torch.Tensor,
            num_scenes: int,
    ) -> torch.Tensor:
        """Permutation-invariant mean pool from ``[N,H]`` to ``[S,H]``."""
        pooled = agent_features.new_zeros(num_scenes, agent_features.shape[-1])
        pooled.index_add_(0, compact_scene_index, agent_features)
        counts = torch.bincount(
            compact_scene_index, minlength=num_scenes).to(
                dtype=agent_features.dtype, device=agent_features.device)
        return pooled / counts.clamp_min(1).unsqueeze(-1)

    def forward(
            self,
            agent_feat: torch.Tensor,
            goal_candidates: torch.Tensor,
            scene_index: Optional[torch.Tensor] = None,
            last_pos: Optional[torch.Tensor] = None,
            candidate_mask: Optional[torch.Tensor] = None,
            candidate_log_prior: Optional[torch.Tensor] = None,
            prior_temperature: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict scene-mode and mode-conditional goal distributions.

        Parameters
        ----------
        agent_feat:
            Social encoder output with shape ``[N, D]``.
        goal_candidates:
            Per-agent endpoint candidates with shape ``[N, K, 2]``.
        scene_index:
            Agent-to-scene/window ids, shape ``[N]``. ``None`` denotes one
            scene.
        last_pos:
            Last observed positions, shape ``[N, 2]``.  Candidate offsets from
            these positions are encoded; if omitted, candidates are used as-is.
        candidate_mask:
            Optional valid-candidate mask ``[N, K]``.
        candidate_log_prior:
            Optional GDTS heatmap candidate log-probabilities or logits with
            shape ``[N,K]``. They are added to every mode's learned residual,
            so the joint model refines rather than replaces the goal heatmap.
        prior_temperature:
            Optional per-call temperature for ``candidate_log_prior``.

        Returns
        -------
        dict
            ``mode_prob`` and ``mode_log_prob`` are ``[S, R]``;
            ``conditional_goal_prob`` and ``conditional_goal_log_prob`` are
            ``[N, R, K]``. ``compact_scene_index`` maps agents to rows in the
            scene-mode tensors.
        """
        if agent_feat.ndim != 2:
            raise ValueError("agent_feat must have shape [N, D].")
        if goal_candidates.ndim != 3:
            raise ValueError("goal_candidates must have shape [N, K, goal_dim].")
        num_agents, num_candidates, goal_dim = goal_candidates.shape
        if agent_feat.shape != (num_agents, self.agent_dim):
            raise ValueError(
                "agent_feat has shape {}; expected [{}, {}].".format(
                    tuple(agent_feat.shape), num_agents, self.agent_dim))
        if goal_dim != self.goal_dim:
            raise ValueError(
                "Expected goal_dim {}, got {}.".format(self.goal_dim, goal_dim))
        if agent_feat.device != goal_candidates.device:
            raise ValueError("agent_feat and goal_candidates must share a device.")
        if not torch.isfinite(agent_feat).all() or \
                not torch.isfinite(goal_candidates).all():
            raise ValueError("Agent features and goal candidates must be finite.")

        device = goal_candidates.device
        scene_ids, compact_scene_index = canonicalize_scene_index(
            scene_index, num_agents, device)
        valid_mask = sanitize_candidate_mask(
            candidate_mask, num_agents, num_candidates, device)

        if last_pos is None:
            relative_goals = goal_candidates
        else:
            if not torch.is_tensor(last_pos):
                last_pos = torch.as_tensor(
                    last_pos, dtype=goal_candidates.dtype, device=device)
            last_pos = last_pos.to(device=device, dtype=goal_candidates.dtype)
            if last_pos.shape != (num_agents, self.goal_dim):
                raise ValueError("last_pos must have shape [N, goal_dim].")
            if not torch.isfinite(last_pos).all():
                raise ValueError("last_pos must be finite.")
            relative_goals = goal_candidates - last_pos.unsqueeze(1)

        # agent_hidden: [N,H], goal_hidden: [N,K,H]
        agent_hidden = self.agent_encoder(agent_feat)
        goal_hidden = self.goal_encoder(relative_goals)

        # Symmetric pooling gives one latent-mode distribution per scene/window.
        scene_hidden = self._scene_mean_pool(
            agent_hidden, compact_scene_index, scene_ids.numel())
        mode_logits = self.mode_head(scene_hidden)                 # [S,R]
        mode_log_prob = F.log_softmax(mode_logits, dim=-1)         # [S,R]

        # Shared heads make the output equivariant to permutations of agents.
        agent_part = agent_hidden[:, None, None, :].expand(
            -1, self.num_social_modes, num_candidates, -1)
        goal_part = goal_hidden[:, None, :, :].expand(
            -1, self.num_social_modes, -1, -1)
        mode_part = self.mode_embedding[None, :, None, :].expand(
            num_agents, -1, num_candidates, -1)
        conditional_input = torch.cat(
            (agent_part, goal_part, mode_part), dim=-1)
        conditional_residual_logits = self.conditional_goal_head(
            conditional_input).squeeze(-1)                        # [N,R,K]

        prior_log_prob = None
        if candidate_log_prior is not None:
            if not torch.is_tensor(candidate_log_prior):
                candidate_log_prior = torch.as_tensor(
                    candidate_log_prior, dtype=goal_candidates.dtype,
                    device=device)
            candidate_log_prior = candidate_log_prior.to(
                device=device, dtype=goal_candidates.dtype)
            if candidate_log_prior.shape != (num_agents, num_candidates):
                raise ValueError(
                    "candidate_log_prior must have shape [N,K].")
            if torch.isnan(candidate_log_prior).any() or \
                    torch.isposinf(candidate_log_prior).any():
                raise ValueError("candidate_log_prior contains invalid values.")
            prior_temp = (self.prior_temperature if prior_temperature is None
                          else float(prior_temperature))
            if prior_temp <= 0:
                raise ValueError("prior_temperature must be positive.")
            # log_softmax accepts both raw logits and already-normalized log
            # probabilities. Invalid candidates retain exactly zero mass.
            prior_log_prob = masked_log_softmax(
                candidate_log_prior, valid_mask)
            conditional_logits = (
                conditional_residual_logits +
                prior_log_prob[:, None, :] / prior_temp)
        else:
            conditional_logits = conditional_residual_logits
        conditional_log_prob = masked_log_softmax(
            conditional_logits, valid_mask)                       # [N,R,K]

        output = {
            "scene_ids": scene_ids,
            "compact_scene_index": compact_scene_index,
            "candidate_mask": valid_mask,
            "mode_logits": mode_logits,
            "mode_log_prob": mode_log_prob,
            "mode_prob": mode_log_prob.exp(),
            "conditional_goal_logits": conditional_logits,
            "conditional_goal_residual_logits":
                conditional_residual_logits,
            "conditional_goal_log_prob": conditional_log_prob,
            "conditional_goal_prob": conditional_log_prob.exp(),
        }
        if prior_log_prob is not None:
            output["candidate_log_prior"] = prior_log_prob
        return output


# Descriptive aliases keep integration sites readable while retaining one
# implementation and one checkpoint namespace.
JointGoalModel = LowRankJointGoal
LowRankJointGoalModel = LowRankJointGoal
JointGoalDistribution = LowRankJointGoal


__all__ = [
    "LowRankJointGoal",
    "LowRankJointGoalModel",
    "JointGoalModel",
    "JointGoalDistribution",
    "canonicalize_scene_index",
    "sanitize_candidate_mask",
    "masked_log_softmax",
]
