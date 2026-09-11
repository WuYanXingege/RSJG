"""Relation-aware graph-wide coupling of fixed marginal trajectory banks.

This is the V4 orchestration layer.  It composes complete-path encoding,
candidate-conditioned latent relations, sparse pair compatibility and
multiway permutation synchronization.  The only hard inference operation is
an integer gather; trajectory coordinates are never averaged or corrected.
"""

from __future__ import annotations

from typing import Dict, Optional
import time

import torch
from torch import nn

from src.models.hungarian_projection import (
    project_permutations,
    validate_bijection,
)
from src.models.permutation_synchronizer import MultiwayPermutationSynchronizer
from src.models.trajectory_pair_energy import TrajectoryPairEnergy
from src.models.trajectory_pair_relation import (
    CandidateConditionedRelation,
    TrajectoryEncoder,
    build_trajectory_pair_geometry,
)


def gather_trajectory_bank(
        trajectories: torch.Tensor,
        permutation: torch.Tensor) -> torch.Tensor:
    """Gather ``aligned[i,slot] = raw[i,permutation[i,slot]]`` only."""
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError('trajectories must have shape [N,K,T,2]')
    if permutation.shape != trajectories.shape[:2]:
        raise ValueError('permutation must have shape [N,K]')
    validate_bijection(permutation, trajectories.shape[1])
    return trajectories.gather(
        1, permutation[:, :, None, None].expand(
            -1, -1, trajectories.shape[2], trajectories.shape[3]))


def assert_same_trajectory_set(
        raw: torch.Tensor,
        aligned: torch.Tensor) -> None:
    """Assert exact per-agent sample-set preservation (order may differ)."""
    if raw.shape != aligned.shape:
        raise AssertionError('raw/aligned trajectory banks have different shapes')
    # Lexicographic sorting is unavailable for tensors.  A strict pairwise
    # equality matching matrix is sufficient because hard projection is a
    # bijection and K is small (20 in the formal experiment).
    equality = raw[:, :, None].eq(aligned[:, None, :]).flatten(3).all(dim=-1)
    if not bool(equality.any(dim=2).all() and equality.any(dim=1).all()):
        raise AssertionError(
            'Hard coupling changed, duplicated, or removed a trajectory')


class RelationAwareMultiwayCoupler(nn.Module):
    """Learn globally consistent candidate permutations over sparse graphs."""

    def __init__(
            self,
            trajectory_dim: int,
            edge_dim: int,
            num_relation_modes: int = 4,
            trajectory_pair_rank: int = 8,
            trajectory_conditioned_relation: bool = True,
            pair_specific_gate: bool = True,
            trajectory_energy_type: str = 'lowrank',
            sync_iterations: int = 4,
            sinkhorn_iterations: int = 8,
            sync_temperature: float = 0.2,
            hard_projection: str = 'hungarian',
            use_identity_bypass: bool = True,
            keep_threshold: float = 0.6,
            trajectory_dt: float = 0.4,
            inference_edge_topk: int = 4,
    ) -> None:
        super().__init__()
        if hard_projection not in {'hungarian', 'greedy'}:
            raise ValueError('hard_projection must be greedy or hungarian')
        if not 0.0 <= keep_threshold <= 1.0:
            raise ValueError('keep_threshold must lie in [0,1]')
        self.hard_projection = hard_projection
        self.use_identity_bypass = bool(use_identity_bypass)
        self.keep_threshold = float(keep_threshold)
        self.trajectory_encoder = TrajectoryEncoder(
            output_dim=trajectory_dim, dt=trajectory_dt)
        self.trajectory_relation = CandidateConditionedRelation(
            trajectory_dim=trajectory_dim,
            edge_dim=edge_dim,
            num_relation_modes=num_relation_modes,
            hidden_dim=trajectory_dim,
            trajectory_conditioned=trajectory_conditioned_relation,
        )
        energy_kwargs = dict(
            trajectory_dim=trajectory_dim,
            edge_dim=edge_dim,
            num_relation_modes=num_relation_modes,
            rank=trajectory_pair_rank,
            hidden_dim=trajectory_dim,
            pair_specific_gate=pair_specific_gate,
        )
        # Newer implementations expose the MLP/low-rank ablation through this
        # constructor argument.  Keeping the TypeError fallback makes a
        # partially upgraded checkpoint readable, while the parser tests
        # ensure the formal code path uses the complete implementation.
        try:
            self.trajectory_energy = TrajectoryPairEnergy(
                energy_type=trajectory_energy_type, **energy_kwargs)
        except TypeError:
            if trajectory_energy_type != 'lowrank':
                raise
            self.trajectory_energy = TrajectoryPairEnergy(**energy_kwargs)
        self.permutation_synchronizer = MultiwayPermutationSynchronizer(
            sync_iterations=sync_iterations,
            sinkhorn_iterations=sinkhorn_iterations,
            tau_sync=sync_temperature,
            inference_edge_topk=inference_edge_topk,
            use_inference_edge_topk=inference_edge_topk > 0,
            use_soft_confidence_mixing=use_identity_bypass,
            hard_projection=hard_projection,
            use_identity_bypass=use_identity_bypass,
            keep_threshold=keep_threshold,
        )

    def forward(
            self,
            trajectories: torch.Tensor,
            last_pos: torch.Tensor,
            edge_index: torch.Tensor,
            edge_feat: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
            relation_prior: Optional[torch.Tensor] = None,
            scene_index: Optional[torch.Tensor] = None,
            *,
            hard: bool = False,
            profile: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return soft synchronization and, when requested, hard reordering."""
        def stamp():
            if profile and trajectories.is_cuda:
                torch.cuda.synchronize(trajectories.device)
            return time.perf_counter()

        started = stamp()
        trajectory_feat = self.trajectory_encoder(trajectories, last_pos)
        after_encoder = stamp()
        pair_geometry = build_trajectory_pair_geometry(
            trajectories, last_pos, edge_index, scene_index=scene_index,
            dt=self.trajectory_encoder.dt)
        relation = self.trajectory_relation(
            trajectory_feat, edge_index, edge_feat, relation_prior,
            pair_geometry, scene_index=scene_index)
        energy = self.trajectory_energy(
            trajectory_feat, edge_index, edge_feat,
            relation['relation_posterior'], pair_geometry,
            edge_weight=edge_weight, scene_index=scene_index)
        after_pair = stamp()
        synchronization = self.permutation_synchronizer(
            energy['pair_score'], edge_index,
            edge_strength=energy['edge_gate'],
            scene_index=scene_index,
            num_agents=trajectories.shape[0],
            inference=hard)
        after_sync = stamp()

        soft_raw = synchronization['synchronized_permutation']
        soft = synchronization['soft_permutation']

        result: Dict[str, torch.Tensor] = {
            'raw_trajectories': trajectories,
            'trajectory_features': trajectory_feat,
            'pair_geometry': pair_geometry,
            'soft_permutation_raw': soft_raw,
            'soft_permutation': soft,
            'P': soft,
            **{f'relation_{key}': value for key, value in relation.items()},
            **{f'energy_{key}': value for key, value in energy.items()},
            **synchronization,
        }
        if profile:
            result['profile_ms'] = {
                'trajectory_encoder_ms': 1000.0 * (
                    after_encoder - started),
                'pair_module_ms': 1000.0 * (after_pair - after_encoder),
                'synchronizer_ms': 1000.0 * (after_sync - after_pair),
            }
        # Avoid prefixed-only access for the tensors consumed by the loss.
        result.update({
            'soft_permutation': soft,
            'P': soft,
            'pair_score': energy['pair_score'],
            'pair_gate': energy['pair_gate'],
            'edge_gate': energy['edge_gate'],
            'relation_posterior': relation['relation_posterior'],
            'relation_prior': relation['relation_prior'],
            'relation_prior_kl': relation['relation_prior_kl'],
        })
        if profile:
            result['profile_ms'] = {
                'trajectory_encoder_ms': 1000.0 * (
                    after_encoder - started),
                'pair_module_ms': 1000.0 * (
                    after_pair - after_encoder),
                'synchronizer_ms': 1000.0 * (
                    after_sync - after_pair),
            }

        if hard:
            permutation = synchronization['hard_permutation']
            identity_permutation = torch.arange(
                trajectories.shape[1], device=trajectories.device,
                dtype=torch.long).unsqueeze(0).expand_as(permutation)
            aligned = gather_trajectory_bank(trajectories, permutation)
            assert_same_trajectory_set(trajectories, aligned)

            greedy = project_permutations(soft_raw, method='greedy')
            greedy_permutation = greedy['permutation']
            if self.use_identity_bypass:
                keep = ~synchronization['identity_bypass']
                greedy_permutation = torch.where(
                    keep[:, None], greedy_permutation, identity_permutation)
            result.update({
                'permutation': permutation,
                'hard_permutation': permutation,
                'hard_assignment': synchronization['hard_assignment'],
                'aligned_trajectories': aligned,
                'changed_fraction': permutation.ne(
                    identity_permutation).float().mean(),
                'projection_metadata': synchronization[
                    'projection_metadata'],
                'greedy_permutation': greedy_permutation,
            })

        floating = [value for value in result.values()
                    if torch.is_tensor(value) and value.is_floating_point()]
        if any(not bool(value.isfinite().all()) for value in floating):
            raise FloatingPointError('V4 coupling produced NaN or Inf')
        return result


MultiwayTrajectoryCoupler = RelationAwareMultiwayCoupler


__all__ = [
    'MultiwayTrajectoryCoupler',
    'RelationAwareMultiwayCoupler',
    'assert_same_trajectory_set',
    'gather_trajectory_bank',
]
