"""Separate marginal K-candidate coverage from joint permutation learning.

Research objective, not an execution-only optimization. NLL remains active.
The shared encoder still receives both objectives; only the direct path from
joint pairing through candidate coordinates into the goal head is blocked.
"""
import math
import torch

LEGACY = 'legacy_joint_goal_v1'
SEPARATED = 'marginal_coverage_separated_pair_v1'


def marginal_goal_coverage(proposals, truth, valid, goal_scale, temperature=0.1):
    """Agent-weighted smooth min over K, before aggregation across agents."""
    if proposals.ndim != 4 or proposals.shape[-1] != 2:
        raise ValueError('proposals must be [B,K,N,2]')
    if truth.shape != proposals.shape[:1] + proposals.shape[2:] or valid.shape != truth.shape[:2]:
        raise ValueError('truth/mask dimensions disagree')
    if temperature <= 0 or proposals.shape[1] != 20:
        raise ValueError('this experiment fixes K=20 and positive temperature')
    with torch.autocast(device_type=proposals.device.type, enabled=False):
        distance = torch.linalg.vector_norm(proposals.float() - truth[:, None].float(), dim=-1)
        distance = distance / goal_scale.float().clamp_min(0.5)
        per_agent = -temperature * (torch.logsumexp(-distance / temperature, dim=1)
                                    - math.log(proposals.shape[1]))
        mask = valid.float()
        return (per_agent * mask).sum() / mask.sum().clamp_min(1)
