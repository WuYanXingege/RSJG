"""Lossless candidate occurrence transport, independent of learned scores.

Bank slots are immutable ONLY inside their verified bank-row/person namespace.
Without transported slots, distinct finite FP32 coordinates admit a deterministic
lexicographic order. Coincident coordinates do not identify noise occurrences.
"""
from dataclasses import dataclass
import torch
from ebjd.unified import gather_candidates


class CandidateIdentityAmbiguity(ValueError):
    """Identical endpoint values cannot recover distinct candidate/noise identities."""


@dataclass(frozen=True)
class CandidateOrder:
    canonical_to_input: torch.Tensor  # [B,N,K]
    input_to_canonical: torch.Tensor
    source: str

    def gather(self, values):
        return gather_candidates(values, self.canonical_to_input.transpose(1, 2))

    def to_input_indices(self, canonical_pi):
        return self.canonical_to_input.gather(2, canonical_pi.transpose(1, 2)).transpose(1, 2)

    def to_canonical_indices(self, input_pi):
        return self.input_to_canonical.gather(2, input_pi.transpose(1, 2)).transpose(1, 2)

    def columns_to_input(self, matrix):
        return matrix.gather(3, self.input_to_canonical[:, :, None].expand_as(matrix))


def canonical_order(goals, valid, candidate_ids=None, *, single_identity=True):
    if goals.dtype != torch.float32 or goals.shape != (len(valid), 20, valid.shape[1], 2):
        raise ValueError('finite FP32 [B,20,N,2] candidates required')
    points = goals.detach().transpose(1, 2)
    if not torch.isfinite(points[valid]).all():
        raise ValueError('nonfinite valid candidate')
    identity = torch.arange(20, device=goals.device).expand(*valid.shape, 20)
    if candidate_ids is not None:
        if (candidate_ids.dtype != torch.int64 or candidate_ids.device != goals.device
                or candidate_ids.shape != identity.shape
                or not torch.equal(candidate_ids.sort(-1).values, identity)):
            raise ValueError('transported occurrence IDs must be an int64 0..19 bijection per person')
        # Slot identity resolves coincident physical endpoints without inventing a
        # coordinate perturbation. Distinct coordinates have the same order with
        # or without IDs, so existing native callbacks keep the same mechanism.
        initial = candidate_ids.argsort(dim=-1, stable=True)
        source = 'exact_FP32_xy_then_transported_original_slot'
    else:
        initial = identity
        source = 'exact_FP32_xy_lexicographic'
    # Exact FP32 comparisons: no rounding, epsilon, tolerance or ID embedding.
    by_y = initial.gather(2, points[..., 1].gather(2, initial).argsort(dim=-1, stable=True))
    order = by_y.gather(2, points[..., 0].gather(2, by_y).argsort(dim=-1, stable=True))
    sorted_points = points.gather(2, order[..., None].expand_as(points))
    duplicates = (sorted_points[:, :, 1:] == sorted_points[:, :, :-1]).all(-1).any(-1)
    if candidate_ids is None and bool((duplicates & valid).any()):
        raise CandidateIdentityAmbiguity('coincident FP32 candidates require transported immutable occurrence IDs; array indices cannot recover identity')
    movable = valid & ((valid.sum(1, keepdim=True) > 1) if single_identity else True)
    # Preserve the native single-person/padding identity convention. A candidate
    # relabel in a single-person scene is a uniform world relabel, not a pairing.
    order = torch.where(movable[..., None], order, identity)
    return CandidateOrder(order, order.argsort(-1), source)


def bank_occurrence_ids(bank, indices, batch):
    """Read-only verified Bank boundary. Identity namespace lives in the receipt.

    Call before augmentation/relabel; carry the returned IDs through every gather.
    Rigid augmentation acts on coordinates only, and does not change these IDs.
    """
    native = getattr(bank, 'native', bank)
    goals = bank.batch(indices, batch)
    ids = torch.arange(20, dtype=torch.int64).expand(len(indices), batch.valid.shape[1], 20).clone()
    namespaces = []
    for index in indices:
        row = native.rows[index]
        namespaces.append(dict(bank_hash=native.hash, row_sha256=row['sha256'], member=row['member'], index=index))
    return goals, ids, namespaces
