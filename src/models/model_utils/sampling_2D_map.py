from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.models.model_utils.kmeans import kmeans
from sklearn.cluster import KMeans, kmeans_plusplus

def normalize_prob_map(x):
    """Normalize a probability map of shape (B, T, H, W) so
    that sum over H and W equal ones"""
    assert len(x.shape) == 4
    if not torch.is_floating_point(x) or not torch.isfinite(x).all() or \
            (x < 0).any():
        raise ValueError('probability map must be finite and non-negative')
    sums = x.sum(-1, keepdim=True).sum(-2, keepdim=True)
    normalized = torch.divide(x, sums.clamp_min(torch.finfo(x.dtype).tiny))
    uniform = torch.full_like(x, 1.0 / (x.shape[-2] * x.shape[-1]))
    return torch.where(sums > 0, normalized, uniform)


def un_normalize_prob_map(x):
    """Un-Normalize a probability map of shape (B, T, H, W) so
    that each pixel has value between 0 and 1"""
    assert len(x.shape) == 4
    (B, T, H, W) = x.shape
    maxs, _ = x.reshape(B, T, -1).max(-1)
    x = torch.divide(x, maxs.unsqueeze(-1).unsqueeze(-1))
    return x


def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x


def mean_over_map(x):
    """
    Mean over map: Input is a batched image of shape (B, T, H, W)
    where sigmoid/softmax is already performed (not logits).
    Output shape is (B, T, 2).
    """
    (B, T, H, W) = x.shape
    x = normalize_prob_map(x)
    pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
    x = x.view(B, T, -1)

    estimated_x = torch.sum(pos_x.reshape(-1) * x, dim=-1, keepdim=True)
    estimated_y = torch.sum(pos_y.reshape(-1) * x, dim=-1, keepdim=True)
    mean_coords = torch.cat([estimated_x, estimated_y], dim=-1)
    return mean_coords


def argmax_over_map(x):
    """
    From probability maps of shape (B, T, H, W), extract the
    coordinates of the maximum values (i.e. argmax).
    Hint: you need to use numpy.amax
    Output shape is (B, T, 2)
    """

    def indexFunc(array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx

    B, T, _, _ = x.shape
    device = x.device
    x = x.detach().cpu().numpy()
    maxVals = np.amax(x, axis=(2, 3))
    max_indices = np.zeros((B, T, 2), dtype=np.int64)
    for index in np.ndindex(x.shape[0], x.shape[1]):
        max_indices[index] = np.asarray(
            indexFunc(x[index], maxVals[index]), dtype=np.int64)[::-1]
    max_indices = torch.from_numpy(max_indices)
    return max_indices.to(device)

def sampling_best(probability_map,num_samples=10000):
    prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
    samples = prob_map.argmax(dim=1)
    samples = samples.repeat(1, num_samples) # (B*T, num_samples)
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))
    return preds

def sampling(probability_map,
             num_samples=10000,
             rel_threshold=0.05,
             replacement=True):
    """Given probability maps of shape (B, T, H, W) sample
    num_samples points for each B and T"""
    if probability_map.ndim != 4 or not probability_map.is_floating_point():
        raise ValueError('probability_map must be floating [B,T,H,W]')
    if not torch.isfinite(probability_map).all() or \
            (probability_map < 0).any():
        raise ValueError('probability_map must be finite and non-negative')
    # new view that has shape=[batch*timestep, H*W]
    prob_map = probability_map.reshape(
        probability_map.size(0) * probability_map.size(1), -1)
    original_prob_map = prob_map
    if rel_threshold is not None:
        # exclude points with very low probability
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).to(prob_map.dtype)

    # Multinomial normalizes rows internally, but explicit row normalization
    # lets us repair a zero-mass agent independently. A global normalization
    # would still leave a zero row invalid when another agent has mass.
    row_sum = prob_map.sum(dim=1, keepdim=True)
    normalized = prob_map / row_sum.clamp_min(torch.finfo(prob_map.dtype).tiny)
    fallback = torch.zeros_like(prob_map)
    fallback.scatter_(1, original_prob_map.argmax(dim=1, keepdim=True), 1.0)
    prob_map = torch.where(row_sum > 0, normalized, fallback)

    # samples.shape=[batch*timestep, num_samples]
    samples = torch.multinomial(prob_map,
                                num_samples=num_samples,
                                replacement=replacement)

    # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))
    return preds


def TTST_test_time_sampling_trick(x, num_goals, device):
    """
    From a probability map of shape (B, 1, H, W), sample num_goals
    goals so that they cover most of the space (thanks to k-means).
    Output shape is (num_goals, B, 1, 2).
    """
    assert x.shape[1] == 1
    # first sample is argmax sample
    num_clusters = num_goals
    goal_samples_argmax = argmax_over_map(x)

    # sample a large amount of goals to be clustered
    goal_samples = sampling(x[:, 0:1], num_samples=1000)
    # from (B, 1, num_samples, 2) to (num_samples, B, 1, 2)
    goal_samples = goal_samples.permute(2, 0, 1, 3)

    # Iterate through all person/batch_num, as this k-Means implementation
    # doesn't support batched clustering
    goal_samples_list = []
    for person in range(goal_samples.shape[1]):
        goal_sample = goal_samples[:, person, 0]

        # Actual k-means clustering, Outputs:
        # cluster_ids_x -  Information to which cluster_idx each point belongs
        # to cluster_centers - list of centroids, which are our new goal samples
        cluster_ids_x, cluster_centers = kmeans(X=goal_sample,
                                                num_clusters=num_clusters,
                                                distance='euclidean',
                                                device=device, tqdm_flag=False,
                                                tol=0.001, iter_limit=1000)
        goal_samples_list.append(cluster_centers)

    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
    goal_samples = torch.cat([goal_samples, goal_samples_argmax.unsqueeze(0)],
                             dim=0)
    return goal_samples


def _topk_goal_candidates(probability_map, num_candidates):
    """Extract deterministic heatmap candidates as ``[N, K, 2]`` (x, y)."""
    num_agents, _, height, width = probability_map.shape
    flat = probability_map[:, 0].reshape(num_agents, -1)
    take = min(num_candidates, flat.shape[-1])
    indices = flat.topk(take, dim=-1).indices
    if take < num_candidates:
        padding = indices[:, :1].expand(-1, num_candidates - take)
        indices = torch.cat((indices, padding), dim=-1)
    x_coord = indices.remainder(width)
    y_coord = torch.div(indices, width, rounding_mode='floor')
    return torch.stack((x_coord, y_coord), dim=-1).to(probability_map.dtype)


def candidate_probabilities(probability_map, goal_candidates, eps=1e-8):
    """Read bilinear heatmap mass at candidate points and normalize over K.

    Parameters
    ----------
    probability_map : torch.Tensor
        Endpoint heatmaps ``[N, 1, H, W]``.
    goal_candidates : torch.Tensor
        Candidate coordinates in heatmap pixels, ``[N, K, 2]`` (x, y).
    """
    if probability_map.ndim != 4 or probability_map.shape[1] != 1:
        raise ValueError('probability_map must have shape [N, 1, H, W]')
    if goal_candidates.ndim != 3 or goal_candidates.shape[-1] != 2:
        raise ValueError('goal_candidates must have shape [N, K, 2]')
    height, width = probability_map.shape[-2:]
    grid = goal_candidates.clone()
    if width > 1:
        grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    else:
        grid[..., 0] = 0.0
    if height > 1:
        grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    else:
        grid[..., 1] = 0.0
    sampled = F.grid_sample(
        probability_map, grid.unsqueeze(2), mode='bilinear',
        padding_mode='border', align_corners=True)[:, 0, :, 0]
    sampled = sampled.clamp_min(eps)
    return sampled / sampled.sum(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def generate_goal_candidates(probability_map, num_candidates, device,
                             use_ttst=True):
    """Separate GDTS candidate generation from structured joint selection.

    The legacy baseline still calls :func:`TTST_test_time_sampling_trick`
    unchanged. Social variants call this function and receive an explicit
    candidate set and categorical probability for each agent.

    Returns
    -------
    goal_candidates : torch.Tensor
        ``[N, K, 2]`` heatmap coordinates.
    candidate_prob : torch.Tensor
        ``[N, K]`` normalized heatmap masses.
    """
    if num_candidates < 1:
        raise ValueError('num_candidates must be positive')
    if probability_map.ndim != 4 or probability_map.shape[1] != 1 or \
            not probability_map.is_floating_point():
        raise ValueError('probability_map must have shape [N,1,H,W]')
    if not torch.isfinite(probability_map).all() or \
            (probability_map < 0).any():
        raise ValueError('probability_map must be finite and non-negative')

    if use_ttst and num_candidates > 1:
        # The original helper appends one argmax to ``num_goals`` clusters.
        raw = TTST_test_time_sampling_trick(
            probability_map, num_goals=num_candidates - 1, device=device)
        goal_candidates = raw.squeeze(2).permute(1, 0, 2)
    else:
        goal_candidates = _topk_goal_candidates(
            probability_map, num_candidates)

    candidate_prob = candidate_probabilities(
        probability_map, goal_candidates)
    return goal_candidates, candidate_prob
