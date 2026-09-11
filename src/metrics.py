import torch
import numpy as np
from scipy.stats import gaussian_kde


def compute_metric_mask(seq_list):
    """
    Get a mask to denote whether to account predictions during metrics
    computation. It is supposed to calculate metrics only for pedestrians
    fully present during observation and prediction time-steps.

    Parameters
    ----------
    seq_list : PyTorch tensor
        Size = (seq_len,N_pedestrians).
        Boolean mask that is =1 if pedestrian i is present at time-step t.

    Returns
    -------
    metric_mask : PyTorch tensor
        Shape: (N_pedestrians,)
        metric_mask[i] = 1 if pedestrian i if fully present during
        observation and prediction time-steps.
    """
    metric_mask = seq_list.cumprod(dim=0)
    # fully present on the whole seq_length interval
    metric_mask = metric_mask[-1] > 0
    return metric_mask


def check_metrics_inputs(predictions,
                         ground_truth,
                         metric_mask):
    num_sample, seq_length, N_agents, num_coords = predictions.shape
    # assert data shape
    assert len(predictions.shape) == 4, \
        f"Expected 4D (MxTxNxC) array for predictions, got {predictions.shape}"
    assert ground_truth.shape == (seq_length, N_agents, num_coords), \
        f"Expected 3D (TxNxC) array for ground_truth, got {ground_truth.shape}"
    assert metric_mask.shape == (N_agents,), \
        f"Expected 1D (N) array for metric_mask, got {metric_mask.shape}"

    # assert all data is valid
    assert torch.isfinite(predictions).all(), \
        "Invalid value found in predictions"
    assert torch.isfinite(ground_truth).all(), \
        "Invalid value found in ground_truth"
    assert torch.isfinite(metric_mask).all(), \
        "Invalid value found in metric_mask"


def ADE_best_of(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask: torch.Tensor,
                obs_length: int = 8) -> list:
    """
    Compute ADE metric - Best-of-K selected.
    The best ADE from the samples is selected, based on the best ADE.
    Torch implementation. Returns a list of floats, one for each full
    example in the batch.

    Parameters
    ----------
    predictions : torch.Tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : torch.Tensor
        The target trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps

    Returns
    ----------
    ADE_error : list of float
        Average displacement error
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # l2-norm for each time-step
    error = torch.norm(predictions - ground_truth, p=2, dim=3)
    # only calculate for fully present pedestrians
    error_full = error[:, obs_length:, metric_mask]

    # mean over time-steps to find best ADE
    error_full_mean = torch.mean(error_full, dim=1)
    # print(torch.std(error_full_mean[:,0],unbiased=False))
    # min ADE over samples
    error_full_mean_min, _ = torch.min(error_full_mean, dim=0)  # ADE

    return error_full_mean_min.tolist()


def FDE_best_of(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask: torch.Tensor,
                obs_length: int = 8) -> tuple:
    """
    Compute FDE metric - Best-of-K selected.
    The best FDE from the samples is selected, based on the best ADE.
    Torch implementation. Returns a list of floats, one for each full
    example in the batch.

    Parameters
    ----------
    predictions : torch.Tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : torch.Tensor
        The target trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps

    Returns
    ----------
    FDE : list of float
        Final displacement error
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # l2-norm for each time-step
    error = torch.norm(predictions - ground_truth, p=2, dim=3)
    # only calculate for fully present pedestrians
    error_full = error[:, -1, metric_mask]

    # best error over samples
    final_error, _ = error_full.min(dim=0)
    return final_error.tolist()


def _check_trajectory_inputs(predictions: torch.Tensor,
                             ground_truth: torch.Tensor):
    """Validate trajectory tensors used by the joint metrics."""
    if not isinstance(predictions, torch.Tensor):
        raise TypeError("predictions must be a torch.Tensor")
    if not isinstance(ground_truth, torch.Tensor):
        raise TypeError("ground_truth must be a torch.Tensor")
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError(
            "predictions must have shape [num_samples, time, agents, 2], "
            f"got {tuple(predictions.shape)}")
    expected_gt_shape = predictions.shape[1:]
    if ground_truth.shape != expected_gt_shape:
        raise ValueError(
            f"ground_truth must have shape {tuple(expected_gt_shape)}, "
            f"got {tuple(ground_truth.shape)}")
    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("predictions must contain a sample and a time step")
    if predictions.device != ground_truth.device:
        raise ValueError("predictions and ground_truth must use the same device")
    if not torch.isfinite(predictions).all():
        raise ValueError("predictions contain NaN or Inf")
    if not torch.isfinite(ground_truth).all():
        raise ValueError("ground_truth contains NaN or Inf")


def _check_goal_inputs(goal_predictions: torch.Tensor,
                       goal_ground_truth: torch.Tensor):
    """Validate endpoint tensors used by the goal metrics."""
    if not isinstance(goal_predictions, torch.Tensor):
        raise TypeError("goal_predictions must be a torch.Tensor")
    if not isinstance(goal_ground_truth, torch.Tensor):
        raise TypeError("goal_ground_truth must be a torch.Tensor")
    if goal_predictions.ndim != 3 or goal_predictions.shape[-1] != 2:
        raise ValueError(
            "goal_predictions must have shape [num_samples, agents, 2], "
            f"got {tuple(goal_predictions.shape)}")
    expected_gt_shape = goal_predictions.shape[1:]
    if goal_ground_truth.shape != expected_gt_shape:
        raise ValueError(
            f"goal_ground_truth must have shape {tuple(expected_gt_shape)}, "
            f"got {tuple(goal_ground_truth.shape)}")
    if goal_predictions.shape[0] == 0:
        raise ValueError("goal_predictions must contain at least one sample")
    if goal_predictions.device != goal_ground_truth.device:
        raise ValueError(
            "goal_predictions and goal_ground_truth must use the same device")
    if not torch.isfinite(goal_predictions).all():
        raise ValueError("goal_predictions contain NaN or Inf")
    if not torch.isfinite(goal_ground_truth).all():
        raise ValueError("goal_ground_truth contains NaN or Inf")


def _agent_mask(metric_mask, num_agents, device):
    """Return a boolean mask on ``device`` without changing caller data."""
    if metric_mask is None:
        return torch.ones(num_agents, dtype=torch.bool, device=device)
    metric_mask = torch.as_tensor(metric_mask, device=device)
    if metric_mask.ndim != 1 or metric_mask.shape[0] != num_agents:
        raise ValueError(
            f"metric_mask must have shape ({num_agents},), "
            f"got {tuple(metric_mask.shape)}")
    if (metric_mask.is_floating_point() or metric_mask.is_complex()) and \
            not torch.isfinite(metric_mask).all():
        raise ValueError("metric_mask contains NaN or Inf")
    return metric_mask.bool()


def _scene_masks(scene_index, metric_mask, num_agents, device):
    """Build per-scene agent masks; agents from different scenes never mix."""
    valid_agents = _agent_mask(metric_mask, num_agents, device)
    if scene_index is None:
        return [valid_agents]

    scene_index = torch.as_tensor(scene_index, device=device)
    if scene_index.ndim != 1 or scene_index.shape[0] != num_agents:
        raise ValueError(
            f"scene_index must have shape ({num_agents},), "
            f"got {tuple(scene_index.shape)}")
    if (scene_index.is_floating_point() or scene_index.is_complex()) and \
            not torch.isfinite(scene_index).all():
        raise ValueError("scene_index contains NaN or Inf")
    return [valid_agents & scene_index.eq(scene_id)
            for scene_id in torch.unique(scene_index, sorted=True)]


def _check_obs_length(obs_length, seq_length):
    if not isinstance(obs_length, int):
        raise TypeError("obs_length must be an integer")
    if obs_length < 0 or obs_length >= seq_length:
        raise ValueError(
            f"obs_length must be in [0, {seq_length - 1}], got {obs_length}")


def _check_nonnegative_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def minADE_at_K(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask=None,
                obs_length: int = 8) -> list:
    """Per-agent marginal minADE@K.

    This is a named alias for the historical :func:`ADE_best_of` behavior:
    every valid agent may select a different best sample. Inputs use the GDTS
    layout ``[samples, time, agents, 2]`` and ``[time, agents, 2]``.
    """
    _check_trajectory_inputs(predictions, ground_truth)
    _check_obs_length(obs_length, predictions.shape[1])
    metric_mask = _agent_mask(
        metric_mask, predictions.shape[2], predictions.device)
    return ADE_best_of(predictions, ground_truth, metric_mask, obs_length)


def minFDE_at_K(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask=None,
                obs_length: int = 8) -> list:
    """Per-agent marginal minFDE@K.

    Every valid agent may select a different best sample. ``obs_length`` is
    accepted for API symmetry and compatibility with :func:`FDE_best_of`.
    """
    _check_trajectory_inputs(predictions, ground_truth)
    metric_mask = _agent_mask(
        metric_mask, predictions.shape[2], predictions.device)
    return FDE_best_of(predictions, ground_truth, metric_mask, obs_length)


def JADE(predictions: torch.Tensor,
         ground_truth: torch.Tensor,
         metric_mask=None,
         scene_index=None,
         obs_length: int = 8) -> list:
    """Joint average displacement error for each scene.

    A single sample index is selected after averaging displacement error over
    all valid agents and future time steps in a scene. If packed tensors contain
    multiple scenes, ``scene_index[n]`` identifies the scene of agent ``n``;
    each scene is evaluated independently. The result is one float per scene.
    """
    _check_trajectory_inputs(predictions, ground_truth)
    _check_obs_length(obs_length, predictions.shape[1])
    scene_masks = _scene_masks(
        scene_index, metric_mask, predictions.shape[2], predictions.device)
    errors = torch.norm(predictions - ground_truth, p=2, dim=-1)
    errors = errors[:, obs_length:]

    result = []
    for scene_mask in scene_masks:
        if not scene_mask.any():
            result.append(0.0)
            continue
        sample_error = errors[:, :, scene_mask].mean(dim=(1, 2))
        result.append(sample_error.min().detach().cpu().item())
    return result


def JFDE(predictions: torch.Tensor,
         ground_truth: torch.Tensor,
         metric_mask=None,
         scene_index=None,
         obs_length: int = 8) -> list:
    """Joint final displacement error for each scene.

    A single sample index is selected after averaging final displacement error
    over every valid agent in a scene. ``obs_length`` is accepted for symmetry
    with :func:`JADE`; final displacement always uses the last time step.
    """
    _check_trajectory_inputs(predictions, ground_truth)
    scene_masks = _scene_masks(
        scene_index, metric_mask, predictions.shape[2], predictions.device)
    final_errors = torch.norm(
        predictions[:, -1] - ground_truth[-1], p=2, dim=-1)

    result = []
    for scene_mask in scene_masks:
        if not scene_mask.any():
            result.append(0.0)
            continue
        sample_error = final_errors[:, scene_mask].mean(dim=1)
        result.append(sample_error.min().detach().cpu().item())
    return result


def _pair_collision_flags(relative_positions,
                          collision_threshold_meter,
                          method,
                          interpolation_steps):
    """Return ``[samples, pairs]`` path-collision flags."""
    point_distances = torch.norm(relative_positions, p=2, dim=-1)
    if method == 'discrete' or relative_positions.shape[1] == 1:
        return point_distances.le(collision_threshold_meter).any(dim=1)

    relative_start = relative_positions[:, :-1]
    relative_delta = relative_positions[:, 1:] - relative_start
    if method == 'segment':
        denominator = relative_delta.square().sum(dim=-1)
        eps = torch.finfo(relative_positions.dtype).eps
        closest_fraction = -(
            relative_start * relative_delta).sum(dim=-1)
        closest_fraction = (closest_fraction /
                            denominator.clamp_min(eps)).clamp(0.0, 1.0)
        closest_relative = (relative_start +
                            closest_fraction.unsqueeze(-1) * relative_delta)
        return torch.norm(closest_relative, p=2, dim=-1).le(
            collision_threshold_meter).any(dim=1)

    interpolation_fractions = torch.linspace(
        0.0, 1.0, interpolation_steps + 2,
        dtype=relative_positions.dtype,
        device=relative_positions.device)
    interpolated_relative = (
        relative_start.unsqueeze(-2) +
        relative_delta.unsqueeze(-2) *
        interpolation_fractions.view(1, 1, 1, -1, 1))
    interpolated_collision = torch.norm(
        interpolated_relative, p=2, dim=-1).le(
            collision_threshold_meter)
    return interpolated_collision.any(dim=1).any(dim=-1)


def collision_rate(predictions_world: torch.Tensor,
                   collision_threshold_meter,
                   metric_mask=None,
                   scene_index=None,
                   obs_length: int = 8,
                   method: str = 'segment',
                   interpolation_steps: int = 2) -> list:
    """Compute per-scene predicted pedestrian collision rate in meters.

    The rate is the fraction of ``(sample, unordered-agent-pair)`` paths that
    come within ``collision_threshold_meter`` at least once in the prediction
    horizon. Only pairs within the same ``scene_index`` are constructed.

    ``method='segment'`` (default) computes the exact closest distance under
    linear motion between adjacent predicted positions. ``'interpolate'`` uses
    ``interpolation_steps`` interior points, while ``'discrete'`` checks only
    provided time steps. Scenes with fewer than two valid agents return 0.0.
    Coordinates must already be converted to world meters.
    """
    if not isinstance(predictions_world, torch.Tensor):
        raise TypeError("predictions_world must be a torch.Tensor")
    if predictions_world.ndim != 4 or predictions_world.shape[-1] != 2:
        raise ValueError(
            "predictions_world must have shape "
            "[num_samples, time, agents, 2], "
            f"got {tuple(predictions_world.shape)}")
    if predictions_world.shape[0] == 0 or predictions_world.shape[1] == 0:
        raise ValueError(
            "predictions_world must contain a sample and a time step")
    if not predictions_world.is_floating_point():
        raise TypeError("predictions_world must be floating point")
    if not torch.isfinite(predictions_world).all():
        raise ValueError("predictions_world contains NaN or Inf")
    _check_obs_length(obs_length, predictions_world.shape[1])
    collision_threshold_meter = _check_nonnegative_finite(
        collision_threshold_meter, "collision_threshold_meter")

    method = method.lower()
    if method == 'interpolation':
        method = 'interpolate'
    if method not in {'discrete', 'segment', 'interpolate'}:
        raise ValueError(
            "method must be 'discrete', 'segment', or 'interpolate'")
    if not isinstance(interpolation_steps, int) or interpolation_steps < 1:
        raise ValueError("interpolation_steps must be an integer >= 1")

    scene_masks = _scene_masks(
        scene_index, metric_mask, predictions_world.shape[2],
        predictions_world.device)
    # Continuous checks include the boundary segment from the final observed
    # point to the first predicted point. Discrete evaluation continues to
    # inspect future frames only.
    future_start = (obs_length if method == 'discrete'
                    else max(obs_length - 1, 0))
    future_positions = predictions_world[:, future_start:]
    result = []
    for scene_mask in scene_masks:
        scene_positions = future_positions[:, :, scene_mask]
        num_agents = scene_positions.shape[2]
        if num_agents < 2:
            result.append(0.0)
            continue

        pair_index = torch.triu_indices(
            num_agents, num_agents, offset=1,
            device=predictions_world.device)
        relative_positions = (
            scene_positions[:, :, pair_index[0]] -
            scene_positions[:, :, pair_index[1]])
        collision_flags = _pair_collision_flags(
            relative_positions, collision_threshold_meter, method,
            interpolation_steps)
        result.append(collision_flags.float().mean().detach().cpu().item())
    return result


def goal_minFDE(goal_predictions_world: torch.Tensor,
                goal_ground_truth_world: torch.Tensor,
                metric_mask=None) -> list:
    """Per-agent minimum endpoint error over K goal samples, in meters."""
    _check_goal_inputs(goal_predictions_world, goal_ground_truth_world)
    metric_mask = _agent_mask(
        metric_mask, goal_predictions_world.shape[1],
        goal_predictions_world.device)
    errors = torch.norm(
        goal_predictions_world - goal_ground_truth_world, p=2, dim=-1)
    return errors[:, metric_mask].min(dim=0).values.detach().cpu().tolist()


def goal_recall_at_K(goal_predictions_world: torch.Tensor,
                     goal_ground_truth_world: torch.Tensor,
                     threshold_meter,
                     metric_mask=None) -> list:
    """Per-agent Goal Recall@K indicators at a metric distance threshold."""
    _check_goal_inputs(goal_predictions_world, goal_ground_truth_world)
    threshold_meter = _check_nonnegative_finite(
        threshold_meter, "threshold_meter")
    metric_mask = _agent_mask(
        metric_mask, goal_predictions_world.shape[1],
        goal_predictions_world.device)
    errors = torch.norm(
        goal_predictions_world - goal_ground_truth_world, p=2, dim=-1)
    recalled = errors[:, metric_mask].min(dim=0).values.le(threshold_meter)
    return recalled.float().detach().cpu().tolist()


def joint_goal_endpoint_error(goal_predictions_world: torch.Tensor,
                              goal_ground_truth_world: torch.Tensor,
                              metric_mask=None,
                              scene_index=None) -> list:
    """Best shared-sample mean endpoint error for every scene, in meters."""
    _check_goal_inputs(goal_predictions_world, goal_ground_truth_world)
    scene_masks = _scene_masks(
        scene_index, metric_mask, goal_predictions_world.shape[1],
        goal_predictions_world.device)
    errors = torch.norm(
        goal_predictions_world - goal_ground_truth_world, p=2, dim=-1)

    result = []
    for scene_mask in scene_masks:
        if not scene_mask.any():
            result.append(0.0)
            continue
        sample_error = errors[:, scene_mask].mean(dim=1)
        result.append(sample_error.min().detach().cpu().item())
    return result


def joint_goal_compatibility(goal_predictions_world: torch.Tensor,
                             goal_ground_truth_world: torch.Tensor,
                             metric_mask=None,
                             scene_index=None) -> list:
    """Best pairwise relative-goal compatibility error for every scene.

    For each shared sample, this compares every within-scene predicted relative
    endpoint displacement ``g_i - g_j`` with the ground-truth displacement.
    The mean pair error is minimized over samples. Lower is better; a translated
    but formation-preserving joint goal therefore has zero compatibility error.
    Scenes with fewer than two valid agents return 0.0.
    """
    _check_goal_inputs(goal_predictions_world, goal_ground_truth_world)
    scene_masks = _scene_masks(
        scene_index, metric_mask, goal_predictions_world.shape[1],
        goal_predictions_world.device)

    result = []
    for scene_mask in scene_masks:
        scene_predictions = goal_predictions_world[:, scene_mask]
        scene_ground_truth = goal_ground_truth_world[scene_mask]
        num_agents = scene_predictions.shape[1]
        if num_agents < 2:
            result.append(0.0)
            continue

        pair_index = torch.triu_indices(
            num_agents, num_agents, offset=1,
            device=goal_predictions_world.device)
        predicted_relative = (
            scene_predictions[:, pair_index[0]] -
            scene_predictions[:, pair_index[1]])
        ground_truth_relative = (
            scene_ground_truth[pair_index[0]] -
            scene_ground_truth[pair_index[1]])
        pair_error = torch.norm(
            predicted_relative - ground_truth_relative, p=2, dim=-1)
        sample_error = pair_error.mean(dim=1)
        result.append(sample_error.min().detach().cpu().item())
    return result


# Readable aliases used by evaluation code and experiment tables.
minADE = minADE_at_K
minFDE = minFDE_at_K
joint_ADE = JADE
joint_FDE = JFDE
collision_rate_world = collision_rate
goal_min_fde = goal_minFDE
goal_recall_at_k = goal_recall_at_K
joint_goal_compatibility_error = joint_goal_compatibility


def KDE_negative_log_likelihood(predictions: torch.Tensor,
                                ground_truth: torch.Tensor,
                                metric_mask: torch.Tensor,
                                obs_length: int = 8,
                                log_pdf_lower_bound: int = -20) -> list:
    """
    Kernel Density Estimation-based Negative Log-Likelihood metric (KDE-NLL).
    Computes KDE-NLL for a batch of agents. Returns a list of floats, one for
    each full agents in the batch.
    Only works in a stochastic settings (i.e. multi-future predictions).
    Numpy-Scipy implementation.

    Parameters
    ----------
    predictions : PyTorch tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : PyTorch tensor
        The ground_truth trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps
    log_pdf_lower_bound : int
        Lower bound for log pdf

    Returns
    ----------
    total_nll : list of float
        List negative log likelihoods (floats).
        len(total_nll): number of pedestrians in the batch with a
        complete ground truth trajectory.
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # only calculate for fully present pedestrians
    predictions_full = predictions[:, obs_length:, metric_mask].\
        permute(2, 0, 1, 3).cpu().numpy()
    ground_truth_full = ground_truth[obs_length:, metric_mask].\
        permute(1, 0, 2).cpu().numpy()

    pred_length = predictions_full.shape[-2]
    total_nll = []

    # for each full agent
    for agent_i in range(predictions_full.shape[0]):
        ground_truth = ground_truth_full[agent_i]  # [pred_length, 2]
        predictions = np.swapaxes(predictions_full[agent_i], 0, 1)
        # [pred_length, num_samples, 2]

        ll = 0.0  # log-likelihood initialization
        same_pred = 0  # number of equal predictions
        # for each timestep
        for timestep in range(pred_length):
            # current ground-truth (x,y)
            curr_gt = ground_truth[timestep]
            # if all identical prediction at particular time-step, skip
            if np.all(predictions[timestep, 1:] == predictions[timestep, :-1]):
                same_pred += 1
                continue
            try:
                # Gaussian KDE
                scipy_kde = gaussian_kde(predictions[timestep].T)
                # We need [0] because it's a (1,)-shaped numpy array.
                log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T),
                                  a_min=log_pdf_lower_bound, a_max=None)[0]
                if np.isnan(log_pdf) or np.isinf(log_pdf) or log_pdf > 100:
                    # difficulties in computing Gaussian_KDE
                    same_pred += 1
                    continue
                ll += log_pdf
            except:  # difficulties in computing Gaussian_KDE
                same_pred += 1

        if same_pred == pred_length:
            raise Exception('LL error: all predictions are identical.')

        nll = - ll / (pred_length - same_pred)
        total_nll.append(nll)
    return total_nll


def FDE_best_of_goal(all_aux_outputs: list,
                     ground_truth: torch.Tensor,
                     metric_mask: torch.Tensor,
                     args,
                     ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Works with a goal architecture model only.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()


def FDE_best_of_goal_world(all_aux_outputs: list,
                           scene,
                           ground_truth: torch.Tensor,
                           metric_mask: torch.Tensor,
                           args,
                           ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor
    # from pixel to world coordinates
    end_point_pred = scene.make_world_coord_torch(end_point_pred)

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()


def FDE_best_of_traj(all_outputs: torch.Tensor,
                     ground_truth: torch.Tensor,
                     metric_mask: torch.Tensor,
                     args,
                     ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Works with a goal architecture model only.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_outputs[:,-1]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()


def FDE_best_of_traj_world(all_outputs: torch.Tensor,
                           scene,
                           ground_truth: torch.Tensor,
                           metric_mask: torch.Tensor,
                           args,
                           ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_outputs[:,-1]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor
    # from pixel to world coordinates
    end_point_pred = scene.make_world_coord_torch(end_point_pred)

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()
