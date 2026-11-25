"""K20 training forward with the native sampler's values and input gradients."""

import torch
from torch.nn import functional as F

from bcep.motion_frame import frame, to_local
from joint_flow_dependency.inference import flow_world, sample_endpoints
from joint_flow_dependency.model import sample_latents, undirected_edges
from joint_flow_dependency.sampling import base_noise, select_joint_samples


def flow_world_with_input_grad(provider, observed, context, z):
    # no_grad wraps the native function for inference. Call the same body here.
    return flow_world.__wrapped__(provider, observed, context, z)


class _FrozenBilinearMap(torch.autograd.Function):
    """Use native interpolation values and its analytical grid derivative.

    The maps are frozen, so no gradient needs to be scattered back to pixels.
    The dedicated grid derivative avoids CUDA grid_sample backward atomics.
    """

    @staticmethod
    def forward(ctx, features, grid):
        if features.requires_grad:
            raise ValueError("map feature cache must be frozen")
        ctx.save_for_backward(features, grid)
        return F.grid_sample(features, grid, align_corners=False)

    @staticmethod
    def backward(ctx, output_gradient):
        features, grid = ctx.saved_tensors
        batch, channels, height, width = features.shape
        grid_x, grid_y = grid.unbind(-1)
        x = ((grid_x + 1) * width - 1) / 2
        y = ((grid_y + 1) * height - 1) / 2
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        flat = features.flatten(2)

        def corner(column, row):
            present = (column >= 0) & (column < width) & (row >= 0) & (row < height)
            index = row.clamp(0, height - 1) * width + column.clamp(0, width - 1)
            values = flat.gather(2, index.reshape(batch, 1, -1).expand(-1, channels, -1))
            return values.reshape(batch, channels, *x.shape[1:]) * present[:, None]

        top_left = corner(x0, y0)
        top_right = corner(x0 + 1, y0)
        bottom_left = corner(x0, y0 + 1)
        bottom_right = corner(x0 + 1, y0 + 1)
        slope_x = (top_right - top_left) * (1 - dy[:, None]) + (bottom_right - bottom_left) * dy[:, None]
        slope_y = (bottom_left - top_left) * (1 - dx[:, None]) + (bottom_right - top_right) * dx[:, None]
        grad_x = (output_gradient * slope_x).sum(1) * (width / 2)
        grad_y = (output_gradient * slope_y).sum(1) * (height / 2)
        return None, torch.stack((grad_x, grad_y), -1)


def _query_map_with_input_grad(features, maps, goals, centers):
    if maps.shape[1:] not in ((6, 256, 256), (1, 256, 256)):
        raise ValueError("expected source map or cached coverage map")
    if features.shape[1:] != (128, 32, 32):
        raise ValueError("feature geometry mismatch")
    pixel = (goals.float() - centers[:, None].float() + 16.0) * (255.0 / 32.0)
    feature = pixel / 8.0
    grid = (2 * (feature + 0.5) / 32.0 - 1).unsqueeze(2)
    sampled = _FrozenBilinearMap.apply(features.float(), grid).squeeze(-1).transpose(1, 2)
    # The native coverage branch uses nearest-neighbor sampling and a hard mask.
    # A cached positive-coverage raster therefore gives the same mask as its sum.
    with torch.no_grad():
        coverage_map = maps.float().sum(1, keepdim=True) if maps.shape[1] == 6 else maps.float()
        coverage = F.grid_sample(
            coverage_map,
            (2 * pixel / 255.0 - 1).unsqueeze(2),
            mode="nearest",
            align_corners=True,
        )[:, 0, :, 0]
        inside = ((feature >= 0) & (feature <= 31)).all(-1) & (coverage > 0)
    return sampled, inside


def head_with_input_grad(provider, observed, maps, valid, goals, context, features):
    """Native CandidateHead arithmetic with a differentiable endpoint input."""
    if goals.shape != (20, len(observed), 2) or not valid.any():
        raise ValueError("expected [20,N,2] and a nonempty scene")
    base, head = provider.base, provider.head
    safe = torch.where(valid[:, None, None], goals.transpose(0, 1), 0.0)
    origin, rotation, scale = frame(
        torch.where(valid[:, None, None], observed, 0.0), 12, base.dt, base.scales[-1]
    )
    local = to_local(safe.float(), origin[:, None], rotation[:, None], scale)
    if maps is None:
        sampled = head.missing.expand(len(valid), 20, -1)
        inside = torch.zeros_like(local[..., 0], dtype=torch.bool)
    else:
        sampled, inside = _query_map_with_input_grad(features, maps, safe, observed[:, -1])
    embedding = head.embedding(
        torch.cat(
            (
                context[:, None].expand(-1, 20, -1),
                torch.where(inside[..., None], sampled, head.missing),
                local,
                inside[..., None].float(),
            ),
            -1,
        )
    )
    raw = head.readout(
        torch.cat((embedding, embedding.mean(1, keepdim=True).expand_as(embedding)), -1)
    )
    shift = 0.5 * raw / (1 + raw.norm(dim=-1, keepdim=True))
    world_shift = (rotation[:, None] @ shift.unsqueeze(-1)).squeeze(-1)
    world_shift = torch.where(valid[:, None, None], world_shift, 0.0)
    return (safe + world_shift.to(safe)).transpose(0, 1), world_shift


def endpoint_forward(provider, dependence, observed, maps, valid, context, features, edges, epsilon, xi):
    """Generate native K20 endpoints without storing any student candidates."""
    if not valid.all():
        raise ValueError("use a packed, single-window batch")
    factor = dependence(observed, context, edges)
    z = sample_latents(factor, epsilon, xi)
    raw = flow_world_with_input_grad(provider, observed, context, z)
    ids, _ = select_joint_samples(raw.detach(), k=20)
    corrected, _ = head_with_input_grad(provider, observed, maps, valid, raw[ids], context, features)
    return dict(raw=raw, raw_indices=ids, corrected=corrected, R=factor["R"])


def check_initial_replay(teacher_provider, teacher_dependence, student_provider,
                         student_dependence, batch, context, features, seed):
    """One real multi-agent training window, with isolated endpoint gradients."""
    observed, valid = batch.observed[0], batch.valid[0]
    if int(valid.sum()) <= 1:
        raise ValueError("replay check needs more than one valid agent")
    edges = undirected_edges(observed)
    epsilon, xi = base_noise(batch.scene_ids[0], batch.agent_ids[0], edges, seed=seed)
    epsilon, xi = epsilon.to(observed.device), xi.to(observed.device)
    teacher = sample_endpoints(
        teacher_provider, teacher_dependence, batch, context, features,
        shared_base=(epsilon, xi), seed=seed,
    )
    maps = batch.semantic_maps[0] if batch.semantic_maps is not None else None
    coverage = None if maps is None else (maps.float().sum(1, keepdim=True) > 0).to(torch.uint8)
    student = endpoint_forward(
        student_provider, student_dependence, observed, coverage, valid,
        context, features, edges, epsilon, xi,
    )
    assert torch.equal(teacher["raw_indices"], student["raw_indices"])
    torch.testing.assert_close(student["corrected"], teacher["corrected"][0], atol=1e-6, rtol=1e-6)
    endpoint_error = (student["corrected"][:, valid] - batch.future[0, valid, -1]).norm(dim=-1)
    endpoint_loss = endpoint_error.amin(0).mean() + endpoint_error.mean(1).amin()
    dependence_parameters = tuple(student_dependence.parameters())
    head_parameters = tuple(student_provider.head.readout[-1].parameters())
    gradients = torch.autograd.grad(endpoint_loss, dependence_parameters + head_parameters)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    count = len(dependence_parameters)
    dependence_l2 = float(torch.stack([x.square().sum() for x in gradients[:count]]).sum().sqrt())
    head_l2 = float(torch.stack([x.square().sum() for x in gradients[count:]]).sum().sqrt())
    assert dependence_l2 > 0 and head_l2 > 0
    return dict(
        passed=True,
        agents=int(valid.sum()),
        edges=len(edges),
        noise_seed=int(seed),
        selected_ids_exact=True,
        endpoint_max_abs_m=float((student["corrected"].detach() - teacher["corrected"][0]).abs().max()),
        endpoint_only_loss=float(endpoint_loss.detach()),
        dependence_gradient_l2=dependence_l2,
        readout_last_gradient_l2=head_l2,
        NLL_used=False,
        optimizer_updates=0,
        DDIM_calls=0,
    )
