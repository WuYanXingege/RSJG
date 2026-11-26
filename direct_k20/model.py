"""Direct K20 endpoint student with frozen, input-differentiable native decoding."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from bcep.motion_frame import frame, to_local
from bcep_v2.provider import GEOMETRY, spatial_encoding
from ebjd.metrics import hard_metric_statistics
from ebjd.sampling import differentiable_sample
from joint_flow_dependency.inference import decode, encode_observed, flow_world
from joint_flow_dependency.model import undirected_edges
from joint_flow_dependency.sampling import keyed_normal
from joint_flow_fourfold.inputs import state_hash


K = 20
FRAMES = 12
NATIVE_ATOL = 1e-6
NATIVE_RTOL = 1e-6


def _mlp(input_features, output_features):
    return nn.Sequential(nn.Linear(input_features, 64), nn.SiLU(),
                         nn.Linear(64, output_features))


class Student(nn.Module):
    """The same ordered query set defines the worlds for all valid agents."""

    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(K, 32))
        self.personal = _mlp(128 + 32, 64)
        self.edge = _mlp(64 + 64 + 4, 64)
        self.latent = _mlp(64 + 64, 2)

    def forward(self, observed, context, edges=None):
        n = len(observed)
        if n < 1 or observed.shape != (n, 8, 2) or context.shape != (n, 128):
            raise ValueError('expected one packed observation and its per-agent context')
        if edges is None:
            edges = undirected_edges(observed)
        first, second = edges.unbind(-1)
        receiver = torch.cat((first, second))
        sender = torch.cat((second, first))
        position = observed[:, -1]
        velocity = (position - observed[:, -4]) / 1.2
        relative = torch.cat((position[sender] - position[receiver],
                              velocity[sender] - velocity[receiver]), dim=-1)
        query_inputs = torch.cat((context.unsqueeze(0).expand(K, -1, -1),
                                  self.queries[:, None].expand(-1, n, -1)), dim=-1)
        personal = self.personal(query_inputs)
        if receiver.numel():
            messages = self.edge(torch.cat((personal[:, receiver], personal[:, sender],
                                            relative.unsqueeze(0).expand(K, -1, -1)), dim=-1))
            incidence = F.one_hot(receiver, num_classes=n).to(personal.dtype).T
            mean_incidence = incidence / incidence.sum(-1, keepdim=True).clamp_min(1.)
            # An explicit mean avoids floating-point atomic scatter reductions.
            interaction = torch.matmul(mean_incidence.unsqueeze(0), messages)
        else:
            interaction = torch.zeros_like(personal)
        return self.latent(torch.cat((personal, interaction), dim=-1))


class _FrozenBilinearMap(torch.autograd.Function):
    """Keep the native CUDA interpolation value with a deterministic grid VJP.

    The four-corner derivative only gathers from the frozen map. The map
    Jacobian is deliberately absent because the encoder is fixed.
    """

    @staticmethod
    def forward(ctx, features, grid, align_corners):
        if features.requires_grad:
            raise ValueError('input-gradient sampling requires frozen map features')
        ctx.save_for_backward(features, grid)
        ctx.align_corners = align_corners
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=align_corners)

    @staticmethod
    def backward(ctx, output_gradient):
        features, grid = ctx.saved_tensors
        batch, channels, height, width = features.shape
        grid_x, grid_y = grid.unbind(-1)
        if ctx.align_corners:
            x, y = (grid_x + 1) * ((width - 1) / 2), (grid_y + 1) * ((height - 1) / 2)
            scale_x, scale_y = (width - 1) / 2, (height - 1) / 2
        else:
            x, y = ((grid_x + 1) * width - 1) / 2, ((grid_y + 1) * height - 1) / 2
            scale_x, scale_y = width / 2, height / 2
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        flat = features.flatten(2)

        def corner(column, row):
            valid = (column >= 0) & (column < width) & (row >= 0) & (row < height)
            index = row.clamp(0, height - 1) * width + column.clamp(0, width - 1)
            values = flat.gather(2, index.reshape(batch, 1, -1).expand(-1, channels, -1))
            return values.reshape(batch, channels, *x.shape[1:]) * valid[:, None]

        top_left = corner(x0, y0)
        top_right = corner(x0 + 1, y0)
        bottom_left = corner(x0, y0 + 1)
        bottom_right = corner(x0 + 1, y0 + 1)
        slope_x = (top_right - top_left) * (1 - dy[:, None]) + (bottom_right - bottom_left) * dy[:, None]
        slope_y = (bottom_left - top_left) * (1 - dx[:, None]) + (bottom_right - top_right) * dx[:, None]
        grad_x = (output_gradient * slope_x).sum(1) * scale_x
        grad_y = (output_gradient * slope_y).sum(1) * scale_y
        return None, torch.stack((grad_x, grad_y), -1), None


def _bridge_grid_sample(features, grid, *, mode, padding_mode, align_corners):
    if (mode, padding_mode, align_corners) != ('bilinear', 'zeros', True):
        raise ValueError('unexpected native bridge map geometry')
    return _FrozenBilinearMap.apply(features, grid, True)


def _head_map(features, maps, goals, centers):
    if maps.shape[1:] != (6, 256, 256) or features.shape[1:] != (128, 32, 32):
        raise ValueError('candidate-head map geometry differs from the frozen provider')
    pixel = (goals.float() - centers[:, None].float() + 16.) * (255. / 32.)
    feature = pixel / 8.
    grid = (2 * (feature + 0.5) / 32.0 - 1).unsqueeze(2)
    sampled = _FrozenBilinearMap.apply(features.float(), grid, False).squeeze(-1).transpose(1, 2)
    with torch.no_grad():
        coverage = F.grid_sample(
            maps.float().sum(1, keepdim=True),
            (2 * pixel / 255.0 - 1).unsqueeze(2),
            mode='nearest',
            align_corners=True,
        )[:, 0, :, 0]
        inside = ((feature >= 0) & (feature <= 31)).all(-1) & (coverage > 0)
    return sampled, inside


def head_with_input_grad(provider, observed, maps, valid, goals, context, features):
    """Match CandidateHead's bounded correction without detaching its input."""
    if goals.shape != (K, len(observed), 2) or not valid.any():
        raise ValueError('expected [20,N,2] and a nonempty scene')
    base, head = provider.base, provider.head
    safe = torch.where(valid[:, None, None], goals.transpose(0, 1), 0.0)
    origin, rotation, scale = frame(torch.where(valid[:, None, None], observed, 0.0),
                                    FRAMES, base.dt, base.scales[-1])
    local = to_local(safe.float(), origin[:, None], rotation[:, None], scale)
    if maps is None:
        sampled = head.missing.expand(len(valid), K, -1)
        inside = torch.zeros_like(local[..., 0], dtype=torch.bool)
    else:
        sampled, inside = _head_map(features, maps, safe, observed[:, -1])
    embedding = head.embedding(torch.cat((
        context[:, None].expand(-1, K, -1),
        torch.where(inside[..., None], sampled, head.missing),
        local,
        inside[..., None].float(),
    ), dim=-1))
    raw = head.readout(torch.cat((embedding, embedding.mean(1, keepdim=True).expand_as(embedding)), dim=-1))
    shift = 0.5 * raw / (1 + raw.norm(dim=-1, keepdim=True))
    shift = (rotation[:, None] @ shift.unsqueeze(-1)).squeeze(-1)
    shift = torch.where(valid[:, None, None], shift, 0.0)
    return (safe + shift.to(safe)).transpose(0, 1)


def student_endpoints(student, provider, observed, maps, valid, context, features, edges=None,
                      *, input_grad=False):
    """Apply exactly one correction to the ordered scene query set."""
    if valid.shape != (len(observed),) or not valid.all():
        raise ValueError('use a packed window containing only valid agents')
    if input_grad:
        latent = student(observed, context, edges)
        raw = flow_world.__wrapped__(provider, observed, context, latent)
        return head_with_input_grad(provider, observed, maps, valid, raw, context, features)
    with torch.no_grad():
        latent = student(observed, context, edges)
        raw = flow_world(provider, observed, context, latent)
        goals, _ = provider.head(provider.base, observed, maps, valid, raw,
                                 context, features, dict(layout=GEOMETRY))
        return goals


def student_noise(window_key, agent_ids, seed, device=None):
    """Bind every bridge draw to the window, agent identity and student slot."""
    agents = tuple(str(agent) for agent in agent_ids)
    if not agents or len(set(agents)) != len(agents):
        raise ValueError('bridge noise requires unique valid agent identities')
    noise = torch.stack([
        torch.stack([
            keyed_normal('direct-k20-bridge-v1', str(window_key), agent, slot, int(seed),
                         shape=(FRAMES, 2))
            for agent in agents
        ])
        for slot in range(K)
    ]).unsqueeze(0)
    return noise.to(device=device)


def decode_student(model, bridge_context, goals, noise, *, input_grad=False, checkpoint_steps=True):
    """Reuse the native 20-step bridge and enable its endpoint gradient explicitly."""
    if goals.ndim != 3 or goals.shape[0] != K or goals.shape[-1] != 2:
        raise ValueError('expected K20 unbatched corrected goals')
    proposal = goals.unsqueeze(0)
    if noise.shape != (1, K, goals.shape[1], FRAMES, 2):
        raise ValueError('noise must match the complete K20 proposal')
    if not input_grad:
        return decode(model, bridge_context, proposal, noise)
    # Recompute each DDIM step during backward. Keep the map sampler installed
    # until recomputation finishes. With no gradient it returns literal native values.
    model.bridge.denoiser.map_sampler = _bridge_grid_sample
    path, _, _ = differentiable_sample(
        model.bridge, bridge_context, noise, steps=20, checkpoint_steps=checkpoint_steps,
        checkpoint_blocks=False, proposal_goals=proposal, proposal_eta=0.,
    )
    if not bool(torch.isfinite(path).all()):
        raise FloatingPointError('nonfinite bridge trajectory')
    torch.testing.assert_close(path[..., -1, :].detach(), proposal.detach(),
                               rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    return path


def teacher_terms(goals, teacher_a, teacher_b):
    """Return matched TA and TB sums, before full-logical-batch normalization."""
    if goals.shape != teacher_a.shape or teacher_b.shape != goals.shape or goals.shape[0] != K:
        raise ValueError('both teachers must match the ordered student agent population')
    a_cost = torch.linalg.vector_norm(goals[:, None] - teacher_a[None], dim=-1).permute(2, 0, 1)
    a_matching = [linear_sum_assignment(cost) for cost in a_cost.detach().cpu().numpy()]
    row = torch.as_tensor([match[0].tolist() for match in a_matching], device=goals.device)
    col = torch.as_tensor([match[1].tolist() for match in a_matching], device=goals.device)
    agents = torch.arange(goals.shape[1], device=goals.device)[:, None]
    ta = a_cost[agents, row, col].sum()
    if len(agents) == 1:
        return ta, None
    b_cost = torch.linalg.vector_norm(goals[:, None] - teacher_b[None], dim=-1).mean(-1)
    row, col = linear_sum_assignment(b_cost.detach().cpu().numpy())
    row = torch.as_tensor(row, device=goals.device)
    col = torch.as_tensor(col, device=goals.device)
    return ta, b_cost[row, col].sum()


def endpoint_terms(goals, truth_endpoint):
    """Unnormalized per-agent endpoint minimum sum and joint scene minimum."""
    if goals.shape[1:] != truth_endpoint.shape or truth_endpoint.ndim != 2:
        raise ValueError('endpoint labels must share the packed agent order')
    error = torch.linalg.vector_norm(goals.float() - truth_endpoint[None].float(), dim=-1)
    joint = error.sum(dim=1) / error.shape[1]
    return error.min(dim=0).values.sum(), joint.min()


def trajectory_terms(path, future, valid):
    """Use the native time-first reductions with gradients to the trajectory."""
    stats = hard_metric_statistics(path, future, valid)
    return stats.agent_minade[stats.agent_valid].sum(), stats.scene_jade.sum()


def preflight(student, model, provider, batch, teacher_a, teacher_b, window_key, seed=2035):
    """Exercise the original operators and the training gradient on one real window."""
    if len(batch.observed) != 1 or not bool(batch.valid.all()) or len(batch.agent_ids[0]) < 2:
        raise ValueError('the direct K20 check needs a packed multi-agent window')
    for frozen in (model, provider):
        if frozen.training or any(parameter.requires_grad for parameter in frozen.parameters()):
            raise ValueError('native provider and bridge must be frozen before the first check')
    frozen_before = (state_hash(model), state_hash(provider))
    student_before = state_hash(student)
    observed, maps, valid = batch.observed[0], batch.semantic_maps[0], batch.valid[0]
    agents = batch.agent_ids[0]
    n = len(agents)
    with torch.no_grad():
        context, features = encode_observed(provider, batch)
        bridge_context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    edges = undirected_edges(observed)
    noise = student_noise(window_key, agents, seed, observed.device)
    latent = student(observed, context, edges)
    raw = flow_world.__wrapped__(provider, observed, context, latent)
    goals = head_with_input_grad(provider, observed, maps, valid, raw, context, features)
    with torch.no_grad():
        native_raw = flow_world(provider, observed, context, latent)
        native_goals, _ = provider.head(provider.base, observed, maps, valid, native_raw,
                                        context, features, dict(layout=GEOMETRY))
        native_path = decode(model, bridge_context, native_goals[None], noise)
    path = decode_student(model, bridge_context, goals, noise, input_grad=True)
    for actual, expected in ((raw, native_raw), (goals, native_goals), (path, native_path)):
        torch.testing.assert_close(actual.detach(), expected, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    first_eleven = torch.linalg.vector_norm(
        path[..., :11, :] - batch.future[:, None, :, :11, :], dim=-1).mean()
    parameters = tuple(student.named_parameters())
    gradients = torch.autograd.grad(first_eleven, tuple(p for _, p in parameters))
    grad_l2 = {}
    for (name, _), gradient in zip(parameters, gradients):
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
            raise ValueError(f'first-eleven-frame gradient missing or nonfinite in {name}')
        grad_l2[name] = float(gradient.norm())
    with torch.no_grad():
        ta, tb = teacher_terms(goals, teacher_a, teacher_b)
        slots = torch.arange(K, device=observed.device)
        agent_order = torch.arange(n, device=observed.device)
        a_order = (slots[:, None] + agent_order[None, :] + 1) % K
        shuffled_a = teacher_a[a_order, agent_order[None, :]]
        shuffled_b = teacher_b[slots.flip(0)]
        relabel_ta, relabel_tb = teacher_terms(goals, shuffled_a, shuffled_b)
        torch.testing.assert_close(ta, relabel_ta, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
        torch.testing.assert_close(tb, relabel_tb, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
        order = agent_order.flip(0)
        reordered_context, reordered_features = spatial_encoding(
            provider.base, observed[order], maps[order], valid[order])
        reordered_goals = student_endpoints(student, provider, observed[order], maps[order], valid[order],
                                            reordered_context, reordered_features)
        torch.testing.assert_close(reordered_goals, goals[:, order], rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
        reordered_agents = [agents[i] for i in order.tolist()]
        reordered_noise = student_noise(window_key, reordered_agents, seed, observed.device)
        if not torch.equal(reordered_noise, noise[:, :, order]):
            raise ValueError('student slot noise did not follow the ordered agent identities')
        reordered_bridge_context = model.encode_context(
            batch.observed[:, order], batch.semantic_maps[:, order], batch.valid[:, order])
        reordered_path = decode_student(model, reordered_bridge_context, reordered_goals, reordered_noise)
        # Graph and attention reductions can change their FP32 summation order
        # under renumbering. The literal endpoint is separately checked at 1e-6.
        torch.testing.assert_close(reordered_path, native_path[:, :, order], rtol=1e-5, atol=1e-5)
        native = hard_metric_statistics(path, batch.future, batch.valid)
        mf, jf = endpoint_terms(goals, batch.future[0, :, -1])
        ma, ja = trajectory_terms(path, batch.future, batch.valid)
        metric_pairs = ((mf / n, native.agent_minfde.mean()), (jf, native.scene_jfde[0]),
                        (ma / n, native.agent_minade.mean()), (ja, native.scene_jade[0]))
        for actual, expected in metric_pairs:
            torch.testing.assert_close(actual, expected, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    if frozen_before != (state_hash(model), state_hash(provider)) or student_before != state_hash(student):
        raise ValueError('the preflight updated model parameters')
    return dict(
        passed=True,
        agents=n,
        edges=len(edges),
        seed=int(seed),
        native_atol=NATIVE_ATOL,
        native_rtol=NATIVE_RTOL,
        native_flow_max_abs_m=float((raw.detach() - native_raw).abs().max()),
        native_head_max_abs_m=float((goals.detach() - native_goals).abs().max()),
        native_bridge_max_abs_m=float((path.detach() - native_path).abs().max()),
        final_frame_max_abs_m=float((path.detach()[0, ..., -1, :] - goals.detach()).abs().max()),
        agent_reorder_max_abs_m=float((reordered_goals - goals.detach()[:, order]).abs().max()),
        agent_reorder_path_max_abs_m=float((reordered_path - native_path[:, :, order]).abs().max()),
        agent_reorder_path_atol_m=1e-5,
        agent_reorder_path_rtol=1e-5,
        teacher_slot_permutation_ta_abs=float((ta - relabel_ta).abs()),
        teacher_slot_permutation_tb_abs=float((tb - relabel_tb).abs()),
        native_metric_max_abs_m=max(float((actual - expected).abs()) for actual, expected in metric_pairs),
        first_eleven_loss=float(first_eleven.detach()),
        first_eleven_grad_l2=grad_l2,
        training_updates=0,
        ddim_calls=3,
        frozen_state_verified=True,
    )
