import torch
from ebjd.objectives import diffusion_loss, gated_geometry_mean, map_loss, relative_motion_loss
from ebjd.representation import cosine_vp
from .joint import clean_path_terms, pair_loss


def bridge_and_pair_loss(
    model,
    batch,
    proposals,
    pair_weight=1.0,
    *,
    time_draw=None,
    noise=None,
):
    """One shared context and one combined objective graph."""
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    coupled, _, _ = model.coupling.assign(
        context, proposals, model.bridge.representation.goal_scale, straight_through=True
    )
    options_pair = {}
    if getattr(model, "pair_backward_proxy", "coordinate") == "candidate_cost":
        options_pair = dict(goals=proposals, soft=model.coupling.final_soft)
    pair, metrics = pair_loss(coupled, batch.future[..., -1, :], batch.valid, **options_pair)
    model.last_pair_metrics = dict(metrics, effective_edges=model.coupling.last_edge_count)
    endpoint = batch.future[..., -1, :].detach()
    clean = model.bridge.representation.encode_target_conditioned(
        batch.future.detach(), context.baseline, endpoint, eta=0.0
    )[:, None]
    if time_draw is None:
        time_draw = torch.randint(1, 101, (batch.observed.shape[0],), device=batch.observed.device)
        time_draw = time_draw.float() / 100
    if noise is None:
        noise = torch.randn_like(clean, dtype=torch.float32)
    alpha, sigma = cosine_vp(time_draw)
    alpha = alpha.to(clean).view(-1, 1, 1, 1, 1)
    sigma = sigma.to(clean).view(-1, 1, 1, 1, 1)
    noisy = alpha * clean.float() + sigma * noise
    target = alpha * noise - sigma * clean.float()
    final, coarse, coarse_path, _ = model.bridge.denoiser(
        noisy, time_draw, context, proposal_goals=endpoint[:, None], proposal_eta=0.0
    )
    final_clean = alpha * noisy - sigma * final.float()
    final_path, _ = model.bridge.representation.decode_conditioned(
        final_clean, context.baseline, endpoint[:, None], eta=0.0
    )
    options = model.training_options
    loss_diff = diffusion_loss(final, coarse, target, batch.valid, options["coarse"])
    per_scene_geo = 0.5 * (
        relative_motion_loss(coarse_path, batch.future, context, 6.0, 2.0, 4.8)
        + relative_motion_loss(final_path, batch.future, context, 6.0, 2.0, 4.8)
    )
    loss_geo = gated_geometry_mean(per_scene_geo, alpha)
    loss_map = map_loss(batch.future, context, 0.5)
    agent_part = loss_diff + options["map"] * loss_map
    auxiliary, raw_sum, signal_sum = clean_path_terms(final_path, batch.future, batch.valid, alpha.square())
    model.last_loss_components = dict(
        clean_path_raw_sum=raw_sum,
        clean_path_signal_sum=signal_sum,
        clean_path_term=0.05 * auxiliary.detach(),
        diffusion_term=loss_diff.detach(),
        geometry_term=options["geometry"] * loss_geo.detach(),
        map_term=options["map"] * loss_map.detach(),
        pair_term=pair_weight * pair.detach(),
    )
    scene_part = 0.05 * auxiliary + options["geometry"] * loss_geo + pair_weight * pair
    bridge = 0.05 * auxiliary + agent_part + options["geometry"] * loss_geo
    return bridge + pair_weight * pair, bridge, pair, agent_part, scene_part
