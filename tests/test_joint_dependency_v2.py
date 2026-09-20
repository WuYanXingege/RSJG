import math
import time

import pytest
import torch

from src.joint_goal_loss import (
    build_soft_goal_target,
    jdv2_mixture_composite_loss,
    jdv2_posterior_distillation,
    jdv2_pseudo_likelihood,
    jdv2_relation_kl,
    jdv2_scene_mode_log_score,
    jdv2_scene_kl,
    jdv2_teacher_probability,
    jdv2_warmup_beta,
)
from src.models.interaction_graph import EDGE_FEATURE_DIM
from src.models.interaction_graph import reverse_edge_features
from src.models.joint_dependency_v2 import (
    DependencyCorrector,
    DynamicHypothesisRelation,
    ParallelConditionalSampler,
    RelationSpecificJointEnergy,
    SceneFutureTeacher,
    SceneLatentPrior,
    UnaryGoalResidual,
)
from src.models.joint_dependency_v2.future_teacher import future_pair_descriptor
from src.models.joint_dependency_v2.joint_sampler import (
    mode_stratified_allocation,
)


def _tiny(seed=3):
    generator = torch.Generator().manual_seed(seed)
    n, k = 4, 5
    scene = torch.tensor([0, 0, 1, 1])
    edge = torch.tensor([[0, 2], [1, 3]])
    agent = torch.randn(n, 128, generator=generator)
    last = torch.randn(n, 2, generator=generator)
    goals = last[:, None] + torch.randn(n, k, 2, generator=generator)
    edge_feat = torch.randn(edge.shape[1], EDGE_FEATURE_DIM,
                            generator=generator)
    base = torch.randn(edge.shape[1], 4, generator=generator)
    return scene, edge, agent, last, goals, edge_feat, base


def test_scene_prior_permutation_invariance_and_packed_isolation():
    scene, _, agent, _, _, _, _ = _tiny()
    model = SceneLatentPrior().eval()
    original = model(agent, scene)
    permutation = torch.tensor([3, 1, 2, 0])
    permuted = model(agent[permutation], scene[permutation])
    assert torch.allclose(original["prob"], permuted["prob"], atol=1e-6)
    changed = agent.clone()
    changed[:2] += 10
    changed_output = model(changed, scene)
    assert torch.equal(original["prob"][1], changed_output["prob"][1])


def test_future_teacher_shapes_finite_and_reversal_consistency():
    scene, edge, _, last, _, _, _ = _tiny()
    future = last[:, None] + torch.randn(4, 12, 2)
    velocity = torch.randn_like(future)
    prior_logits = torch.randn(2, 4, requires_grad=True)
    teacher = SceneFutureTeacher().eval()
    posterior = teacher.scene_posterior(
        future, velocity, last, prior_logits, scene)
    assert posterior["prob"].shape == (2, 4)
    assert torch.isfinite(posterior["prob"]).all()
    posterior["prob"].square().sum().backward()
    assert prior_logits.grad is None
    assert teacher.posterior_head[0].in_features == 128
    descriptor = future_pair_descriptor(future, last, edge)
    relation = teacher.relation_posterior(descriptor)["prob"]
    reversed_edge = edge.flip(0)
    reversed_descriptor = future_pair_descriptor(future, last, reversed_edge)
    reversed_relation = teacher.relation_posterior(reversed_descriptor)["prob"]
    assert torch.allclose(relation, reversed_relation, atol=1e-6)


def test_unary_zero_initialization_equals_candidate_prior():
    _, _, agent, last, goals, _, _ = _tiny()
    log_prior = torch.log_softmax(torch.randn(4, 5), dim=-1)
    unary = UnaryGoalResidual(prior_temperature=1.0)
    output = unary(agent, goals, last, log_prior)
    assert torch.equal(output["score"], log_prior)
    assert torch.count_nonzero(output["residual"]) == 0


def test_unary_candidate_mask_is_hard_and_finite_on_valid_entries():
    _, _, agent, last, goals, _, _ = _tiny()
    log_prior = torch.log_softmax(torch.randn(4, 5), dim=-1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[:, -1] = False
    output = UnaryGoalResidual()(agent, goals, last, log_prior, mask)
    assert torch.isneginf(output["score"][:, -1]).all()
    assert torch.isfinite(output["score"][:, :-1]).all()


def test_dynamic_relation_geometry_is_reversal_invariant_and_hypothesis_aware():
    _, edge, _, last, goals, _, base = _tiny()
    relation = DynamicHypothesisRelation(graph_radius=6.0).eval()
    geometry = relation.geometry(
        goals[edge[0], :, None], goals[edge[1], None, :],
        last[edge[0], None, None], last[edge[1], None, None])
    reverse = relation.geometry(
        goals[edge[1], None, :], goals[edge[0], :, None],
        last[edge[1], None, None], last[edge[0], None, None])
    assert torch.allclose(geometry, reverse, atol=1e-6)
    full = relation.full_pair_relation(base, goals, last, edge)
    assert full["prob"].shape == (2, 4, 5, 5, 4)
    assert not torch.equal(full["prob"][:, :, 0, 0],
                           full["prob"][:, :, 1, 1])


def test_full_relation_matches_selected_neighbor_gather():
    scene, edge, _, last, goals, _, base = _tiny()
    relation = DynamicHypothesisRelation(graph_radius=6.0).eval()
    full = relation.full_pair_relation(base, goals, last, edge)["prob"]
    selected = torch.tensor([[1, 2], [3, 0], [2, 4], [1, 3]])
    scene_mode = torch.tensor([[0, 2], [1, 3]])
    edge_mode = scene_mode[scene[edge[0]]]
    actual = relation.selected_neighbor_relation(
        base, goals, last, edge, selected, edge_mode, "source")["prob"]
    expected = torch.empty_like(actual)
    for e in range(edge.shape[1]):
        for p in range(2):
            expected[e, p] = full[
                e, edge_mode[e, p], :, selected[edge[1, e], p]]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_relation_specific_energy_and_logsumexp_reference():
    _, edge, agent, last, goals, edge_feat, _ = _tiny()
    model = RelationSpecificJointEnergy().eval()
    factors = model.factors(agent, goals, last, edge, edge_feat)
    energy = model.relation_energy(
        factors["left_factor"], factors["right_factor"])
    assert energy.shape == (2, 4, 5, 5, 4)
    e, z, k, l, m = 0, 1, 2, 3, 0
    manual = -torch.dot(
        factors["left_factor"][e, z, m, k].float(),
        factors["right_factor"][e, z, m, l].float()) / math.sqrt(8)
    assert torch.allclose(energy[e, z, k, l, m], manual)
    relation_log = torch.log_softmax(torch.randn_like(energy), dim=-1)
    effective = model.effective_energy(energy, relation_log)
    reference = -torch.logsumexp(relation_log - energy, dim=-1)
    assert torch.equal(effective, reference)


def test_strict_no_z_component_shapes_and_parameter_namespace():
    scene, edge, agent, last, goals, edge_feat, base = _tiny()
    teacher = SceneFutureTeacher(use_scene_latent=False)
    unary = UnaryGoalResidual(use_scene_latent=False)
    relation = DynamicHypothesisRelation(
        graph_radius=6.0, use_scene_latent=False).eval()
    energy = RelationSpecificJointEnergy(use_scene_latent=False).eval()
    sampler = ParallelConditionalSampler(strict_no_z=True).eval()

    assert set(teacher.state_dict()) == {
        'relation_teacher.0.weight', 'relation_teacher.0.bias',
        'relation_teacher.2.weight', 'relation_teacher.2.bias',
        'relation_teacher.4.weight', 'relation_teacher.4.bias'}
    assert not any('scene' in key or 'posterior' in key
                   for key in unary.state_dict())
    assert relation.query.shape == (4, 16)
    assert relation.bias.shape == (4,)
    assert not any('scene_embedding' in key for key in energy.state_dict())

    full_relation = relation.full_pair_relation(
        base, goals, last, edge, mode_enabled=False)
    assert full_relation['prob'].shape == (2, 5, 5, 4)
    factors = energy.factors(
        agent, goals, last, edge, edge_feat, mode_enabled=False)
    assert factors['left_factor'].shape == (2, 4, 5, 8)
    relation_energy = energy.relation_energy(
        factors['left_factor'], factors['right_factor'])
    assert relation_energy.shape == (2, 5, 5, 4)
    effective = energy.effective_energy(
        relation_energy, full_relation['log_prob'])
    assert effective.shape == (2, 5, 5)

    sampled = sampler(
        torch.randn(4, 5), goals, None, scene, edge, edge_feat, agent, last,
        base, relation, energy, sampling_mode='map',
        use_scene_latent=False)
    assert sampled['candidate_index'].shape == (4, 20)
    assert sampled['initial_candidate_index'].shape == (4, 20)
    assert sampled['relation_prob'].shape == (2, 20, 4)
    assert not ({'scene_mode', 'agent_scene_mode', 'edge_scene_mode'} &
                sampled.keys())


def test_strict_no_z_empty_edge_and_permutation_contracts():
    scene, edge, agent, last, goals, edge_feat, base = _tiny()
    relation = DynamicHypothesisRelation(
        graph_radius=6.0, use_scene_latent=False).eval()
    energy = RelationSpecificJointEnergy(use_scene_latent=False).eval()
    original = relation.full_pair_relation(
        base, goals, last, edge, mode_enabled=False)['prob']

    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    new_edge = inverse[edge]
    flip = new_edge[0] > new_edge[1]
    new_edge[:, flip] = new_edge.flip(0)[:, flip]
    order = torch.argsort(new_edge[0] * 4 + new_edge[1])
    new_edge = new_edge[:, order]
    permuted = relation.full_pair_relation(
        base[order], goals[permutation], last[permutation], new_edge,
        mode_enabled=False)['prob']
    assert torch.allclose(original[order], permuted, atol=1e-6)

    empty_edge = torch.empty((2, 0), dtype=torch.long)
    empty_base = torch.empty((0, 4))
    empty_feat = torch.empty((0, EDGE_FEATURE_DIM))
    empty_relation = relation.full_pair_relation(
        empty_base, goals[:1], last[:1], empty_edge,
        mode_enabled=False)['prob']
    empty_factors = energy.factors(
        agent[:1], goals[:1], last[:1], empty_edge, empty_feat,
        mode_enabled=False)
    assert empty_relation.shape == (0, 5, 5, 4)
    assert empty_factors['left_factor'].shape == (0, 4, 5, 8)


def test_energy_endpoint_reversal_transposes_candidate_axes():
    _, edge, agent, last, goals, edge_feat, _ = _tiny()
    model = RelationSpecificJointEnergy().eval()
    forward = model.factors(agent, goals, last, edge, edge_feat)
    forward_energy = model.relation_energy(
        forward["left_factor"], forward["right_factor"])
    reverse_edge = edge.flip(0)
    reverse = model.factors(
        agent, goals, last, reverse_edge, reverse_edge_features(edge_feat))
    reverse_energy = model.relation_energy(
        reverse["left_factor"], reverse["right_factor"])
    assert torch.allclose(
        forward_energy, reverse_energy.transpose(2, 3), atol=1e-6)


def test_pseudo_likelihood_no_edges_reduces_to_unary_scene_balanced():
    scene, _, _, _, goals, _, _ = _tiny()
    unary = torch.randn(4, 5)
    target = build_soft_goal_target(goals, goals[:, 0], sigma_goal=1.0)
    pz = torch.softmax(torch.randn(2, 4), dim=-1)
    edge = torch.empty((2, 0), dtype=torch.long)
    energy = torch.empty((0, 4, 5, 5))
    loss = jdv2_pseudo_likelihood(unary, target, pz, scene, edge, energy)
    expected_agent = -(target * torch.log_softmax(unary, dim=-1)).sum(-1)
    expected = 0.5 * (expected_agent[:2].mean() + expected_agent[2:].mean())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_scene_mode_score_and_mixture_match_manual_enumeration():
    torch.manual_seed(4)
    scene = torch.tensor([0, 0, 0, 1])
    unary = torch.randn(4, 3, requires_grad=True)
    target = torch.softmax(torch.randn(4, 3), dim=-1)
    local = torch.randn(4, 4, 3, requires_grad=True)
    score = jdv2_scene_mode_log_score(unary, target, scene, local)
    conditional = torch.log_softmax(unary[:, None] - local, dim=-1)
    per_agent = (target[:, None] * conditional).sum(dim=-1)
    expected = torch.stack((per_agent[:3].mean(0), per_agent[3:].mean(0)))
    torch.testing.assert_close(score, expected)

    prior_logits = torch.randn(2, 4, requires_grad=True)
    post = score + 0.2
    result = jdv2_mixture_composite_loss(prior_logits, post, score)
    combined = 0.5 * (post + score)
    expected_joint = torch.log_softmax(prior_logits, -1) + combined
    torch.testing.assert_close(
        result["loss"], -torch.logsumexp(expected_joint, -1).mean())
    torch.testing.assert_close(
        result["responsibility"], torch.softmax(expected_joint, -1))
    assert result["responsibility"].shape == (2, 4)


def test_mixture_and_posterior_distillation_gradient_ownership():
    prior_logits = torch.tensor(
        [[0.2, -0.1, 0.4, -0.3]], requires_grad=True)
    post_score = torch.tensor(
        [[-1.0, -0.2, -0.7, -1.3]], requires_grad=True)
    prior_score = torch.tensor(
        [[-0.8, -0.3, -0.9, -1.1]], requires_grad=True)
    mixture = jdv2_mixture_composite_loss(
        prior_logits, post_score, prior_score)
    mixture["loss"].backward(retain_graph=True)
    p = torch.softmax(prior_logits.detach(), -1)
    gamma = mixture["responsibility"].detach()
    torch.testing.assert_close(prior_logits.grad, p - gamma)
    assert post_score.grad is not None and prior_score.grad is not None

    prior_logits.grad = None
    post_score.grad = None
    prior_score.grad = None
    q_logits = torch.randn(1, 4, requires_grad=True)
    q_loss = jdv2_posterior_distillation(
        mixture["responsibility"], q_logits)
    q_loss.backward()
    assert q_logits.grad is not None
    assert prior_logits.grad is None
    assert post_score.grad is None
    assert prior_score.grad is None


def test_future_posterior_uses_future_evidence_not_history_features():
    teacher = SceneFutureTeacher().eval()
    prior_logits = torch.randn(2, 4)
    future_a = torch.randn(2, 128)
    future_b = future_a.clone()
    future_b[0] += 1.0
    output_a = teacher.posterior_from_future_scene(future_a, prior_logits)
    output_b = teacher.posterior_from_future_scene(future_b, prior_logits)
    assert not torch.equal(output_a["prob"][0], output_b["prob"][0])
    assert torch.equal(output_a["prob"][1], output_b["prob"][1])
    torch.testing.assert_close(
        output_a["centered_evidence_logits"].mean(-1),
        torch.zeros(2), atol=1e-6, rtol=0)


def test_relation_kl_and_scene_kl_are_finite_with_empty_edge():
    scene, _, _, _, goals, _, _ = _tiny()
    target = build_soft_goal_target(goals, goals[:, 0], sigma_goal=1.0)
    qz_log = torch.log_softmax(torch.randn(2, 4), dim=-1)
    pz_log = torch.log_softmax(torch.randn(2, 4), dim=-1)
    assert torch.isfinite(jdv2_scene_kl(qz_log, pz_log))
    edge = torch.empty((2, 0), dtype=torch.long)
    qrel = torch.empty((0, 4))
    prel = torch.empty((0, 4, 5, 5, 4), requires_grad=True)
    loss = jdv2_relation_kl(qrel, prel, qz_log.exp(), target, edge, scene)
    assert loss.item() == 0
    assert loss.requires_grad


def test_relation_kl_detaches_scene_responsibility_weight():
    scene = torch.tensor([0, 0])
    edge = torch.tensor([[0], [1]])
    target = torch.softmax(torch.randn(2, 3), dim=-1)
    teacher_log = torch.log_softmax(
        torch.randn(1, 4, requires_grad=True), dim=-1)
    deploy_logits = torch.randn(1, 4, 3, 3, 4, requires_grad=True)
    deploy_log = torch.log_softmax(deploy_logits, dim=-1)
    gamma = torch.softmax(
        torch.randn(1, 4, requires_grad=True), dim=-1)
    gamma.retain_grad()
    loss = jdv2_relation_kl(
        teacher_log, deploy_log, gamma, target, edge, scene)
    loss.backward()
    assert gamma.grad is None
    assert deploy_logits.grad is not None


def test_mode_allocation_and_sampler_joint_columns():
    scene, edge, agent, last, goals, edge_feat, base = _tiny()
    allocation = mode_stratified_allocation(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 20)
    assert torch.bincount(allocation[0], minlength=4).tolist() == [2, 4, 6, 8]
    relation = DynamicHypothesisRelation(graph_radius=6.0).eval()
    energy = RelationSpecificJointEnergy().eval()
    sampler = ParallelConditionalSampler().eval()
    unary = torch.randn(4, 5)
    pz = torch.softmax(torch.randn(2, 4), dim=-1)
    output = sampler(
        unary, goals, pz, scene, edge, edge_feat, agent, last, base,
        relation, energy, sampling_mode="map")
    assert output["candidate_index"].shape == (4, 20)
    assert output["goals"].shape == (4, 20, 2)
    assert output["scene_mode"].shape == (2, 20)
    assert output["relation_prob"].shape == (2, 20, 4)
    assert torch.equal(output["agent_scene_mode"], output["scene_mode"][scene])


def test_sampler_respects_candidate_masks():
    scene, edge, agent, last, goals, edge_feat, base = _tiny()
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, 2] = True
    output = ParallelConditionalSampler()(
        torch.randn(4, 5), goals, torch.full((2, 4), 0.25), scene,
        edge, edge_feat, agent, last, base,
        DynamicHypothesisRelation(6.0), RelationSpecificJointEnergy(),
        candidate_mask=mask, sampling_mode="sample")
    assert torch.equal(output["candidate_index"],
                       torch.full((4, 20), 2, dtype=torch.long))


def test_corrector_zero_init_gradient_and_zero_edge_contract():
    _, edge, _, last, _, _, _ = _tiny()
    model = DependencyCorrector(dt=0.4)
    velocity = torch.randn(4, 12, 2, requires_grad=True)
    relation = torch.randn(edge.shape[1], 16)
    residual = model(velocity, last, edge, relation, torch.tensor(10))
    assert torch.count_nonzero(residual) == 0
    residual.sum().backward()
    assert model.output[-1].weight.grad is not None
    empty = model(
        velocity.detach(), last, torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 16)), torch.tensor(10))
    assert torch.equal(empty, torch.zeros_like(empty))


def test_corrector_has_no_cross_scene_or_isolated_agent_leakage():
    model = DependencyCorrector(dt=0.4)
    with torch.no_grad():
        model.output[-1].weight.normal_()
        model.output[-1].bias.normal_()
    velocity = torch.randn(4, 12, 2)
    last = torch.randn(4, 2)
    edge = torch.tensor([[0], [1]])
    relation = torch.randn(1, 16)
    original = model(velocity, last, edge, relation, torch.tensor(5))
    changed_velocity = velocity.clone()
    changed_velocity[2:] += 100
    changed = model(changed_velocity, last, edge, relation, torch.tensor(5))
    assert torch.equal(original[:2], changed[:2])
    assert torch.equal(original[2:], torch.zeros_like(original[2:]))
    assert torch.equal(changed[2:], torch.zeros_like(changed[2:]))


def test_model_components_are_agent_permutation_equivariant():
    scene, edge, agent, last, goals, edge_feat, base = _tiny()
    prior = SceneLatentPrior().eval()
    relation = DynamicHypothesisRelation(6.0).eval()
    unary = UnaryGoalResidual().eval()
    log_prior = torch.log_softmax(torch.randn(4, 5), dim=-1)
    original_prior = prior(agent, scene)["prob"]
    original_unary = unary(agent, goals, last, log_prior)["score"]
    original_relation = relation.full_pair_relation(
        base, goals, last, edge)["prob"]

    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    new_edge = inverse[edge]
    flip = new_edge[0] > new_edge[1]
    new_edge[:, flip] = new_edge.flip(0)[:, flip]
    order = torch.argsort(new_edge[0] * 4 + new_edge[1])
    new_edge = new_edge[:, order]
    new_base = base.clone()
    new_base[flip] = new_base[flip]  # physical relation logits are symmetric
    new_base = new_base[order]
    permuted_prior = prior(agent[permutation], scene[permutation])["prob"]
    permuted_unary = unary(
        agent[permutation], goals[permutation], last[permutation],
        log_prior[permutation])["score"]
    permuted_relation = relation.full_pair_relation(
        new_base, goals[permutation], last[permutation], new_edge)["prob"]
    assert torch.allclose(original_prior, permuted_prior, atol=1e-6)
    assert torch.allclose(original_unary, permuted_unary[inverse], atol=1e-6)
    assert torch.allclose(
        original_relation[order], permuted_relation, atol=1e-6)


def test_bfloat16_autocast_smoke_is_finite_on_cpu():
    scene, _, agent, _, _, _, _ = _tiny()
    model = SceneLatentPrior().eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(agent, scene)
    assert torch.isfinite(output["prob"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fp16_autocast_smoke_is_finite_on_cuda():
    scene, _, agent, _, _, _, _ = _tiny()
    model = SceneLatentPrior().cuda().eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(agent.cuda(), scene.cuda())
    assert torch.isfinite(output["prob"]).all()


def test_crowded_sparse_scene_selected_path_stress():
    torch.manual_seed(19)
    n, k, edge_count = 64, 21, 256
    pairs = torch.triu_indices(n, n, offset=1)[:, :edge_count]
    scene = torch.zeros(n, dtype=torch.long)
    agent = torch.randn(n, 128)
    last = torch.randn(n, 2)
    goals = last[:, None] + torch.randn(n, k, 2)
    edge_feat = torch.randn(edge_count, EDGE_FEATURE_DIM)
    base = torch.randn(edge_count, 4)
    started = time.perf_counter()
    output = ParallelConditionalSampler()(
        torch.randn(n, k), goals, torch.full((1, 4), 0.25), scene,
        pairs, edge_feat, agent, last, base,
        DynamicHypothesisRelation(6.0), RelationSpecificJointEnergy(),
        sampling_mode="map")
    elapsed = time.perf_counter() - started
    assert output["candidate_index"].shape == (n, 20)
    assert output["relation_prob"].shape == (edge_count, 20, 4)
    assert elapsed < 30


@pytest.mark.parametrize("progress,expected", [
    (0.0, 0.0), (0.1, 0.05), (0.2, 0.1), (1.0, 0.1)])
def test_kl_warmup(progress, expected):
    assert jdv2_warmup_beta(progress) == pytest.approx(expected)


@pytest.mark.parametrize("progress,expected", [
    (0.0, 1.0), (0.2, 1.0), (0.4, 0.5), (0.6, 0.0), (1.0, 0.0)])
def test_teacher_curriculum(progress, expected):
    assert jdv2_teacher_probability(progress) == pytest.approx(expected)
