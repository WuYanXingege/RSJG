import copy
import itertools
import json
from pathlib import Path
import sys

import torch

from ebjd.finite_k_risk import discrete_best_of_k, finite_k_endpoint_risk
from ebjd.spatial_density_goal import SpatialGoalDensity
from test_unified_coupling import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_order_statistic_matches_enumeration_and_improves_with_k():
    d = torch.tensor([3., .2, 1.], dtype=torch.float64)
    p = torch.tensor([.2, .3, .5], dtype=torch.float64, requires_grad=True)
    expected = sum(min(d[i] for i in ids) * p[list(ids)].prod()
                   for ids in itertools.product(range(3), repeat=3))
    torch.testing.assert_close(discrete_best_of_k(d, p, 3), expected)
    torch.testing.assert_close(discrete_best_of_k(d, p, 1), (d*p).sum())
    assert discrete_best_of_k(d, p, 20) < discrete_best_of_k(d, p, 3)
    torch.autograd.grad(discrete_best_of_k(d, p, 20), p)


def test_coverage_gradient_moves_probability_toward_nearer_endpoint():
    logits = torch.zeros(2, requires_grad=True)
    loss = discrete_best_of_k(torch.tensor([.1, 2.]), logits.softmax(-1), 20)
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]


def test_full_support_quadrature_finite_gradients_rng_and_padding():
    torch.manual_seed(33)
    ctx = context(); head = SpatialGoalDensity(32, grid_size=8)
    ctx.social_h.requires_grad_(); ctx.map_features.requires_grad_()
    truth = ctx.map_center + torch.tensor([1., 2.])
    truth[0, 0] += 40.
    truth[~ctx.valid] = float('nan')
    d = head(ctx); rng = torch.get_rng_state().clone()
    before = copy.deepcopy(head.state_dict())
    risk = finite_k_endpoint_risk(head, ctx, truth, torch.tensor(1.2), d)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(risk) and risk > 0
    risk.backward()
    for module in (head.spatial_features, head.social_film, head.spatial_readout, head.tail):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert ctx.map_features.grad.abs().sum() > 0
    head.load_state_dict(before, strict=True)


def test_formal_grid_quadrature_matches_continuous_sampling_expectation():
    # Acceptance concerns the fixed 64x64 grid, not an arbitrarily coarse grid.
    # A 4x4 diagnostic has ~0.3m order-four bias and is explicitly NOT accepted.
    from dataclasses import replace
    torch.manual_seed(81)
    ctx = context(batch=1, agents=1)
    head = SpatialGoalDensity(32, grid_size=64)
    d = head(ctx); truth = ctx.map_center + 1.
    values = [float(finite_k_endpoint_risk(head, ctx, truth, torch.tensor(1.), d, order=o).detach()) for o in (2,4)]
    total = 10000
    c = replace(ctx, valid=ctx.valid.expand(total,-1), social_h=ctx.social_h.expand(total,-1,-1),
                map_center=ctx.map_center.expand(total,-1,-1))
    dd = type(d)(*(v.expand(total,*v.shape[1:]) for v in (d.spatial_logits,d.tail_logit,d.side_logits,d.tail_scale)))
    g = head.sample(c, torch.tensor(1.), distribution=dd)
    sampled = torch.linalg.vector_norm(g-truth[:,None],dim=-1).min(1).values.mean().item()
    assert abs(values[0]-sampled) < .05
    assert abs(values[1]-values[0]) < .01


def test_grouped_objective_preserves_scene_and_agent_weighting():
    from conftest import make_batch
    from ebjd.model import EBJDModel
    from ebjd.spatial_density_goal import SpatialDensityCouplingBridge
    from train_learned_goal_bridge import objective
    torch.set_num_threads(2);torch.manual_seed(72)
    model=SpatialDensityCouplingBridge(EBJDModel(hidden_dim=32,attention_heads=4,ff_dim=64,
        proposal_conditioned=True,block_type='endpoint_gated_dit_v1',coarse_blocks=4,refinement_blocks=4),grid_size=8)
    model.train();batch=make_batch((1,2,3),pixels=32)
    batch.agent_ids=[ids[:count] for ids,count in zip(batch.agent_ids,(1,2,3))]
    contract=json.loads((Path(__file__).resolve().parents[1]/'configs/research/finite_k_density_bridge_v1.json').read_text())
    times=torch.tensor([.2,.7,.9]);noise=torch.randn(3,1,3,12,2);goals=torch.randn(3,3,20,3)
    serial=0.
    for i,count in enumerate((1,2,3)):
        a,s,_=objective(model,batch.compact_scenes(i,i+1),contract,time_draw=times[i:i+1],
            noise=noise[i:i+1,:,:count],goal_noise=goals[i:i+1,:count])
        serial=serial+a*count/6+s/3
    serial.backward();grad={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    a,s,_=objective(model,batch,contract,time_draw=times,noise=noise,goal_noise=goals)
    grouped=a+s;grouped.backward()
    torch.testing.assert_close(serial,grouped,rtol=2e-5,atol=2e-5)
    for n,p in model.named_parameters():
        if n in grad:torch.testing.assert_close(grad[n],p.grad,rtol=1e-3,atol=2e-5)


def test_fp32_assignment_scores_keep_gradient_under_autocast():
    from ebjd.unified import LearnableJointCouplingHead, joint_pair_loss
    ctx=context();ctx.social_h.requires_grad_()
    head=LearnableJointCouplingHead(32,singleton_backward='identity')
    head.score_in_fp32=True
    with torch.autocast('cpu',dtype=torch.bfloat16):
        g=torch.randn(2,20,3,2)
        h,indices,scores=head.assign(ctx,g,torch.tensor(1.),straight_through=True)
        loss=joint_pair_loss(h,torch.randn(2,3,2),ctx.valid,torch.tensor(1.))
    assert scores.dtype==torch.float32
    loss.backward()
    assert ctx.social_h.grad is not None and torch.isfinite(ctx.social_h.grad).all()
    assert sum(p.grad.abs().sum() for p in head.parameters() if p.grad is not None)>0
