"""Constructor-free GDTS/trainer fixture. NO real network constructor/forward.

Only raw Parameters and callable algebra stubs replace encode/relation/energy.
The production dispatch, MC loss, coefficients, optimizer step, save/load and
progress methods are called unchanged. This is NOT a model smoke test.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from src.models.model import GDTS
from src.trainer import trainer
from src.jdv2_objective_state import MC, FIELD

COUNTS = dict(production_loss_calls=0, synthetic_encode_calls=0,
              relation_chunks=0, energy_factor_chunks=0, energy_mixtures=0,
              backward_calls=0, toy_optimizer_updates=0, draws=0, sampled_rows=0,
              checkpoint_save_calls=0, checkpoint_load_calls=0)


def args(directory, mode=MC):
    return SimpleNamespace(
        jdv2_goal_objective=mode, jdv2_neighbor_draws=4, jdv2_neighbor_seed=314159,
        seed=3101, device='cpu', phase='train', dataset='eth5', test_set='eth',
        goal_model_type='joint_dependency_v2', jdv2_active=True,
        jdv2_latent_objective='strict_no_z', training_stage='joint_goal',
        trajectory_coupling='none', amp_enabled=False, amp_dtype='fp32',
        goal_soft_sigma=1., jdv2_relation_modes=4, jdv2_edge_chunk_size=2,
        use_dynamic_relation=True, use_joint_energy=True, jdv2_stage_progress=0.,
        jdv2_cache_manifest_hash='synthetic-cache',
        jdv2_source_checkpoint_hash='synthetic-base',
        clean_split_manifest_hash='synthetic-split', clean_split_protocol=False,
        jdv2_cache_source_commit='dcf5fde1aa77f62a25a9209fd91e4f854a21510a',
        num_epochs=3, optimizer='Adam', scheduler='ExponentialLR',
        clip=1., learning_rate=.003, model_dir=str(directory),
        load_checkpoint='last', trajectory_alignment=False)


def leaf(shape, phase):
    t = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    return torch.nn.Parameter(.4*torch.sin(t*.37+phase))


def harness(directory, mode=MC, n=5, edges=True):
    item = trainer.__new__(trainer)
    item.args = args(directory, mode)
    item.device = torch.device('cpu')
    item.net = model = GDTS.__new__(GDTS)
    torch.nn.Module.__init__(model)
    model.args, model.device = item.args, item.device
    model.jdv2_active = model.strict_no_z = True
    model.last_joint_diagnostics = {}
    model.jdv2_raw_unary = leaf((n, 3), .2)
    model.jdv2_social = leaf((n, 3), .7)
    edge = (torch.tensor([[0, 0, 1], [1, 2, 3]]) if edges and n in (4, 5)
            else torch.empty((2, 0), dtype=torch.long))
    e = edge.shape[1]
    model.jdv2_deploy = leaf((e, 3, 3, 4), .3)
    model.jdv2_teacher = leaf((e, 4), .5)
    model.jdv2_left = leaf((e, 4, 3, 2), .8)
    model.jdv2_right = leaf((e, 4, 3, 2), 1.1)
    goals = torch.arange(n*3*2, dtype=torch.float32).reshape(n, 3, 2)/19
    scenes = torch.tensor([11]*4 if n == 4 else [11]*(n-1)+[9001])
    mask = torch.ones((n, 3), dtype=torch.bool)
    mask[-1, -1] = False
    inputs = dict(scene_index=scenes,
                  world_coord=torch.zeros((1, n, 2)),
                  obs_traj_world=torch.zeros((n, 1, 2)))

    def encode(inputs, if_test=False, for_loss=False):
        assert for_loss and not if_test
        COUNTS['synthetic_encode_calls'] += 1
        structured = dict(goal_candidates_world=goals, candidate_mask=mask,
                          unary_score=model.jdv2_raw_unary+.2*model.jdv2_social,
                          edge_index=edge, edge_feat=torch.arange(e)[:, None],
                          base_relation_logits=torch.arange(e)[:, None].expand(-1, 4),
                          agent_feat=model.jdv2_social,
                          relation_teacher={'log_prob': torch.log_softmax(model.jdv2_teacher, -1)})
        model.fixture_structured = structured
        return None, structured

    def relation(base, goals, last, chunk, mode_enabled):
        assert not mode_enabled
        COUNTS['relation_chunks'] += 1
        ids = base[:, 0].long()
        lp = torch.log_softmax(model.jdv2_deploy[ids], -1)
        return dict(log_prob=lp, prob=lp.exp())

    def factors(social, goals, last, chunk, feat, mode_enabled):
        assert not mode_enabled
        COUNTS['energy_factor_chunks'] += 1
        ids = feat[:, 0].long()
        return dict(left_factor=model.jdv2_left[ids]+.1*social[chunk[0], None, :, None],
                    right_factor=model.jdv2_right[ids])

    def effective(energy, lp):
        COUNTS['energy_mixtures'] += 1
        return -torch.logsumexp(lp-energy, -1)

    model.encode = encode
    model.jdv2_dynamic_relation = SimpleNamespace(full_pair_relation=relation)
    model.jdv2_joint_energy = SimpleNamespace(
        factors=factors, relation_energy=lambda l, r: -torch.einsum(
            'emkr,emlr->eklm', l, r)/math.sqrt(2), effective_energy=effective)
    model.best_valid_metric = lambda: 'JADE'
    item.optimizer = item._set_optimizer(model.parameters())
    item._optimizer_parameter_groups = lambda: list(model.parameters())
    item.scheduler = item._set_scheduler(item.optimizer)
    item.scaler = SimpleNamespace(is_enabled=lambda: False)
    item.data_loaders = {'train': [None]*3}
    item._pending_training_state = None
    item._stage_optimizer_steps_completed = 0
    item._best_selection = (float('inf'), float('inf'))
    item.best_metrics = {'JADE': float('inf')}
    item.best_metrics_epochs = {'JADE': -1}
    item._validations_without_improvement = item._collapse_signature_epochs = 0
    item._jdv2_architecture_config = lambda: {'raw_fixture': True}
    item._jdv2_ablation_config = lambda: {'fixture': True}
    item._jdv2_inference_sampler_config = lambda: {}
    item.curve_metric_names, item.curve_loss_names = [], []
    item.inputs, item.mask, item.edge = inputs, mask, edge
    if mode == MC:
        item._initialize_mc_objective()
        original = item.jdv2_objective_rng.draw
        def counted_draw(*values):
            z = original(*values)
            COUNTS['draws'] += 1
            COUNTS['sampled_rows'] += z.numel()
            item.last_ids = z.clone()
            return z
        item.jdv2_objective_rng.draw = counted_draw
    return item


def losses(item):
    COUNTS['production_loss_calls'] += 1
    return item.net._jdv2_goal_losses(item.inputs)


def plain(value):
    if torch.is_tensor(value):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def run_steps(item, start, stop):
    trace = []
    for step in range(start, stop):
        item.inputs['world_coord'].fill_(step/100)
        values = losses(item)
        coefficients = item.net.set_losses_coeffs()
        total = sum(coefficients[k]*v for k, v in values.items())
        total.backward()
        COUNTS['backward_calls'] += 1
        gradients = {k: p.grad.clone() if p.grad is not None else None
                     for k, p in item.net.named_parameters()}
        item._mc_optimizer_step()
        COUNTS['toy_optimizer_updates'] += 1
        if (step+1) % 3 == 0:
            item.scheduler.step()
            item._mc_complete_epoch((step+1)//3)
        trace.append(plain(dict(
            step=step, ids=item.last_ids, losses=values, total=total,
            coefficients=coefficients, gradients=gradients,
            parameters=item.net.state_dict(), optimizer=item.optimizer.state_dict(),
            scheduler=item.scheduler.state_dict(), progress=item.args.jdv2_stage_progress,
            rng_sha256=hashlib.sha256(bytes(item.jdv2_objective_rng.generator.get_state().tolist())).hexdigest(),
            draws=item.jdv2_objective_rng.draw_calls,
            rows=item.jdv2_objective_rng.sampled_agent_draws,
            updates=item._stage_optimizer_steps_completed)))
    return trace


def save(item, epoch=1, **flags):
    COUNTS['checkpoint_save_calls'] += 1
    return item._save_checkpoint(epoch, **flags)


def load(item, path, **kwargs):
    COUNTS['checkpoint_load_calls'] += 1
    return item._load_state_file(str(path), **kwargs)


if __name__ == '__main__':
    import contextlib
    import sys
    torch.set_num_threads(1)
    def forbidden(*a, **kw):
        raise AssertionError('No real module forward or CUDA in resume child')
    torch.nn.Module._call_impl = forbidden
    torch.cuda._lazy_init = forbidden
    item = harness(Path(sys.argv[1]))
    with contextlib.redirect_stdout(sys.stderr):
        start_epoch = item._load_or_restart()
    COUNTS['checkpoint_load_calls'] += 1
    assert start_epoch == 2
    result = run_steps(item, 3, 9)
    print(json.dumps({'trace': result, 'counts': COUNTS}, allow_nan=False))
