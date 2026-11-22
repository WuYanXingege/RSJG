"""Small real-window interface checks; no optimizer or parameter update occurs."""
import csv
import hashlib
import io
import json
import random
import subprocess
from unittest.mock import patch

import numpy as np
import torch
from bcep.motion_frame import frame, to_local, to_world
from bcep_joint.common import write
from bcep_joint.inference import generate as native_h0
from ebjd.metrics import hard_metric_statistics
from ebjd.unified import gather_candidates
from . import inference
from .inputs import ARCHIVE, ARCHIVE_DIR, CHECKPOINT_SHA, RUN, SOURCE
from .model import delta_nll, sample_latents
from .sampling import bridge_noise, shuffled_correspondence


def _archive():
    def blob(name):
        return subprocess.check_output(['git', 'show', ARCHIVE+':'+ARCHIVE_DIR+name], cwd=SOURCE)
    raw = blob('DEV_WINDOWS.csv')
    hashes = json.loads(blob('ARTIFACT_SHA256.json'))
    assert hashlib.sha256(raw).hexdigest() == hashes['DEV_WINDOWS.csv']
    reference = json.loads(blob('REFERENCE_IDENTITY.json'))
    assert reference['identity']['checkpoint_sha256'] == CHECKPOINT_SHA
    assert reference['identity']['candidate_seed'] == 2035
    assert reference['identity']['K'] == reference['identity']['steps'] == 20
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == 590 and sum(int(r['agents']) >= 9 for r in rows) == 127
    return rows, dict(commit=ARCHIVE, csv_sha256=hashes['DEV_WINDOWS.csv'])


@torch.no_grad()
def _latent_truth(provider, observed, context, truth):
    origin, rotation, scale = frame(observed, 12, provider.base.dt, provider.base.scales[-1])
    local = to_local(truth, origin, rotation, scale)
    z, _ = provider.base.flow.transform(local, context, 12*provider.base.dt, inverse=True)
    restored_local, _ = provider.base.flow.transform(z, context, 12*provider.base.dt)
    restored_world = to_world(restored_local, origin, rotation, scale)
    # This is a numerical inverse check, not a bitwise claim about four FP32 splines.
    torch.testing.assert_close(restored_local, local, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(restored_world, truth, rtol=1e-5, atol=1e-4)
    return z.detach(), dict(local_max_abs=float((restored_local-local).abs().max()),
                            world_max_abs_m=float((restored_world-truth).abs().max()),
                            inverse_atol=1e-4, inverse_rtol=1e-5)


def _matrix_and_gradient(dependence, observed, context, z):
    dependence.zero_grad(set_to_none=True)
    with torch.enable_grad():
        pack = dependence(observed.detach(), context.detach())
        covariance = pack['R']
        _, info = torch.linalg.cholesky_ex(covariance.detach())
        assert int(info) == 0
        n = len(observed)
        diagonal = torch.stack([covariance[2*i:2*i+2, 2*i:2*i+2] for i in range(n)])
        identity = torch.eye(2, device=covariance.device).expand_as(diagonal)
        diagonal_error = float((diagonal.detach()-identity).abs().max())
        torch.testing.assert_close(diagonal.detach(), identity, rtol=0, atol=2e-6)
        minimum_eigenvalue = float(torch.linalg.eigvalsh(covariance.detach().cpu().double()).min())
        assert minimum_eigenvalue > 0
        loss = delta_nll(covariance, z.detach())
        assert torch.isfinite(loss) and loss.requires_grad
        loss.backward()
        gradients = [p.grad for p in dependence.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        first_grad = float(dependence.net[0].weight.grad.norm())
        last_grad = float(dependence.net[-1].weight.grad.norm())
        if n == 1:
            assert torch.equal(covariance.detach(), torch.eye(2, device=covariance.device))
            assert float(loss.detach()) == 0 and first_grad == last_grad == 0
        else:
            assert first_grad > 0 and last_grad > 0
    return {k: v.detach() if torch.is_tensor(v) else v for k, v in pack.items()}, dict(
        agents=n, edges=len(pack['edges']), R_min_eigenvalue=minimum_eigenvalue,
        R_diagonal_max_abs=diagonal_error, initial_delta_nll_nat_per_person=float(loss.detach()),
        first_layer_gradient_l2=first_grad, last_layer_gradient_l2=last_grad,
        singleton_connected_zero=n == 1)


def preflight(inputs, model, provider, dependence, device):
    """Run three dev replays plus a maximum-size train covariance/backward check.

    The dependency weights, all RNG streams, gradients, and module train/eval
    modes are restored. The caller therefore starts the one official fit from
    its unchanged seed-3201 initialization.
    """
    device = torch.device(device)
    models = (model, provider, dependence)
    modes = [(m, m.training) for network in models for m in network.modules()]
    parameters = list(dependence.parameters())
    old_grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    before = {k: v.detach().clone() for k, v in dependence.state_dict().items()}
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
    results = []
    assert not any(p.requires_grad for network in (model, provider) for p in network.parameters())
    archived, archive_identity = _archive()
    ids = inputs.split['selection_dev_indices']
    checks = (lambda n: n == 1, lambda n: 2 <= n <= 4, lambda n: n >= 9)
    positions = [next(pos for pos, idx in enumerate(ids)
                      if criterion(len(inputs.split['members'][idx]['agent_ids']))) for criterion in checks]
    try:
        for network in models:
            network.eval()
        for pos in positions:
            idx = ids[pos]
            host_batch = inputs.data.batch([idx], 'selection_dev', truth=False)
            batch = host_batch.to(device)
            n = int(batch.valid.sum())
            assert batch.valid.all() and not batch.future.count_nonzero()
            calls = dict(correction=0, denoiser=0, ddim=[])
            def head_called(*_):
                calls['correction'] += 1
            def denoiser_called(*_):
                calls['denoiser'] += 1
            def denied(*_, **__):
                raise AssertionError('new route invoked the old pairing/provider-candidate path')
            original_ddim = inference.differentiable_sample
            def counted_ddim(*args, **kwargs):
                calls['ddim'].append(dict(steps=args[3] if len(args) > 3 else kwargs['steps'],
                                         eta=kwargs['proposal_eta']))
                return original_ddim(*args, **kwargs)
            head_hook = provider.head.register_forward_hook(head_called)
            denoiser_hook = model.bridge.denoiser.register_forward_hook(denoiser_called)
            try:
                with patch.object(model.coupling, 'assign', denied), \
                     patch.object(provider, 'candidates', denied), \
                     patch.object(inference, 'differentiable_sample', counted_ddim):
                    predicted, endpoints, detail = inference.generate(model, provider, dependence, batch)
            finally:
                head_hook.remove(); denoiser_hook.remove()
            assert calls == dict(correction=1, denoiser=20, ddim=[dict(steps=20, eta=0.)])
            assert detail['ddim_calls'] == detail['correction_calls'] == 1
            selected_ids = endpoints['raw_indices']
            assert len(selected_ids.unique()) == 20 and (selected_ids >= 0).all() and (selected_ids < 256).all()
            assert torch.equal(endpoints['selected'], endpoints['raw'][selected_ids])
            assert torch.equal(endpoints['noise'], bridge_noise(batch.scene_ids[0], batch.agent_ids[0], selected_ids.cpu().tolist()).to(device))

            # Future loading happens after the observed-only prediction is complete.
            truth = inputs.data.batch([idx], 'selection_dev', truth=True).future.to(device)
            z, inverse = _latent_truth(provider, batch.observed[0], detail['flow_context'], truth[0, :, -1])
            pack, matrix = _matrix_and_gradient(dependence, batch.observed[0], detail['flow_context'], z)
            torch.testing.assert_close(pack['R'], endpoints['R'], rtol=0, atol=0)
            sampled = sample_latents(pack, endpoints['epsilon'], endpoints['xi'])
            # Independent direct u/L construction checks the factor's coordinate ordering.
            u = endpoints['epsilon'].clone()
            for e, (i, j) in enumerate(pack['edges'].tolist()):
                u[:, i] += endpoints['xi'][:, e] @ pack['B'][e, 0].T
                u[:, j] += endpoints['xi'][:, e] @ pack['B'][e, 1].T
            direct = torch.linalg.solve_triangular(pack['L'][None], u[..., None], upper=False).squeeze(-1)
            torch.testing.assert_close(sampled, direct, rtol=1e-5, atol=1e-6)
            if n == 1:
                assert torch.equal(sampled, endpoints['epsilon'])
            endpoint_minfde = inference.endpoint_minfde(endpoints['corrected'][0], truth[0, :, -1])
            for repetition in range(5):
                goals, noise, pi = shuffled_correspondence(endpoints['corrected'], endpoints['noise'],
                    batch.scene_ids[0], batch.agent_ids[0], repetition)
                assert torch.equal(gather_candidates(goals, pi.argsort(1)), endpoints['corrected'])
                assert torch.equal(gather_candidates(noise, pi.argsort(1)), endpoints['noise'])
                assert torch.equal(inference.endpoint_minfde(goals[0], truth[0, :, -1]), endpoint_minfde)
                if n == 1:
                    assert torch.equal(goals, endpoints['corrected']) and torch.equal(noise, endpoints['noise'])

            historical_goals = inputs.banks['selection_dev'].batch([idx], host_batch).to(device)
            _, h0, _, _ = native_h0(model, batch.observed, batch.semantic_maps, batch.valid,
                batch.scene_ids, batch.agent_ids, historical_goals, 2035, mode='H')
            h0_stats = hard_metric_statistics(h0, truth, batch.valid)
            reference = archived[pos]
            assert reference['window'] == f'dev_{pos+1:04d}' and int(reference['agents']) == n
            replay_diff = dict(JADE=float(h0_stats.scene_jade)-float(reference['H0_JADE']),
                               JFDE=float(h0_stats.scene_jfde)-float(reference['H0_JFDE']))
            assert max(abs(v) for v in replay_diff.values()) <= 1e-6
            results.append(dict(window=f'dev_{pos+1:04d}', **matrix, inverse=inverse,
                native_H0_minus_archive_m=replay_diff, native_H0_replay_atol_m=1e-6,
                raw_ids_unique=True, raw_id_keyed_bridge_noise_exact=True,
                endpoint_noise_transport_exact=True, five_shuffles_endpoint_minfde_exact=True,
                new_route_Hungarian_bypassed=True, new_route_old_candidates_bypassed=True,
                correction_calls=1, ddim_calls=1, denoiser_calls=20,
                sampler_vs_explicit_u_solve_max_abs=float((sampled-direct).abs().max()),
                endpoint_path_max_abs_m=float((predicted[..., -1, :]-endpoints['corrected']).abs().max())))

        largest = max(inputs.split['train_indices'], key=lambda idx: len(inputs.split['members'][idx]['agent_ids']))
        batch = inputs.data.batch([largest], 'train', truth=False).to(device)
        with torch.no_grad():
            context, _ = inference.encode_observed(provider, batch)
            truth = inputs.data.batch([largest], 'train', truth=True).future[0, :, -1].to(device)
        z, inverse = _latent_truth(provider, batch.observed[0], context, truth)
        _, largest_matrix = _matrix_and_gradient(dependence, batch.observed[0], context, z)
        largest_matrix['inverse'] = inverse
        assert largest_matrix['agents'] == 57
        assert all(torch.equal(before[k], value) for k, value in dependence.state_dict().items())
        assert all(p.grad is None for network in (model, provider) for p in network.parameters())
    finally:
        for parameter, grad in zip(parameters, old_grads):
            parameter.grad = grad
        for module, training in modes:
            module.training = training
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(rng['cuda'])
    assert all(torch.equal(before[k], value) for k, value in dependence.state_dict().items())
    receipt = dict(passed=True, real_dev_windows=results, maximum_train_matrix_check=largest_matrix,
        archive=archive_identity, dependency_weights_unchanged=True, all_rng_restored=True,
        gradients_restored=True, optimizer_constructions=0, optimizer_updates=0,
        same_R_for_likelihood_and_sampling=True, future_used_only_for_targets_and_offline_metrics=True)
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN/'PREFLIGHT.json', receipt)
    return receipt
