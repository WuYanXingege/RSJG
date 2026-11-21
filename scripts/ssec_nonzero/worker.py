"""Fresh-process, bounded synthetic updates using the unchanged native engine."""
import argparse
import contextlib
import fcntl
import json
import random
import time

import numpy as np
import torch

from bcep_joint.common import save, seed_all
from bcep_joint.inference import check_permutation, keyed_noise
from bcep_scene_energy.formal_callbacks import SlotModelView, SlotCouplingView
from bcep_scene_energy.formal_contract import model_hash
from bcep_scene_energy.formal_monitor import StepObserver, PREFIXES
from bcep_scene_energy.inference import generate
from bcep_scene_energy.model import build
from bcep_v2 import state as native
from bcep_v2.stages import joint_callbacks
from ebjd.unified import gather_candidates
from .common import OUT, FROZEN, frozen, read, write, sha, tensor_sha, cpu_copy, rng_state, worker_guard, setup_cuda
from .fixtures import SyntheticData, SyntheticBank

PHASES = dict(continuous=('continuous', False, 4, 4), partial=('split', False, 2, 2),
              resume=('split', True, 4, 2), replay_crash=('replay', True, 4, 1),
              replay_resume=('replay', True, 4, 2))


class PlannedInterruption(Exception):
    pass


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


class AuditedCoupling(SlotCouplingView):
    def __init__(self, original, observer):
        super().__init__(original, observer)
        self.events = []

    def assign(self, context, goals, scale, **kw):
        assert kw['straight_through'] and not goals.requires_grad
        before = tensor_sha(goals)
        hard, pi, scores, detail = super().assign(context, goals, scale, return_details=True, **kw)
        check_permutation(pi, context.valid)
        assert torch.equal(hard.detach(), gather_candidates(goals, pi))
        assert tensor_sha(goals) == before and torch.isfinite(scores).all()
        identity = detail['candidate_identity']
        assert 'transported_original_slot' in identity.source
        assert torch.equal(identity.to_input_indices(detail['canonical_final_pi']), pi)
        assert all(r['certified_monotone'] for r in self.last['observations'])
        self.events.append(dict(people=context.valid.sum(1).tolist(), original_candidate_sha256=before,
            initial_pi_sha256=tensor_sha(detail['initial_pi']), final_pi_sha256=tensor_sha(pi),
            score_sha256=tensor_sha(scores), canonical_mapping_sha256=tensor_sha(identity.canonical_to_input),
            original_slots_preserved=True, frozen_endpoints=True, bijection=True,
            decisions=self.last['observations'], trace=[d['trace'] for d in detail['scenes']]))
        return hard, pi, scores


class Recorder:
    def __init__(self, model, folder, phase, start, budget):
        self.model, self.folder, self.phase, self.start, self.budget = model, folder, phase, start, budget
        self.observer = StepObserver(model, folder / 'MODULE_UPDATES.jsonl', start)
        self.done = 0
        self.ready = False
        self.rows = []

    def callback(self, data, bank, config):
        assert type(data) is SyntheticData and type(bank) is SyntheticBank
        view = SlotModelView(self.model, self.observer)
        view.coupling = AuditedCoupling(self.model.coupling, self.observer)
        actual, _ = joint_callbacks(view, data, 'cuda', config, bank, bank)

        def backward(ids, order, views):
            assert len(ids) == 16 and self.done < self.budget
            self.observer.groups = []
            view.coupling.events = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            metrics = actual(ids, order, views)
            torch.cuda.synchronize()
            # Exact same detached monitoring as formal_callbacks; only the
            # data adapter/type gate differs. Loss/backward are native.
            self.observer.capture_backward(metrics, time.perf_counter() - start, torch.cuda.max_memory_allocated())
            scenes = [r for group in self.observer.groups for r in group['scenes']]
            for key in ['attempted', 'accepted', 'rejected', 'strict_ties', 'reliable_decreases', 'roundoff_uncertain', 'accepted_energy_delta']:
                metrics['ssec_' + key] = sum(r[key] for r in scenes)
            assert metrics['valid_scene_count'] == 16
            assert {1, 2, 9, 57} <= {n for g in self.observer.groups for n in g['valid_people']}
            assert all(self.observer.grad[p]['gradient_l2'] > 0 for p in PREFIXES)
            self.group, self.order_rng, self.view_rng = list(ids), order, views
            self.events, self.metrics, self.ready = view.coupling.events, metrics, True
            return metrics

        return backward

    def pre(self, opt, args, kwargs):
        assert self.ready and self.done < self.budget
        assert type(opt) is torch.optim.AdamW
        assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in self.model.parameters() if p.requires_grad}
        assert all(p.dtype == torch.float32 and p.is_cuda for p in self.model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
        for g in opt.param_groups:
            assert g['betas'] == (.9, .999) and g['eps'] == 1e-8 and g['lr'] == 2e-5
        self.clipped_gradients = {n: tensor_sha(p.grad) for n, p in self.model.named_parameters() if p.grad is not None}

    def post(self, opt, args, kwargs):
        # Registered AFTER the unmodified StepObserver hook. Its actual
        # parameter deltas have already been added to native diagnostics.
        self.done += 1
        self.ready = False
        update = self.start + self.done
        assert self.observer.update == update
        row = dict(phase=self.phase, logical_update=update, physical_optimizer_event=self.done,
                   real_optimizer_inputs=0, synthetic_windows=self.group)
        append(OUT / 'OPTIMIZER_EVENTS.jsonl', row)
        for p in PREFIXES:
            assert np.isfinite(self.metrics[p + 'parameter_delta_l2']) and self.metrics[p + 'parameter_delta_l2'] > 0
        assert all(torch.isfinite(p).all() for p in self.model.state_dict().values())
        checkpoint = dict(model=cpu_copy(self.model.state_dict()), optimizer=cpu_copy(opt.state_dict()),
            rng=cpu_copy(rng_state(self.order_rng, self.view_rng)), metrics=dict(self.metrics), ids=self.group,
            clipped_gradient_sha256=self.clipped_gradients, coupling_events=self.events,
            parameter_names=list(dict(self.model.named_parameters())), buffer_names=list(dict(self.model.named_buffers())))
        name = f'{self.phase}_update_{update:03d}.pt'
        save(checkpoint, self.folder / name)
        summary = dict(row, checkpoint=name, checkpoint_sha256=sha(self.folder / name),
                       model_sha256=model_hash(self.model), metrics=dict(self.metrics),
                       candidate_invariants=True, energy_monotone=True)
        self.rows.append(summary)
        append(self.folder / 'AUDIT_UPDATES.jsonl', summary)
        print(self.phase, 'logical update', update, 'PASS finite gradients/actual deltas and identities', flush=True)
        if self.phase == 'replay_crash':
            # Real AdamW update + durable MODULE_UPDATES entry, but native
            # state cursor/update and last.pt have NOT been committed yet.
            raise PlannedInterruption('after real update 3 journal, before checkpoint commit')

    @contextlib.contextmanager
    def hooks(self):
        with self.observer.native_optimizer_hook():
            factory = native.optimizer

            def audited(*args, **kwargs):
                opt = factory(*args, **kwargs)
                opt.register_step_pre_hook(self.pre)
                opt.register_step_post_hook(self.post)
                return opt

            native.optimizer = audited
            try:
                yield
            finally:
                native.optimizer = factory


@torch.no_grad()
def check_ddim(model, data, bank, folder):
    import bcep_joint.inference as inference
    rows = []
    initial_hash = model_hash(model)
    for index in data.manifest['first_epoch_order'][:4]:
        batch = data.batch([index], 'train')
        g = bank.batch([index], batch).cuda()
        x = batch.to('cuda')
        slots = torch.arange(20, device='cuda').expand(1, x.valid.shape[1], 20)
        original = inference.differentiable_sample
        captured = []

        def sample(*a, **kw):
            captured.append(dict(steps=a[3], noise=a[2].detach().cpu(), goals=kw['proposal_goals'].detach().cpu()))
            return original(*a, **kw)

        inference.differentiable_sample = sample
        try:
            gp, hp, pi, checks = generate(model, x.observed, x.semantic_maps, x.valid, x.scene_ids,
                x.agent_ids, g, 2035, candidate_ids=slots, mode='GH')
        finally:
            inference.differentiable_sample = original
        z = keyed_noise(x.scene_ids, x.agent_ids, x.valid, 2035)
        assert len(captured) == 2 and all(c['steps'] == 20 for c in captured)
        assert torch.equal(captured[0]['noise'], z)
        assert torch.equal(captured[1]['noise'], gather_candidates(z, pi.cpu()))
        assert torch.equal(gp[..., -1, :], g) and torch.equal(hp[..., -1, :], gather_candidates(g, pi))
        check_permutation(pi, x.valid)
        rows.append(dict(index=index, N=int(x.valid.sum()), checks=checks, actual_sampler_calls=2, DDIM_steps=20,
            G_sha256=tensor_sha(gp), H_sha256=tensor_sha(hp), final_pi_sha256=tensor_sha(pi),
            actual_G_noise_sha256=tensor_sha(captured[0]['noise']), actual_H_noise_sha256=tensor_sha(captured[1]['noise']),
            actual_keyed_noise_exact=True, exact_frozen_endpoint=True))
    assert model_hash(model) == initial_hash
    write(folder / 'DDIM_IDENTITY.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    a = parser.parse_args()
    guard = worker_guard()
    setup_cuda()
    cfg = frozen()
    arm, resume, until, budget = PHASES[a.phase]
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / (a.phase + '_RECEIPT.json')).exists(), 'phase already completed; no extra updates allowed'
    with (OUT / 'synthetic_worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed_all(3201)
        model, origin = build(cfg['config'], variant='energy_only')
        assert model_hash(model) == cfg['initial_model_sha256']
        model.cuda()
        assert model.coupling.csee is None and model.coupling.coordinator.rejection_gradient == 'proposal_st'
        data = SyntheticData()
        bank = SyntheticBank(data)
        identity = dict(experiment='SSEC_INDEPENDENT_SYNTHETIC_NONZERO_RECOVERY', stage='joint',
                        fold='SYNTHETIC_ONLY', source=cfg['implementation_hash'], official_inputs=False)
        initial_last = sha(folder / 'last.pt') if resume else None
        start = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)['update'] if resume else 0
        assert start == (2 if resume else 0)
        if resume:
            # Deliberately disturb all global RNGs before native restoration;
            # no other seed or alternative initialization experiment is used.
            random.random(); np.random.rand(7); torch.rand(7); torch.rand(7, device='cuda')
            with torch.no_grad():
                next(model.parameters()).add_(1.)
            assert model_hash(model) != cfg['initial_model_sha256']
        recorder = Recorder(model, folder, a.phase, start, budget)
        backward = recorder.callback(data, bank, cfg['config'])

        def no_validation():
            raise AssertionError('four updates cannot complete this 168-update epoch')

        started = time.perf_counter()
        interrupted = False
        try:
            with recorder.hooks():
                # Wall-clock progress is kept outside the comparison state.
                # False omits elapsed time/progress I/O and the extra first-step
                # progress checkpoint; native update/order/numeric statistics
                # and the actual persist/resume functions are intact.
                result = native.run_stage(folder, identity, cfg['recipe'], model, data.manifest['train_ids'],
                    backward, no_validation, until, resume=resume, record_progress=False)
        except PlannedInterruption:
            assert a.phase == 'replay_crash' and recorder.done == 1
            assert sha(folder / 'last.pt') == initial_last
            interrupted = True
        assert recorder.done == budget and guard['denied_cross_workspace_accesses'] == 0
        if not interrupted:
            assert result['update'] == until and result['cursor'] == until * 16
            assert result['completed_epoch'] == 0 and result['max_updates'] == 16800
            if until == 4:
                check_ddim(model, data, bank, folder)
        write(folder / (a.phase + '_RECEIPT.json'), dict(phase=a.phase, passed=True,
            actual_synthetic_optimizer_updates=recorder.done, real_optimizer_updates=0,
            start_update=start, stop_update=recorder.observer.update, planned_interruption=interrupted,
            checkpoint_unchanged_on_interruption=interrupted, native_controller=True, record_progress=False,
            native_state_has_no_wall_clock_fields=True, frozen_config_sha256=sha(FROZEN),
            initial_model_sha256=cfg['initial_model_sha256'], final_model_sha256=model_hash(model),
            parameters=sum(p.numel() for p in model.parameters()), buffers=list(dict(model.named_buffers())),
            source_sha256=cfg['implementation_hash'], duration_seconds=time.perf_counter()-started,
            data_accesses=data.accesses, input_guard=guard, rows=recorder.rows,
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         deterministic=torch.are_deterministic_algorithms_enabled(), TF32=torch.backends.cuda.matmul.allow_tf32)))
        print(a.phase, 'COMPLETE', recorder.done, 'synthetic updates; real updates 0', flush=True)


if __name__ == '__main__':
    main()
