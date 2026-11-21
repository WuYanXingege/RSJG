"""Frozen native replay only. No optimizer, backward, selection, or training entry."""
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write, seed_all
import bcep_joint.inference as inference
from bcep_v3.formal_inputs import FormalInputs, paths
from bcep_v3.formal import model_hash, source_manifest
from bcep_v3.model import build
from bcep_v2.runtime import configure_cuda
from ebjd.unified import gather_candidates
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from ebjd.sampling import differentiable_sample

BASE = '0243b781fd7d983b6d3f6ab9b04e4e29158f57d7'
FROZEN_SHA = '294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43'
CHECKPOINT_SHA = 'ae2ff4c81d0df7e1e42515a598208d2a2a7d6042513052bcb1fabd13bcf2be76'
OUT = SOURCE.parent/'RSJG_EBJD_assets/experiments/zara2_v3_frozen_feedback_audit_20261121'
GUARD = dict(optimizer_constructions=0, optimizer_step_calls=0, backward_calls=0)


def install_no_training_guard():
    def deny_constructor(*a, **kw):
        GUARD['optimizer_constructions'] += 1
        raise RuntimeError('AUDIT FORBIDS optimizer construction')
    def deny_step(*a, **kw):
        GUARD['optimizer_step_calls'] += 1
        raise RuntimeError('AUDIT FORBIDS every optimizer.step')
    def deny_backward(*a, **kw):
        GUARD['backward_calls'] += 1
        raise RuntimeError('AUDIT FORBIDS backward')
    torch.optim.Optimizer.__init__ = deny_constructor
    for cls in vars(torch.optim).values():
        if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer):
            cls.step = deny_step
    torch.Tensor.backward = deny_backward
    torch.autograd.backward = deny_backward


def protected_manifest():
    result = {}
    for label, root in paths().items():
        if label not in ('formal', 'candidate_cost', 'sharedworld', 'run'):
            continue
        for p in sorted(root.rglob('*')):
            if p.is_file():
                result[label+'/'+str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def numpy(t):
    return t.detach().cpu().contiguous().numpy()


def endpoint_error(goals, truth, valid):
    error = (goals-truth[:, None, :, -1]).norm(dim=-1)
    return ((error*valid[:, None]).sum(-1)/valid.sum(1)[:, None]).min(1).values


@torch.no_grad()
def main():
    install_no_training_guard()
    if OUT.exists():
        raise RuntimeError('Audit output already exists; never overwrite an audit receipt.')
    OUT.mkdir(parents=True)
    (OUT/'windows').mkdir()
    started = time.time()
    write(OUT/'STATUS.json', dict(status='VERIFYING', optimizer_updates=0))
    frozen = read(paths()['run']/'FROZEN_CONFIG.json')
    assert canonical_hash(frozen) == FROZEN_SHA
    assert source_manifest() == frozen['implementation_files']
    for name in frozen['implementation_files']:
        assert subprocess.check_output(['git','show',BASE+':'+name],cwd=SOURCE) == (SOURCE/name).read_bytes()
    checkpoint = paths()['run']/'joint/epoch_012.pt'
    assert digest(checkpoint) == CHECKPOINT_SHA
    assert read(paths()['run']/'STATUS.json')['status'] == 'COMPLETE'
    before = protected_manifest()
    write(OUT/'PROTECTED_BEFORE.json', before)
    inputs = FormalInputs()
    assert inputs.config == frozen['config'] and inputs.recipe == frozen['recipe']
    assert canonical_hash(inputs.split) == frozen['split_sha256']
    assert {b.purpose:b.hash for b in inputs.banks} == frozen['bank_sha256']
    torch.set_num_threads(1); configure_cuda(); seed_all(3201)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['completed_epoch'] == 12 and state['update'] == 2016
    assert state['identity'] == frozen['identity']
    model, _ = build(frozen['config'])  # Exactly one construction, then strict trained-state load.
    model.load_state_dict(state['model'], strict=True)
    # Preserve native requires_grad flags: changing them changes forward dispatch
    # and caused a reproducible 1-ULP JADE shift even inside no_grad (probe archived).
    # Freeze values with no_grad + forbidden optimizer/backward + full state hashes.
    model.eval(); model.cuda()
    trained_hash = model_hash(model)
    del state
    native = model.coupling.assign
    old_features = model.coupling.forward_features
    old_messages = model.coupling.messages
    old_encode = model.encode_context
    old_noise = inference.keyed_noise
    captured = {}
    def features(*a, **kw):
        value = old_features(*a, **kw); captured['A_scores'] = value[0]; return value
    def messages(*a, **kw):
        value = old_messages(*a, **kw); captured['old_feedback'] = value; return value
    def encode(*a, **kw):
        value = old_encode(*a, **kw); captured['context'] = value; return value
    def noise(*a, **kw):
        value = old_noise(*a, **kw); captured['noise'] = value; return value
    def assign(*a, **kw):
        value = native(*a, **kw, return_details=True)
        captured.update(C_pi=value[1], C_scores=value[2], A_native_pi=value[3]['initial_pi'],
                        B_scores=value[3]['old_final'], HCRA_delta=value[3]['score_delta'])
        return value[:3]
    model.coupling.forward_features = features
    model.coupling.messages = messages
    model.coupling.assign = assign
    model.encode_context = encode
    inference.keyed_noise = noise
    with (paths()['run']/'dev/epoch_012_windows.csv').open() as f:
        archived = {int(r['index']):r for r in csv.DictReader(f)}
    ids = inputs.split['selection_dev_indices']
    assert list(archived) == ids and len(ids) == 590
    rows, files = [], []
    acc_on, acc_off = MetricAccumulator(), MetricAccumulator()
    def base_only_gate(context, trajectory, goals, time_value, base_gate, scale):
        b,p,n = goals.shape[:3]
        gate = base_gate.reshape(b,1,1).expand(b,p,n)*context.valid[:,None]
        pair = (gate[..., :,None]+gate[...,None,:])*.5
        pair = pair*(context.valid[:,None,:,None]&context.valid[:,None,None,:])
        return gate,pair
    original_gate = model.bridge.denoiser.grg_ctf.forward
    for pos,index in enumerate(ids):
        captured.clear()
        batch = inputs.data.batch([index], 'selection_dev')
        goals = inputs.banks[1].batch([index], batch).cuda()
        x = batch.to('cuda')
        _, path, pi, info = inference.generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,goals,2035,mode='H')
        assert torch.equal(captured['B_scores'],captured['A_scores']+captured['old_feedback'])
        assert torch.equal(captured['C_scores'],captured['B_scores']+captured['HCRA_delta'])
        hgoals = gather_candidates(goals, pi)
        jfde = float(endpoint_error(hgoals,x.future,x.valid))
        stat = hard_metric_statistics(path,x.future,x.valid)
        jade = float(stat.scene_jade)
        reference = archived[index]
        # EXACT per-window comparison, not merely matching a six-decimal aggregate.
        if jfde != float(reference['H_JFDE']) or jade != float(reference['H_JADE']):
            write(OUT/'STATUS.json',dict(status='REPLAY_MISMATCH',index=index,JFDE=jfde,archived_JFDE=float(reference['H_JFDE']),
                JADE=jade,archived_JADE=float(reference['H_JADE']),optimizer_updates=0))
            raise RuntimeError('Native replay differs from archived per-window metric; attribution forbidden.')
        acc_on.update_statistics(stat)
        zh = gather_candidates(captured['noise'].to('cuda'), pi)
        # Only residual gating is disabled. Same trained model, context, pi, endpoint and keyed noise.
        model.bridge.denoiser.grg_ctf.forward = base_only_gate
        try:
            off, _, _ = differentiable_sample(model.bridge,captured['context'],zh,20,proposal_goals=hgoals,proposal_eta=0.)
        finally:
            model.bridge.denoiser.grg_ctf.forward = original_gate
        offstat = hard_metric_statistics(off,x.future,x.valid); acc_off.update_statistics(offstat)
        assert torch.equal(path[..., -1,:],off[..., -1,:])
        torch.testing.assert_close(off[..., -1,:],hgoals,atol=1e-6,rtol=1e-6)
        row = dict(index=index,n_agents=int(x.valid.sum()),C_JFDE=jfde,C_path_JFDE=float(stat.scene_jfde),
            GRG_on_H_JADE=jade,GRG_off_H_JADE=float(offstat.scene_jade),
            off_minus_on_H_JADE=float(offstat.scene_jade)-jade,archive_JFDE_exact=True,archive_JADE_exact=True,
            candidate_sha256=inputs.banks[1].native.rows[index]['sha256'],
            native_pi_sha256=hashlib.sha256(numpy(pi).tobytes()).hexdigest(),
            keyed_noise_sha256=hashlib.sha256(numpy(captured['noise']).tobytes()).hexdigest(),
            minFDE=float(stat.agent_minfde[x.valid].double().mean()),oracle=float(stat.scene_marginal_fde))
        rows.append(row)
        arrays = {k:numpy(captured[k]) for k in ['A_scores','old_feedback','B_scores','HCRA_delta','C_scores','A_native_pi','C_pi']}
        arrays.update(goals=numpy(goals),observed=numpy(x.observed),truth_endpoint=numpy(x.future[:,:,-1]),
                      valid=numpy(x.valid),baseline=numpy(captured['context'].baseline),
                      native_H_JADE=np.array(jade),native_H_JFDE=np.array(jfde))
        dest = OUT/'windows'/f'{index}.npz'
        tmp = dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.replace(dest)
        files.append(dict(index=index,file=str(dest.relative_to(OUT)),sha256=digest(dest)))
        if (pos+1)%25 == 0 or pos==0:
            progress = dict(status='NATIVE_REPLAY_AND_GRG_FROZEN_CHECK',windows=pos+1,total=590,
                            elapsed_seconds=time.time()-started,optimizer_updates=0)
            write(OUT/'STATUS.json',progress); print(json.dumps(progress),flush=True)
    assert model_hash(model) == trained_hash and digest(checkpoint) == CHECKPOINT_SHA
    assert GUARD == dict(optimizer_constructions=0,optimizer_step_calls=0,backward_calls=0)
    after = protected_manifest(); assert after == before
    proof = dict(base_commit=BASE,frozen_sha256=FROZEN_SHA,checkpoint_sha256=CHECKPOINT_SHA,epoch=12,
        trained_model_sha256_before=trained_hash,trained_model_sha256_after=model_hash(model),
        model_constructions=1,strict_trained_state_load=True,native_requires_grad_flags_preserved=True,
        autograd_enabled=False,parameter_gradients_present=any(p.grad is not None for p in model.parameters()),
        windows=590,multi_windows=sum(r['n_agents']>1 for r in rows),dense_windows=sum(r['n_agents']>=9 for r in rows),
        agents=sum(r['n_agents'] for r in rows),candidate_seed=2035,K=20,DDIM=20,
        bank_sha256=frozen['bank_sha256'],split_sha256=frozen['split_sha256'],
        archive_per_window_JFDE_exact=True,archive_per_window_JADE_exact=True,
        historical_permutation_archive_exists=False,
        approved_reference='User accepted fixed-source native replay as permutation reference; archive stores only metrics.',
        native_replay_permutation_manifest_sha256=canonical_hash(files),
        native_metrics=acc_on.as_dict(),GRG_off_metrics=acc_off.as_dict(),
        protected_files=len(before),protected_bytes=sum(v['bytes'] for v in before.values()),
        protection_manifest_sha256=canonical_hash(before),protected_all_unchanged=True,
        guard=GUARD,real_optimizer_updates=0,synthetic_optimizer_updates=0,
        elapsed_seconds=time.time()-started,torch_version=torch.__version__,numpy_version=np.__version__,
        device=torch.cuda.get_device_name(),cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated())
    write_csv(OUT/'REPLAY_WINDOWS.csv',rows); write(OUT/'WINDOW_FILES.json',files)
    write(OUT/'REPLAY_PROOF.json',proof)
    write(OUT/'STATUS.json',dict(status='NATIVE_REPLAY_VERIFIED',windows=590,optimizer_updates=0))
    print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
