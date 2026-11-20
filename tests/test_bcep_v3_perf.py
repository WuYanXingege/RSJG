"""CPU synthetic exactness. Never opens official trajectories or banks."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bcep_joint.checks import equal
from bcep_joint.common import seed_all
from bcep_joint.inference import keyed_noise
from bcep_v2.state import run_stage, optimizer
from bcep_v2.stages import joint_callbacks as native_callbacks
from bcep_v3.fixtures import SyntheticData, SyntheticBank, joint_config
from bcep_v3.formal_telemetry import ModuleTelemetry
from bcep_v3.model import build, NEW_PREFIXES
from bcep_v3.perf.telemetry import BatchedModuleTelemetry
from bcep_v3.perf.stages import ScalarJournal, compact_checked, joint_callbacks
from bcep_v3.perf.options import checkpoint_arm

OUT = Path(os.environ.get('BCEP_PERF_OUTPUT', '/tmp/bcep_v3_perf_cpu'))
RESULTS = {}


@pytest.fixture(scope='session', autouse=True)
def cpu_synthetic_boundary():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit must hide all CUDA devices'
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    steps = []
    original = torch.optim.AdamW.step
    def count_step(opt,*args,**kwargs):
        assert all(p.device.type=='cpu' for group in opt.param_groups for p in group['params'])
        result = original(opt,*args,**kwargs)
        steps.append(len(steps)+1)
        return result
    torch.optim.AdamW.step = count_step
    from bcep_formal.data import FormalData
    from bcep_v2.bank import Bank
    prior_item,prior_batch = FormalData.item,Bank.batch
    def deny(*a,**kw):raise AssertionError('official data/bank access forbidden during synthetic tests')
    FormalData.item = deny; Bank.batch = deny
    try:
        yield
    finally:
        torch.optim.AdamW.step = original; FormalData.item = prior_item; Bank.batch = prior_batch
        RESULTS.update(synthetic_cpu_optimizer_updates=len(steps), real_optimizer_updates=0,
                       cuda_initialized=torch.cuda.is_initialized(), cuda_rng_exactness='UNTESTED_GPU_PROFILE_PENDING',
                       formal_data_and_bank_calls_denied=True)
        (OUT/'EXACTNESS.json').write_text(json.dumps(RESULTS,indent=2)+'\n')


def compare_logs(a,b):
    assert len(a)==len(b)
    maximum=0.
    for x,y in zip(a,b):
        assert x.keys()==y.keys()
        for k in ['update','domain']:assert x[k]==y[k]
        for prefix in NEW_PREFIXES:
            for key,v in x['modules'][prefix].items():
                w=y['modules'][prefix][key]
                if key=='present_tensors':assert v==w
                else:
                    assert (v>0)==(w>0)
                    assert v==pytest.approx(w,rel=1e-12,abs=1e-15)
                    maximum=max(maximum,abs(v-w))
    return maximum


def snapshot(model,opt,order,view):
    return copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict(),
        grads={n:p.grad for n,p in model.named_parameters()}, cpu_rng=torch.get_rng_state(),
        numpy=np.random.get_state(), python=__import__('random').getstate(),
        order_rng=order.get_state(),view_rng=view.get_state()))


def one_step(counts,arm,layout=None):
    seed_all(3201);config=joint_config();data=SyntheticData(counts,pixels=32);bank=SyntheticBank()
    if layout=='colocated':
        for item in data.items:item['observed'].zero_()
    elif layout=='people_permuted':
        for item in data.items:
            for key in ['observed','future','semantic_maps']:item[key]=item[key].flip(0)
            item['agent_ids']=list(reversed(item['agent_ids']))
    elif layout=='candidate_permuted':
        class PermutedBank(SyntheticBank):
            def batch(self,ids,batch):
                g=super().batch(ids,batch);generator=torch.Generator().manual_seed(614)
                pi=torch.stack([torch.stack([torch.randperm(20,generator=generator) for _ in range(g.shape[2])],1) for _ in ids])
                from ebjd.unified import gather_candidates
                return gather_candidates(g,pi)
        bank=PermutedBank()
    model,_=build(config)
    # Exercise nonzero new residuals; identical local synthetic weights in all arms.
    with torch.no_grad():
        model.coupling.csee.out.weight.fill_(.001)
        model.coupling.hcra.delta[-1].weight.fill_(.002)
        model.bridge.denoiser.grg_ctf.net[-1].weight.fill_(.001)
    observer=(ModuleTelemetry if arm=='baseline' else BatchedModuleTelemetry)(model,domain='synthetic')
    order=torch.Generator().manual_seed(3201);view=torch.Generator().manual_seed(3202)
    backward,_=joint_callbacks(model,data,'cpu',config,bank,bank,defer_diagnostics=arm=='p012',cpu_mask_check=arm=='p012')
    with observer.native_optimizer_hook():
        opt=optimizer(model,1e-4);model.train();opt.zero_grad(set_to_none=True)
        metrics=backward(list(range(len(counts))),order,view);observer.capture_backward()
        before=snapshot(model,opt,order,view)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        after=snapshot(model,opt,order,view)
    return before,after,metrics,observer.rows


@pytest.mark.parametrize('counts',[(1,),(2,),(9,),(57,),(1,2,9)])
def test_full_model_step_exact_across_shapes(counts):
    a=one_step(counts,'baseline');b=one_step(counts,'p012')
    for x,y in zip(a[:3],b[:3]):equal(x,y)
    error=compare_logs(a[3],b[3])
    RESULTS.setdefault('full_step_shapes',[]).append(dict(counts=counts,model_grads_optimizer_rng_bitwise=True,statistics_max_abs=error))


@pytest.mark.parametrize('layout',['colocated','candidate_permuted','people_permuted'])
def test_nontrivial_layouts_keep_training_state_exact(layout):
    a=one_step((2,9),'baseline',layout);b=one_step((2,9),'p012',layout)
    for x,y in zip(a[:3],b[:3]):equal(x,y)
    error=compare_logs(a[3],b[3])
    RESULTS.setdefault('nontrivial_layouts',[]).append(dict(layout=layout,old_new_same_input_state_bitwise=True,statistics_max_abs=error,
        note='Old/new on identical reordered input; does not claim a unique permutation for tied Hungarian optima.'))


def test_full16_tail_native_recovery_and_separate_p0(tmp_path):
    import importlib.util
    from unittest.mock import patch
    import bcep_v2.state as native
    spec=importlib.util.spec_from_file_location('audit_benchmark',Path(__file__).resolve().parents[1]/'scripts/perf_audit/benchmark.py')
    hashing=importlib.util.module_from_spec(spec);spec.loader.exec_module(hashing)
    counts=(1,2,9,57,3,4,8,1,2,9,4,3,1,2,3,4,1,9,2)
    config=joint_config();data=SyntheticData(counts,pixels=32);bank=SyntheticBank()
    recipe=dict(seed=3201,lr=1e-4,weight_decay=1e-4,clip=5.,logical_scenes=16,max_epochs=2,minimum=2,patience=15,scheduler_horizon=100)
    identity=dict(stage='joint',fold='TEST_ONLY',experiment='isolated_cpu_perf')
    def run(folder,arm,limit,resume=False):
        seed_all(3201);model,_=build(config)
        old=torch.load(folder/'last.pt',map_location='cpu',weights_only=False) if resume else None
        observer=(ModuleTelemetry if arm=='baseline' else BatchedModuleTelemetry)(model,folder/'updates.jsonl',old['update'] if old else 0,'synthetic')
        backward,_=joint_callbacks(model,data,'cpu',config,bank,bank,defer_diagnostics=arm=='p012',cpu_mask_check=arm=='p012')
        current={}
        def wrapped(*args):
            current.update(ids=list(args[0]),order=args[1],view=args[2])
            result=backward(*args);observer.capture_backward();return result
        def validate():
            # Selection/retention interface probe only, not an accuracy evaluation.
            return 1.
        with observer.native_optimizer_hook():
            factory=native.optimizer
            def instrument(*args,**kwargs):
                opt=factory(*args,**kwargs)
                def receipt(phase):
                    def hook(opt,*ignored):
                        payload=snapshot(model,opt,current['order'],current['view']);payload['group_ids']=current['ids']
                        record=dict(phase=phase,update=observer.update+(phase=='before'),sha256=hashing.hash_state(payload))
                        with (folder/'step_states.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    return hook
                opt.register_step_pre_hook(receipt('before'));opt.register_step_post_hook(receipt('after'))
                return opt
            with patch.object(native,'optimizer',instrument):
                state=run_stage(folder,identity,recipe,model,list(range(len(counts))),wrapped,validate,limit,resume=resume)
        return copy.deepcopy(state)
    a=run(tmp_path/'base','baseline',4);p0=run(tmp_path/'p0','p0',4);b=run(tmp_path/'p012','p012',4)
    equal(a,p0);equal(a,b)
    partial=run(tmp_path/'resume','p012',1)
    assert partial['completed_epoch']==0 and partial['cursor']==16 and partial['order'] is not None
    recovered=run(tmp_path/'resume','p012',4,True);equal(b,recovered)
    log=lambda folder:[json.loads(x) for x in (tmp_path/folder/'updates.jsonl').read_text().splitlines()]
    err=max(compare_logs(log('base'),log(k)) for k in ['p0','p012','resume'])
    assert [r['update'] for r in log('resume')]==[1,2,3,4]
    assert [r['epoch'] for r in json.loads((tmp_path/'p012/TOP5.json').read_text())]==[1,2]
    steps=lambda name:[json.loads(x) for x in (tmp_path/name/'step_states.jsonl').read_text().splitlines()]
    for arm in ['p0','p012','resume']:equal(steps('base'),steps(arm))
    RESULTS['native_full16_tail_recovery']=dict(completed_epochs=2,updates_per_arm=4,
        partial_cursor=16,tail_windows=3,whole_state_bitwise=True,statistics_max_abs=err,
        each_update_pre_post_state_sha256_equal=True,pre_post_receipts=steps('base'),
        compared_fields=['model','optimizer','scheduler','rng','history','cursor','order','stale','trainable','update'])


class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__();self.coupling=torch.nn.Module();self.bridge=torch.nn.Module();self.bridge.denoiser=torch.nn.Module()
        self.coupling.csee=torch.nn.Linear(2,2);self.coupling.hcra=torch.nn.Linear(2,2);self.bridge.denoiser.grg_ctf=torch.nn.Linear(2,2)


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_gradient_fail_before_step_and_no_missing_grad_problem(bad):
    m=Mini();t=BatchedModuleTelemetry(m)
    assert all(r['gradient_l2']==0 for r in t.capture_backward().values())
    before=copy.deepcopy(m.state_dict());next(m.parameters()).grad=torch.full_like(next(m.parameters()),bad)
    with pytest.raises(FloatingPointError):t.capture_backward()
    equal(before,m.state_dict());assert t.update==0


def test_bounded_rows_parameter_identity_transfer_count_and_partial_tail(tmp_path):
    m=Mini();t=BatchedModuleTelemetry(m,tmp_path/'log.jsonl',max_history_rows=3)
    assert all(t.named[n] is p for n,p in m.named_parameters())
    for _ in range(70):
        for p in m.parameters():p.grad=torch.ones_like(p)
        t.capture_backward()
        # Not optimizer steps: deterministic synthetic parameter perturbation for logger stress.
        with torch.no_grad():
            for p in m.parameters():p.add_(.001)
        t.step_done()
        assert not t.before and len(t.rows)<=3
    assert t.host_transfers==140 and t.rows[-1]['update']==70
    lines=(tmp_path/'log.jsonl').read_text().splitlines();assert len(lines)==70
    (tmp_path/'partial.jsonl').write_text('{"update":')
    with pytest.raises(ValueError,match='incomplete'):BatchedModuleTelemetry(m,tmp_path/'partial.jsonl')


def test_replayed_updates_keep_journal_and_follow_checkpoint_counter(tmp_path):
    m=Mini();path=tmp_path/'replay.jsonl';observer=BatchedModuleTelemetry(m,path,domain='synthetic')
    saved=None
    def event(o):
        for p in m.parameters():p.grad=torch.ones_like(p)
        o.capture_backward()
        with torch.no_grad():
            for p in m.parameters():p.add_(.001)
        o.step_done()
    for index in range(3):
        event(observer)
        if index==0:saved=copy.deepcopy(m.state_dict())
    m.load_state_dict(saved);resumed=BatchedModuleTelemetry(m,path,start_update=1,domain='synthetic');event(resumed)
    records=[json.loads(x) for x in path.read_text().splitlines()]
    assert [r['update'] for r in records]==[1,2,3,2]
    compare_logs([records[1]],[records[3]])
    RESULTS['journal_replay']=dict(physical_events_retained=4,effective_last_update=2,
        repeated_update_uses_checkpoint_counter=True,statistics_equal=True,manual_parameter_perturbations_not_optimizer_steps=4)


def test_deferred_fields_float_order_finite_and_no_graph_retention():
    a,b=ScalarJournal(),ScalarJournal(True)
    for i in range(19):
        x=torch.tensor((i+1)*.0123,requires_grad=True)
        for journal in [a,b]:
            journal.add('weighted',x,3/7,4);journal.add('count',3,count=True)
    assert all(not value.requires_grad for _,value,_,_,_ in b.records if torch.is_tensor(value))
    equal(a.finish(),b.finish());assert b.host_transfers==1 and b.records==[]
    bad=ScalarJournal(True);bad.add('bad',torch.tensor(float('nan')))
    with pytest.raises(FloatingPointError):bad.finish()


def test_cpu_mask_check_rejects_noncanonical_before_transfer(monkeypatch):
    x=SyntheticData((2,3),pixels=32).batch([0,1],'train');x.valid[0]=torch.tensor([True,False,True])
    for arm in [False,True]:
        with pytest.raises(ValueError,match='noncanonical'):compact_checked(x,0,2,'cpu',arm)
    def denied(*a,**kw):pytest.fail('invalid CPU identity reached device transfer')
    monkeypatch.setattr(type(x),'to',denied)
    with pytest.raises(ValueError):compact_checked(x,0,2,'cuda',True)


def test_checkpoint_switches_restore_and_cpu_gradients(tmp_path):
    seed_all(3201);config=joint_config();data=SyntheticData((2,),pixels=32);bank=SyntheticBank();model,_=build(config)
    initial=copy.deepcopy(model.state_dict());records=[]
    for map_on,bridge_on in [(True,True),(False,True),(True,False),(False,False)]:
        model.load_state_dict(initial);model.train();model.zero_grad(set_to_none=True);seed_all(77)
        order=torch.Generator().manual_seed(5);view=torch.Generator().manual_seed(6)
        with checkpoint_arm(model,map_checkpoint=map_on,bridge_checkpoint=bridge_on):
            backward,_=native_callbacks(model,data,'cpu',config,bank,bank);metrics=backward([0],order,view)
        assert model.bridge.encoder.activation_checkpointing and model.bridge.denoiser.activation_checkpointing
        records.append(dict(metrics=metrics,grads={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()},rng=torch.get_rng_state()))
    for r in records[1:]:equal(records[0],r)
    RESULTS['checkpoint_arms']=dict(cpu_two_agent_output_grad_rng_bitwise=True,gpu_and_57_agent_full_resolution='UNTESTED',default_enabled_unchanged=True)


def test_keyed_noise_candidates_and_people_permutations_byte_exact():
    valid=torch.tensor([[True,True,False],[True,True,True]])
    scenes=['SYNTHETIC_A','SYNTHETIC_B'];agents=[['a','b'],['c','d','e']]
    rng=torch.get_rng_state();a=keyed_noise(scenes,agents,valid,2035);assert torch.equal(rng,torch.get_rng_state())
    permutation=torch.stack([torch.stack([torch.randperm(20) for _ in range(3)],1) for _ in range(2)])
    from ebjd.unified import gather_candidates
    b=gather_candidates(a,permutation);assert torch.equal(gather_candidates(b,permutation.argsort(1)),a)
    swapped=keyed_noise(scenes,[list(reversed(r)) for r in agents],valid,2035)
    assert torch.equal(swapped[0,:,:2],a[0,:,:2].flip(1)) and torch.equal(swapped[1],a[1].flip(1))
