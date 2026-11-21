"""Formal-entry guards and unchanged writer/stop-rule verification; ZERO steps."""
import ast,copy,hashlib,inspect,json
from pathlib import Path
import pytest,torch
from bcep_scene_energy import formal
from bcep_scene_energy.formal_contract import require_authorization,BASE,EXPERIMENT,READY
from bcep_v2 import state as native


def roots(tmp):return {'run':tmp/'run','audit':tmp/'audit'}

def test_unauthorized_before_data_model_optimizer_or_output(monkeypatch,tmp_path):
    rr=roots(tmp_path);monkeypatch.setattr(formal,'paths',lambda:rr);monkeypatch.delenv('RUN_ONE_ZARA2_SSEC',raising=False)
    from bcep_v3 import formal_inputs
    monkeypatch.setattr(formal_inputs,'FormalInputs',lambda:pytest.fail('unauthorized data construction'))
    monkeypatch.setattr(formal,'freeze',lambda *a:pytest.fail('unauthorized model construction'))
    monkeypatch.setattr(native,'optimizer',lambda *a:pytest.fail('unauthorized optimizer'))
    with pytest.raises(PermissionError):formal.main(['train','--frozen-sha','0'*64])
    assert not rr['run'].exists()


def test_wrong_sha_before_constructors_even_with_flag(monkeypatch,tmp_path):
    rr=roots(tmp_path);rr['run'].mkdir();(rr['run']/'FROZEN_CONFIG.json').write_text('{}')
    monkeypatch.setattr(formal,'paths',lambda:rr);monkeypatch.setenv('RUN_ONE_ZARA2_SSEC','true')
    monkeypatch.setattr(formal,'freeze',lambda *a:pytest.fail('model constructed with bad SHA'))
    from bcep_v3 import formal_inputs
    monkeypatch.setattr(formal_inputs,'FormalInputs',lambda:pytest.fail('data constructed with bad SHA'))
    with pytest.raises(PermissionError,match='SHA256'):formal.main(['train','--frozen-sha','0'*64])


def test_exact_sha_still_requires_matching_readiness_and_source(monkeypatch,tmp_path):
    import bcep_scene_energy.formal_contract as contract
    rr=roots(tmp_path);rr['run'].mkdir();rr['audit'].mkdir()
    f=rr['run']/'FROZEN_CONFIG.json';cfg=dict(base_source=BASE,experiment=EXPERIMENT,variant='energy_only',implementation_files={'x':'y'})
    f.write_text(json.dumps(cfg));sha=hashlib.sha256(f.read_bytes()).hexdigest()
    monkeypatch.setenv('RUN_ONE_ZARA2_SSEC','true');monkeypatch.setattr(contract,'source_manifest',lambda:{'x':'y'})
    with pytest.raises(PermissionError,match='readiness'):require_authorization('train',sha,rr)
    r=rr['audit']/'READINESS.json';r.write_text(json.dumps(dict(status='BLOCKED',frozen_file_sha256=sha,real_optimizer_updates=0,synthetic_optimizer_updates=0)))
    with pytest.raises(PermissionError):require_authorization('train',sha,rr)
    r.write_text(json.dumps(dict(status=READY,frozen_file_sha256=sha,real_optimizer_updates=0,synthetic_optimizer_updates=0)))
    assert require_authorization('train',sha,rr)==cfg  # Gate only; NEVER call train.
    monkeypatch.setattr(contract,'source_manifest',lambda:{'x':'changed'})
    with pytest.raises(PermissionError,match='source'):require_authorization('train',sha,rr)


def test_only_single_fixed_entry_options():
    with pytest.raises(SystemExit):formal.main(['train','--seed','3202'])
    from bcep_v3.formal_inputs import TrainDevData
    data=object.__new__(TrainDevData)
    with pytest.raises(ValueError):data.item(0,'heldout_confirmation')


def native_tree():return ast.parse(inspect.getsource(native.run_stage)).body[0]

def test_actual_atomic_checkpoint_writer_top5_and_complete_fields(tmp_path):
    # Execute the ORIGINAL nested persist body, without entering the training
    # loop. Epoch labels are explicit WRITER FIXTURES, never trained epochs.
    tree=native_tree();node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='persist')
    model=torch.nn.Linear(2,2);opt=native.optimizer(model,1e-4)
    state=dict(identity=dict(stage='joint',fold='WRITER_FIXTURE_NO_TRAINING'),recipe={},trainable=list(dict(model.named_parameters())),
        epoch=1,completed_epoch=0,update=0,cursor=0,order=None,history=[],stale=0,fixture_not_a_trained_epoch=True)
    ns=dict(vars(native),state=state,model=model,opt=opt,plateau=None,device=torch.device('cpu'),until_update=0,
        folder=tmp_path,last=tmp_path/'last.pt',order_rng=torch.Generator().manual_seed(1),view_rng=torch.Generator().manual_seed(2))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'native.run_stage.persist','exec'),ns)
    for epoch,value in enumerate([.8,.6,.6,.7,.5,.9],1):
        state.update(epoch=epoch+1,completed_epoch=epoch)
        state['history'].append(dict(epoch=epoch,validation=value,file=f'epoch_{epoch:03d}.pt',update=0))
        ns['persist']('WRITER_FIXTURE',True)
    top=json.loads((tmp_path/'TOP5.json').read_text());assert [x['epoch'] for x in top]==[5,2,3,4,1]
    assert len(list(tmp_path.glob('epoch_*.pt')))==6
    from bcep.contracts import digest
    assert all(digest(tmp_path/r['file'])==r['sha256'] for r in top)
    assert (tmp_path/'best.pt').stat().st_ino==(tmp_path/'epoch_005.pt').stat().st_ino
    assert (tmp_path/'last.pt').stat().st_ino==(tmp_path/'epoch_006.pt').stat().st_ino
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert saved['update']==0 and saved['fixture_not_a_trained_epoch']
    assert {'model','optimizer','scheduler','rng','cursor','order','history','stale','completed_epoch','update'}<=saved.keys()
    assert set(saved['rng'])=={'cpu','cuda','python','numpy','order','view'}


def test_original_minimum_patience_cap_and_exact_tie_rule():
    tree=native_tree();loop=next(n for n in tree.body if isinstance(n,ast.While))
    names={'best','stale','done'}
    nodes=[n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    assert len(nodes)==3
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'native.stop.rules','exec')
    for epoch,best_epoch,expected in [(29,1,False),(30,1,True),(40,26,False),(41,26,True),(100,99,True)]:
        # Keep chronological history; exact ties must select the first epoch.
        history=[dict(epoch=i,validation=0. if i>=best_epoch else 1.) for i in range(1,epoch+1)]
        ns=dict(state={'history':history},epoch=epoch,recipe=dict(minimum=30,patience=15,max_epochs=100))
        exec(code,ns);assert ns['best']['epoch']==best_epoch and ns['done']==expected


def test_original_scheduler_without_optimizer_updates():
    from bcep_joint.model import set_lr
    m=torch.nn.Linear(1,1);opt=native.optimizer(m,1e-4)
    vals=[set_lr(opt,e,100)['factor'] for e in [1,5,6,100]]
    assert vals[0]==.2 and vals[1]==1. and vals[2]<1 and abs(vals[3]-.1)<1e-12
    assert not opt.state


def test_single_and_multi_observations_finite():
    from test_bcep_scene_energy import fixture,coupling
    from bcep_scene_energy.formal_monitor import decision_statistics
    import math
    for n in [1,2]:
        m=coupling();c,g=fixture(n)
        with torch.no_grad():*_,detail=m.assign(c,g,torch.tensor(1.),straight_through=False,return_details=True)
        row=decision_statistics(detail,False)[0]
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        if n==1:assert row['sinkhorn_row_entropy']==0 and row['attempted']==0


def test_native_post_hook_registered_without_invocation(tmp_path):
    from bcep_scene_energy.formal_monitor import StepObserver
    model=torch.nn.Module();model.coupling=torch.nn.Module();model.coupling.energy=torch.nn.Module()
    model.coupling.energy.phi=torch.nn.Linear(2,2);model.coupling.energy.rho=torch.nn.Linear(2,2)
    model.bridge=torch.nn.Module();model.bridge.encoder=torch.nn.Linear(2,2)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    sum(p.square().sum() for p in model.parameters()).backward()
    observer=StepObserver(model,tmp_path/'not_created.jsonl');observer.capture_backward({},.1,0)
    factory=native.optimizer
    with observer.native_optimizer_hook():
        opt=native.optimizer(model,1e-4)
        assert len(opt._optimizer_step_post_hooks)==1 and not opt.state
    assert native.optimizer is factory and observer.update==0 and not observer.path.exists()
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_epoch_monitor_replay_supersedes_old_attempt_without_any_steps(tmp_path):
    from bcep_scene_energy.formal_monitor import StepObserver,PREFIXES
    observer=StepObserver(torch.nn.Linear(1,1),tmp_path/'fixture_records.jsonl')
    with pytest.raises(RuntimeError,match='incomplete'):observer.epoch_summary(1)
    # Plain diagnostic JSON fixtures, not optimizer events or trained epochs.
    rows=[]
    for update in range(1,169):
        rows.append(dict(update=update,fixture=True,modules={p:dict(gradient_l2=1.,parameter_delta_l2=2.,parameter_delta_max=3.) for p in PREFIXES},
            physical_groups=[],logical_group_forward_backward_seconds=.1,allocated_peak_bytes=1))
    replay=copy.deepcopy(rows[0]);replay['modules'][PREFIXES[0]]['parameter_delta_l2']=170.;rows.append(replay)
    observer.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=observer.epoch_summary(1)
    assert result['optimizer_updates']==168 and observer.update==0
    assert result['modules_mean_per_update'][PREFIXES[0]]['parameter_delta_l2']==3.
