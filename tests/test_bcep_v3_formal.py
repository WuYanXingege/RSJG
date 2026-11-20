import copy,json
import pytest
import torch
from bcep_v3.formal import require_authorization
from bcep_v3.formal_inputs import TrainDevData
from bcep_v3.formal_telemetry import no_optimizer_steps,ModuleTelemetry
from bcep_v2.state import run_stage


def test_training_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv('RUN_ONE_ZARA2_V3',raising=False)
    with pytest.raises(PermissionError):require_authorization('train','x')
    require_authorization('prepare',None)
    monkeypatch.setenv('RUN_ONE_ZARA2_V3','true')
    with pytest.raises(PermissionError):require_authorization('train',None)
    require_authorization('train','frozen-digest')


def test_real_readonly_step_guard():
    m=torch.nn.Linear(1,1);opt=torch.optim.AdamW(m.parameters());before=copy.deepcopy(m.state_dict())
    m(torch.ones(1,1)).sum().backward()
    with no_optimizer_steps(),pytest.raises(PermissionError):opt.step()
    assert all(torch.equal(x,m.state_dict()[n]) for n,x in before.items())


def test_heldout_rejected_before_file_read():
    data=object.__new__(TrainDevData)
    with pytest.raises(ValueError,match='only train'):data.item(0,'heldout_confirmation')


def test_native_min30_patience15_topfive_ties_and_full_state(tmp_path):
    m=torch.nn.Linear(1,1)
    recipe=dict(seed=3201,lr=1e-4,weight_decay=1e-4,clip=5.,logical_scenes=1,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
    def backward(*args):
        loss=m(torch.ones(1,1)).square().mean();loss.backward();return {'loss':float(loss.detach())}
    s=run_stage(tmp_path,dict(stage='joint',fold='TEST_ONLY'),recipe,m,[0],backward,lambda:1.,100,resume=False)
    assert s['completed_epoch']==30 and s['stop_reason']=='early_stop' and s['update']==30
    top=json.loads((tmp_path/'TOP5.json').read_text());assert [x['epoch'] for x in top]==[1,2,3,4,5]
    assert len(list(tmp_path.glob('epoch_*.pt')))==30
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert set(['model','optimizer','scheduler','rng','cursor','order','stale','history','completed_epoch','update'])<=set(saved)
    assert saved['cursor']==0 and saved['order'] is None
    assert set(saved['rng'])=={'cpu','cuda','python','numpy','order','view'}
    with pytest.raises(ValueError,match='recipe'):
        run_stage(tmp_path,saved['identity'],dict(recipe,seed=5),m,[0],backward,lambda:1.,100,resume=True)


def test_cli_train_denied_before_loading_inputs(monkeypatch):
    import sys
    from bcep_v3 import formal
    monkeypatch.delenv('RUN_ONE_ZARA2_V3',raising=False)
    monkeypatch.setattr(sys,'argv',['formal','train','--frozen-sha','invalid'])
    monkeypatch.setattr(formal,'FormalInputs',lambda:pytest.fail('data loaded before authorization'))
    with pytest.raises(PermissionError):formal.main()
