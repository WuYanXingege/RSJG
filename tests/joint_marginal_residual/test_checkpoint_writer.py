"""Exercise the actual resumable writer; no formal research scores are made."""
import copy
import random
import numpy as np
import pytest
import torch
from bcep.contracts import digest
from bcep_joint.common import read
from joint_marginal_residual.model import ResidualHead
from joint_marginal_residual.selection import evaluate_gate,rank_epoch
from joint_marginal_residual.train import persist,restore_rng


def training_objects():
    random.seed(3201);np.random.seed(3201);torch.manual_seed(3201)
    model=ResidualHead();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(3201)
    g=torch.Generator().manual_seed(2036)
    batch=(torch.randn(20,3,12,2,generator=g),torch.randn(20,3,154,generator=g),
           torch.randn(20,3,5,generator=g),torch.ones(3,dtype=torch.bool))
    state=dict(schema='residual-complete-resume-v1',recipe=dict(seed=3201),teacher_identity=dict(sha='fixed'),
        config_sha256='fixed-config',ablation=False,epoch=1,completed_epoch=0,update=0,cursor=0,order=None,
        history=[],epoch_train_seconds=0.,epoch_loss_sum=0.,epoch_updates=0,wall_seconds=0.)
    return model,opt,rng,batch,state


def step(model,opt,batch):
    opt.zero_grad(set_to_none=True)
    model(*batch).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()


def metric_groups(fde=.5,jfde=1.,jade=.4):
    values=dict(minFDE=fde,JFDE=jfde,minADE=.19,JADE=jade,scene_count=10,agent_count=30)
    return dict(all=values,N_ge_9=dict(values,scene_count=2,agent_count=20))


def record(epoch,fde,jfde,jade=.39):
    metrics=metric_groups(fde,jfde,jade)
    gate=evaluate_gate(metric_groups(),metrics)
    # Give the baseline a strictly larger minADE so valid records can pass.
    baseline=metric_groups();baseline['all']['minADE']=.2;baseline['N_ge_9']['minADE']=.2
    gate=evaluate_gate(baseline,metrics)
    return dict(epoch=epoch,file=f'epoch_{epoch:03d}.pt',metrics=metrics,gate=gate)


def test_actual_writer_keeps_all_and_official_top_five_and_complete_recovery(tmp_path):
    model,opt,rng,batch,state=training_objects()
    choices=[(.495,.99),(.475,.99),(.477,.99),(.470,.99),(.460,.99),(.470,.98),(.470,.98),(.480,.99)]
    for epoch,(fde,jfde) in enumerate(choices,1):
        step(model,opt,batch)
        state.update(epoch=epoch+1,completed_epoch=epoch,update=epoch,cursor=0,order=None)
        state['history'].append(record(epoch,fde,jfde,jade=.42 if epoch==5 else .39))
        persist(tmp_path,state,model,opt,rng,completed=True)
        files=list(tmp_path.glob('epoch_*.pt'));assert len(files)==epoch
        top=read(tmp_path/'TOP5.json');assert len(top)==min(5,epoch)
        assert [r['epoch'] for r in top]==[r['epoch'] for r in sorted(state['history'],key=rank_epoch)[:5]]
        for entry in top:
            assert digest(tmp_path/entry['file'])==entry['sha256']
        best=min(state['history'],key=rank_epoch)
        assert digest(tmp_path/'best.pt')==digest(tmp_path/best['file'])
        assert digest(tmp_path/'last.pt')==digest(tmp_path/f'epoch_{epoch:03d}.pt')
    assert [r['epoch'] for r in read(tmp_path/'TOP5.json')]==[6,7,4,2,3]
    for epoch in range(1,9):
        saved=torch.load(tmp_path/f'epoch_{epoch:03d}.pt',weights_only=False)
        assert saved['completed_epoch']==epoch and saved['epoch']==epoch+1
        assert saved['cursor']==0 and saved['order'] is None and saved['update']==epoch
        assert len(saved['history'])==epoch and saved['scheduler'] is None
        assert saved['optimizer']['state']
        assert set(saved['rng'])=={'python','numpy','torch','order'}
        assert saved['teacher_identity']==state['teacher_identity']

    epoch8=digest(tmp_path/'epoch_008.pt');old_top=read(tmp_path/'TOP5.json')
    state.update(cursor=32,order=torch.randperm(67,generator=rng).tolist(),epoch_updates=1)
    step(model,opt,batch);state['update']+=1
    persist(tmp_path,state,model,opt,rng,completed=False)
    assert len(list(tmp_path.glob('epoch_*.pt')))==8 and not (tmp_path/'epoch_009.pt').exists()
    assert digest(tmp_path/'epoch_008.pt')==epoch8
    assert digest(tmp_path/'last.pt')!=epoch8
    assert read(tmp_path/'TOP5.json')==old_top
    recovery=torch.load(tmp_path/'last.pt',weights_only=False)
    assert recovery['completed_epoch']==8 and recovery['epoch']==9 and recovery['cursor']==32
    assert recovery['order']==state['order']
    restored=ResidualHead();restored.load_state_dict(recovery['model'],strict=True)
    resumed_opt=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=1e-4)
    resumed_opt.load_state_dict(recovery['optimizer'])
    step(model,opt,batch);step(restored,resumed_opt,batch)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    restore_rng(recovery['rng'],rng)
    expected=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(10,generator=rng))
    random.random();np.random.rand();torch.rand(5);torch.randperm(6,generator=rng)
    restore_rng(recovery['rng'],rng)
    got=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(10,generator=rng))
    assert expected[:2]==got[:2] and torch.equal(expected[2],got[2]) and torch.equal(expected[3],got[3])
    assert not list(tmp_path.glob('*.tmp')) and not list(tmp_path.glob('*.link'))


def test_atomic_partial_save_failure_preserves_previous_recovery(tmp_path,monkeypatch):
    model,opt,rng,batch,state=training_objects()
    step(model,opt,batch);state.update(cursor=32,order=list(range(50)),update=1)
    persist(tmp_path,state,model,opt,rng,completed=False)
    before=(tmp_path/'last.pt').read_bytes()
    def interrupted_save(value,path):
        with open(path,'wb') as f:f.write(b'interrupted temporary checkpoint')
        raise RuntimeError('injected write failure before atomic replacement')
    monkeypatch.setattr(torch,'save',interrupted_save)
    with pytest.raises(RuntimeError,match='injected write failure'):
        persist(tmp_path,state,model,opt,rng,completed=False)
    assert (tmp_path/'last.pt').read_bytes()==before
    assert not list(tmp_path.glob('epoch_*.pt'))
    assert not (tmp_path/'TOP5.json').exists()
