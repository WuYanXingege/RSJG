import copy
import torch
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from joint_marginal_residual.features import coverage_statistics,frozen_features
from joint_marginal_residual.model import ResidualHead
from joint_marginal_residual.objective import metric_sums,paired_objective,aggregate_metric_sums
from joint_marginal_residual.selection import evaluate_gate,rank_epoch,select_epoch


def example(n=4):
    g=torch.Generator().manual_seed(2035)
    raw=torch.randn(256,n,2,generator=g)
    ids=torch.arange(20)*3
    observed=torch.randn(n,8,2,generator=g)
    context=torch.randn(n,128,generator=g)
    paths=torch.randn(20,n,12,2,generator=g)
    valid=torch.ones(n,dtype=torch.bool)
    features=frozen_features(observed,context,paths,coverage_statistics(raw,ids,valid),valid)
    truth=torch.randn(n,12,2,generator=g)
    return raw,ids,observed,context,paths,valid,features,truth


def test_assignment_gap_raw_id_ties_empty_clusters_singleton():
    raw=torch.tensor([[[0.,0.],[0.,0.]],[[0.,0.],[1.,0.]],[[1.,0.],[0.,0.]],[[1.,0.],[1.,0.]]])
    c=coverage_statistics(raw,torch.tensor([3,0]),return_assignments=True)
    assert c['j_slot'].tolist()==[1,1,1,0]
    assert c['p_slot'].tolist()==[[1,1],[1,0],[0,1],[0,0]]
    assert c['diagnostics']['joint_coverage_m']==.25
    assert c['diagnostics']['personal_coverage_m']==0.
    assert c['diagnostics']['mean_assignment_excess_m']==.25
    duplicate=torch.zeros(4,2,2)
    d=coverage_statistics(duplicate,torch.tensor([3,0]))
    assert not d['stats_j'][0].count_nonzero()
    assert not d['stats_p'][0].count_nonzero()
    single=coverage_statistics(torch.randn(12,1,2),torch.tensor([8,2,6]))
    assert torch.equal(single['stats_j'],single['stats_p'])
    assert not single['mean_a_j'].count_nonzero()


def test_features_agent_and_slot_equivariance_and_no_grad():
    raw,ids,observed,ctx,b,valid,f,_=example(5)
    valid[-1]=False
    c=coverage_statistics(raw.requires_grad_(),ids,valid)
    f=frozen_features(observed.requires_grad_(),ctx.requires_grad_(),b.requires_grad_(),c,valid)
    assert not f['base_features'].requires_grad and not f['difference_features'].requires_grad
    assert not f['base_features'][:,~valid].count_nonzero()
    people=torch.tensor([4,2,0,3,1]);slots=torch.arange(19,-1,-1)
    d=coverage_statistics(raw[:,people],ids[slots],valid[people])
    other=frozen_features(observed[people],ctx[people],b[slots][:,people],d,valid[people])
    for name in ('base_features','difference_features'):
        assert torch.equal(other[name],f[name][slots][:,people])


def test_zero_head_two_readout_gradients_and_equivariance():
    _,_,_,_,b,valid,f,truth=example(4)
    torch.manual_seed(3201);head=ResidualHead()
    teacher=b.clone().requires_grad_()
    out,detail=head(teacher,f['base_features'],f['difference_features'],valid,return_details=True)
    assert torch.equal(out,b)
    scales=torch.ones(4,requires_grad=True)
    teacher_parts=[metric_sums(teacher[None],truth[None],valid[None])]
    loss=paired_objective([metric_sums(out[None],truth[None],valid[None])],teacher_parts,scales)
    loss['loss'].backward()
    assert head.common_readout[-1].weight.grad.norm()>0
    assert head.person_readout[-1].weight.grad.norm()>0
    assert teacher.grad is None and scales.grad is None
    torch.optim.AdamW(head.parameters(),lr=3e-4).step()
    y,detail=head(b,f['base_features'],f['difference_features'],valid,return_details=True)
    torch.testing.assert_close(detail['relative'].sum(1),torch.zeros(20,4),atol=1e-7,rtol=0)
    slots=torch.arange(19,-1,-1);people=torch.tensor([2,0,3,1])
    other=head(b[slots][:,people],f['base_features'][slots][:,people],f['difference_features'][slots][:,people],valid[people])
    torch.testing.assert_close(other,y[slots][:,people],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(y[:,:,-1]-b[:,:,-1],detail['dg'],atol=1e-7,rtol=1e-6)


def test_control_keeps_structure_and_single_person_common_can_move():
    _,_,_,_,b,valid,f,_=example(1)
    torch.manual_seed(3201);head=ResidualHead();control=copy.deepcopy(head)
    assert list(head.state_dict())==list(control.state_dict())
    for k,v in head.state_dict().items():assert torch.equal(v,control.state_dict()[k])
    with torch.no_grad():head.common_readout[-1].bias.fill_(.1)
    normal,detail=head(b,f['base_features'],f['difference_features'],valid,return_details=True)
    assert not detail['relative'].count_nonzero()
    assert not torch.equal(normal,b)
    direct=head(b,f['base_features'],torch.zeros_like(f['difference_features']),valid)
    ablation=head(b,f['base_features'],f['difference_features'],valid,ablation=True)
    assert torch.equal(direct,ablation)


def test_logical_batch_global_weighting_matches_native_and_gradient():
    torch.manual_seed(10)
    b=torch.randn(2,20,7,12,2)
    truth=torch.randn(2,7,12,2)
    valid=torch.tensor([[1,0,0,0,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
    delta=torch.randn_like(b)*.03
    full=(b+delta).detach().requires_grad_()
    split=(b+delta).detach().requires_grad_()
    teacher=b.detach().requires_grad_()
    scales=torch.tensor([.2,.3,.4,.8],requires_grad=True)
    one=paired_objective([metric_sums(full,truth,valid)],[metric_sums(teacher,truth,valid)],scales)
    chunks=paired_objective([metric_sums(split[i:i+1],truth[i:i+1],valid[i:i+1]) for i in range(2)],
        [metric_sums(teacher[i:i+1],truth[i:i+1],valid[i:i+1]) for i in range(2)],scales)
    assert torch.equal(one['loss'],chunks['loss'])
    one['loss'].backward();chunks['loss'].backward()
    assert torch.equal(full.grad,split.grad)
    assert teacher.grad is None and scales.grad is None
    acc=MetricAccumulator();acc.update(full.detach(),truth,valid);ref=acc.as_dict()
    assert one['student'].tolist()==[ref[k] for k in ('minADE','minFDE','JADE','JFDE')]
    stats=hard_metric_statistics(full.detach(),truth,valid)
    assert float(one['student'][1].detach())!=float(stats.scene_marginal_fde.mean())


def metrics(fde=.5,jfde=1.,ade=.2,jade=.4,dense=True):
    row=dict(minFDE=fde,JFDE=jfde,minADE=ade,JADE=jade,scene_count=10,agent_count=30)
    return {'all':row,'N_ge_9':dict(row,scene_count=2,agent_count=20) if dense else None}


def test_unrounded_gates_strict_zero_and_official_top5_ranking():
    baseline=metrics()
    student=metrics(fde=.47,jfde=.99,ade=.19,jade=.39)
    gate=evaluate_gate(baseline,student);assert gate['passed']
    strict=metrics(fde=.47,jfde=.99,ade=.2,jade=.39)
    bad=evaluate_gate(baseline,strict)
    assert not bad['passed'] and bad['max_violation_m']==0.
    assert 'all.minADE' in bad['hardest_conditions']
    records=[dict(epoch=2,metrics=strict,gate=bad),dict(epoch=5,metrics=student,gate=gate),dict(epoch=4,metrics=student,gate=gate)]
    assert select_epoch(records)['epoch']==4
    assert sorted(records,key=rank_epoch)[-1]['epoch']==2
    empty=evaluate_gate(metrics(dense=False),metrics(fde=.47,jfde=.99,ade=.19,jade=.39,dense=False))
    assert empty['passed'] and sum(c['status']=='N/A' for c in empty['checks'] if not c['available'])==2
