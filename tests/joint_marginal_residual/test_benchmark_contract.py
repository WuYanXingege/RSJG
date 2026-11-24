import copy
import torch
from joint_marginal_residual.benchmark import compare,TOLERANCE_POLICY,_ids,performance_rejections,implicated_optimization


def test_noise_and_raw_id_remain_exact_even_inside_float_tolerance(tmp_path):
    a={'stage':{'path':torch.ones(2),'noise':torch.ones(2),'raw_indices':torch.tensor([3,1])}}
    b=copy.deepcopy(a);b['stage']['noise'][0]=torch.nextafter(torch.tensor(1.),torch.tensor(2.))
    torch.save(a,tmp_path/'a.pt');torch.save(b,tmp_path/'b.pt')
    result=compare(tmp_path/'a.pt',tmp_path/'b.pt',TOLERANCE_POLICY)
    assert not result['passed'] and result['failures'][0]['field']=='/stage/noise'
    b=copy.deepcopy(a);b['stage']['raw_indices']=torch.tensor([1,3]);torch.save(b,tmp_path/'b.pt')
    assert not compare(tmp_path/'a.pt',tmp_path/'b.pt',TOLERANCE_POLICY)['passed']


def test_fixed_float_budget_and_exact_metric_winners(tmp_path):
    a={'path':torch.ones(2),'gradient':torch.ones(2),'winners':{'jfde':torch.tensor([1])}}
    b=copy.deepcopy(a);b['path'][0]=torch.nextafter(torch.tensor(1.),torch.tensor(2.))
    torch.save(a,tmp_path/'a.pt');torch.save(b,tmp_path/'b.pt')
    assert compare(tmp_path/'a.pt',tmp_path/'b.pt',TOLERANCE_POLICY)['passed']
    b['winners']['jfde'][0]=2;torch.save(b,tmp_path/'b.pt')
    assert not compare(tmp_path/'a.pt',tmp_path/'b.pt',TOLERANCE_POLICY)['passed']
    b=copy.deepcopy(a);b['path'][0]+=0.001;torch.save(b,tmp_path/'b.pt')
    assert not compare(tmp_path/'a.pt',tmp_path/'b.pt',TOLERANCE_POLICY)['passed']


def test_real_population_selection_is_fixed_and_preserves_logical_count():
    split={'train_indices':list(range(20)),'members':[{'agent_ids':list(range(n))} for n in [2,3,9,1]+[2]*16]}
    selected=_ids(split,'train',16)
    assert len(selected)==len(set(selected))==16
    assert selected[:3]==[3,0,2]


def test_timing_admission_needs_median_gain_and_paired_wins():
    observations=[]
    for variant in ('baseline','optimized'):
        for trial in range(3):
            time=1. if variant=='baseline' else [0.95,1.01,1.01][trial]
            observations.append(dict(variant=variant,trial=trial,observed_to_B_seconds=time,
                stages={s:dict(update_seconds=time,validation_seconds=time) for s in ('density','head','coordinate')}))
    rejected=performance_rejections(observations,set())
    assert len(rejected)==5
    assert performance_rejections(observations,{'modulation','proposal','schedule','hungarian','geometry','logging','head_cache'})==[]
    assert implicated_optimization([{'field':'/coordinate/outputs/0/permutation'}],set())=='hungarian'
