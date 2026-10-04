#!/usr/bin/env python3
"""Small CPU fixture observer tests and authenticated real input; no real module forward."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_message_observer import MessageObserver,compare_traces
from tools.jdv2_message_diagnostic import authenticate
from tools.jdv2_restart_observer import Recorder,state_fingerprint,compare_tensor
from tools.jdv2_stage_a_bank import REPO,sha256


class CountSequential(torch.nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args);self.calls=0
    def forward(self,x):
        self.calls+=1
        return super().forward(x)


class Fixture(torch.nn.Module):
    """CPU-only tiny fixture with matched observation anchors, not a GPU replay implementation."""
    def __init__(self):
        super().__init__()
        self.message_mlp=CountSequential(torch.nn.Linear(10,4),torch.nn.ReLU(inplace=True),torch.nn.Linear(4,4))
        self.update_mlp=CountSequential(torch.nn.Linear(8,4),torch.nn.ReLU(inplace=True),torch.nn.Linear(4,4))
        self.norm=torch.nn.LayerNorm(4);self.fail=False
    def forward(self,agent_feat,edge_index,edge_feat,edge_weight):
        if self.fail:raise ValueError('fixture original exception')
        source,target=edge_index
        message_to_source=self.message_mlp(torch.cat((agent_feat[source],agent_feat[target],edge_feat),-1))
        message_to_target=self.message_mlp(torch.cat((agent_feat[target],agent_feat[source],edge_feat),-1))
        message_to_source=message_to_source*edge_weight[:,None]
        message_to_target=message_to_target*edge_weight[:,None]
        aggregated = torch.zeros_like(agent_feat)
        aggregated.index_add_(0,source,message_to_source)
        aggregated.index_add_(0,target,message_to_target)
        normalizer = agent_feat.new_zeros((agent_feat.shape[0],))
        normalizer.index_add_(0,source,edge_weight)
        normalizer.index_add_(0,target,edge_weight)
        aggregated = aggregated / normalizer.clamp_min(1.0)[:,None]
        update = self.update_mlp(torch.cat((agent_feat,aggregated),-1))
        return self.norm(agent_feat+update)


def checks():
    model=Fixture().eval();x=torch.arange(12,dtype=torch.float32).reshape(3,4)/10
    edges=torch.tensor([[0,0,1],[1,2,2]]);feat=torch.ones(3,2);weights=torch.tensor([.5,.7,.9])
    args=(x,edges,feat,weights);original=[t.clone() for t in args]
    state=state_fingerprint(model);before=torch.get_rng_state().clone();rows=[]
    with torch.no_grad():
        plain=model(*args)
        with MessageObserver(model) as observer:
            actual=model(*args)
        assert actual is observer.recorder.tensors['message/final_return']
        assert actual is observer.recorder.tensors['norm/output']
        assert compare_tensor(plain,actual)['bitwise_equal']
        assert model.message_mlp.calls==4 and model.update_mlp.calls==2
    assert before.equal(torch.get_rng_state()) and state_fingerprint(model)==state
    assert all(a.equal(b) for a,b in zip(args,original))
    rows.append('original return identity and equal CPU output; expected two MLP calls and one update per forward; RNG/input/parameters unchanged')
    assert sys.gettrace() is None and sys.getprofile() is None
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    rows.append('Python trace and parent-module hooks restored after scope')
    data,meta=observer.finish();assert not meta['invalid_snapshots'] and meta['native_dtype_roundtrip']
    assert meta['tensors']['feature_accumulation/raw_after_both_adds']['version_at_capture']==2
    assert meta['tensors']['normalizer/after_both_adds']['version_at_capture']==2
    assert not meta['first_add_snapshot_captured'] and not meta['linear_pre_inplace_ReLU_snapshot_captured']
    rows.append('post-both-add snapshots valid despite in-place ReLU inside parent MLP; native dtype roundtrip')
    model.fail=True
    try:
        with MessageObserver(model):model(*args)
    except ValueError as error:assert str(error)=='fixture original exception'
    else:raise AssertionError('swallowed exception')
    assert sys.gettrace() is None and all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    rows.append('original exceptions propagate and scope restores observation state')
    accumulator=torch.zeros(3,4);normalizer=torch.zeros(3);early=Recorder()
    accumulator.index_add_(0,edges[0],x);normalizer.index_add_(0,edges[0],weights)
    early.take('first_feature',accumulator);early.take('first_normalizer',normalizer)
    accumulator.index_add_(0,edges[1],x);normalizer.index_add_(0,edges[1],weights)
    _,invalid=early.finish();assert set(invalid['invalid_snapshots'])=={'first_feature','first_normalizer'}
    late=Recorder();late.take('raw',accumulator);late.take('normalizer',normalizer)
    normalized=accumulator/normalizer.clamp_min(1)[:,None];late.take('normalized',normalized)
    _,good=late.finish();assert not good['invalid_snapshots']
    assert good['tensors']['raw']['alias_group']!=good['tensors']['normalized']['alias_group']
    rows.append('early feature and normalizer refs invalidate on second add; terminal refs remain valid after out-of-place division')
    self_gate=compare_traces(data,data,meta,meta);assert not self_gate['gate_pass']
    changed=dict(data);changed['feature_accumulation/raw_after_both_adds']=data['feature_accumulation/raw_after_both_adds']+1
    gate=compare_traces(data,changed,meta,meta);assert gate['gate_pass'] and gate['eligible_region']=='feature_accumulation'
    changed['weighted/source']=data['weighted/source']+1
    assert not compare_traces(data,changed,meta,meta)['gate_pass']
    invalid_meta={**meta,'invalid_snapshots':['feature_accumulation/raw_after_both_adds']}
    assert not compare_traces(data,data,meta,invalid_meta)['gate_pass']
    rows.append('extra gate rejects equal traces, changed upstream entrance and invalid snapshots; selects only first eligible region')
    assert not compare_tensor(torch.tensor([0.]),torch.tensor([-0.]))['bitwise_equal']
    assert not compare_tensor(torch.tensor([1.]),torch.tensor([1.],dtype=torch.bfloat16))['bitwise_equal']
    rows.append('signed-zero/dtype differences not mistaken for bitwise equality')
    inputs,layer,info=authenticate();assert len(inputs)==4 and len(layer.state_dict())==10
    rows.append('real T0/T1 inputs authenticated, agent hash matched, strict10-key message state equals historical encoder state; zero real forward')
    return {'status':'PASS','device':'CPU_ONLY','checks':rows,'count':len(rows),'real_message_forward_calls':0,
        'canonical_GPU_neutrality':'NOT_CERTIFIED_BY_CPU_CHECKS',
        'tested_sources_sha256':{p:sha256(REPO/p) for p in ('tools/jdv2_message_observer.py','tools/jdv2_message_diagnostic.py','tools/jdv2_message_cpu.py')},
        'input_authentication':info}


if __name__=='__main__':print(json.dumps(checks(),indent=2))
