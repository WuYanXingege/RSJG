#!/usr/bin/env python3
"""Tiny CPU injection/observer checks; no actual goal/sampler/model forward."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_h_sampler_contract import InjectH,authenticate,Recorder,compare_tensor,SOURCES,REPO,sha256
from tools.jdv2_h_sampler_probe import Observer
from tools.jdv2_restart_observer import state_fingerprint


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__();self.original_calls=0;self.weight=torch.nn.Parameter(torch.ones(2))
    def forward(self,*args,**kwargs):
        self.original_calls+=1
        raise AssertionError('Original encoder must never run')


def checks():
    model=Stub().eval();h=torch.empty(2,3);inputs=torch.ones(3);before=inputs.clone()
    state=state_fingerprint(model);rng=torch.get_rng_state().clone();original=model.forward
    random_functions={k:getattr(torch,k) for k in ('rand','randn','randn_like','multinomial')}
    rows=[];saved=[]
    for value in (1.,2.):
        h.fill_(value)
        with InjectH(model,h) as injection,Observer() as observer:
            out=model(inputs)
            assert out is h and injection.calls==1
            observer.callback(round_index=0,score=out,mask=None,candidate_index=None,previous_candidate_index=None)
            observer.recorder.take('h',out)
            data,meta=observer.recorder.finish()
            assert not meta['invalid_snapshots'] and meta['native_dtype_roundtrip']
            saved.append(data['h'])
        assert model.forward==original and sys.getprofile() is None
    assert model.original_calls==0 and saved[0].eq(1).all() and saved[1].eq(2).all()
    assert saved[0].untyped_storage()._cdata!=saved[1].untyped_storage()._cdata
    rows.append('Injection returns exact designated object once; zero original encoder calls; per-call dump preserves distinct h before buffer reuse')
    assert inputs.equal(before) and state_fingerprint(model)==state and rng.equal(torch.get_rng_state())
    assert all(getattr(torch,k) is v for k,v in random_functions.items())
    rows.append('Inputs/parameters/global RNG unchanged; no random function replacements or callback recomputation')
    try:
        with InjectH(model,h),Observer():raise ValueError('original exception')
    except ValueError as error:assert str(error)=='original exception'
    assert model.forward==original and sys.getprofile() is None and 'forward' not in model.__dict__
    rows.append('Exception propagates; instance forward override removed and Python observation restored')
    try:
        with InjectH(model,h):model(inputs);model(inputs)
    except RuntimeError as error:assert 'more than once' in str(error)
    else:raise AssertionError('Duplicate injection allowed')
    rows.append('Second injection within one fragment refused without falling back to encoder')
    r=Recorder();r.take('early',h);h.add_(1);_,m=r.finish();assert m['invalid_snapshots']==['early']
    r=Recorder();r.take('fp32',h);r.take('bf16',torch.tensor([1.,-0.],dtype=torch.bfloat16));data,m=r.finish()
    assert m['native_dtype_roundtrip'] and not m['invalid_snapshots'] and data['bf16'].dtype==torch.bfloat16
    rows.append('Later in-place write invalidates retained early reference; terminal native FP32/BF16 roundtrip valid')
    # Test parent boundary recorder on a fixture, with one real random draw only.
    def fixture(x,generator):return x+torch.rand(x.shape,generator=generator)
    g=torch.Generator().manual_seed(9);initial=g.get_state();plain=fixture(inputs,g);plain_after=g.get_state();g.set_state(initial)
    observer=Observer();observer.codes[fixture.__code__]='fixture'
    with observer:actual=fixture(inputs,g)
    assert observer.counts['fixture']==1 and g.get_state().equal(plain_after)
    assert compare_tensor(plain,actual)['bitwise_equal'] and observer.recorder.tensors['fixture/000/output'] is actual
    rows.append('Restricted boundary profile preserves original return, arithmetic count and actual local random consumption on tiny fixture')
    hs,inputs,info=authenticate()
    rows.append('Authenticated predeclared real h pair and complete captured inputs; no real forward during authentication')
    return {'status':'PASS','checks':rows,'count':len(rows),'real_model_forward_calls':0,'real_sampler_calls':0,
        'canonical_GPU_trace_neutrality':'NOT_CERTIFIED_BY_CPU_CHECKS','tested_sources_sha256':{p:sha256(REPO/p) for p in SOURCES},'input_authentication':info}


if __name__=='__main__':print(json.dumps(checks(),indent=2))
