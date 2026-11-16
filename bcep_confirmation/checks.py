"""Focused purpose, initialization, recovery and frozen selector checks."""
import copy,importlib.util,random
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import canonical_hash,check_item
from bcep.provider import BCEP
from bcep_control.mdn import MDNProvider
from selector_control.algorithms import kmeans_centers
from selector_control.core import local_seed
from .common import ROOT,state_digest,write,budget
from .protocol import PurposeDataset,validate_split
from .runner import run_pilot

def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError,AssertionError):return
    raise AssertionError('negative control unexpectedly allowed')

def cpu_checks(split,assets):
    with budget(assets,'CPU_PREPARATION',1800):
        validate_split(split);bad=copy.deepcopy(split)
        confirmation_seq=split['members'][split['heldout_confirmation_indices'][0]]['sequence']
        train_seq=split['members'][split['train_indices'][0]]['sequence']
        bad['recording_ids'][confirmation_seq]=bad['recording_ids'][train_seq]
        rejected(lambda:validate_split(bad))
        bad=copy.deepcopy(split);bad['train_indices'].append(bad['heldout_confirmation_indices'][0]);bad['member_hashes']['train']=canonical_hash([bad['members'][i] for i in bad['train_indices']])
        rejected(lambda:validate_split(bad))
        train=PurposeDataset(split,'training');idx=split['heldout_confirmation_indices'][0]
        rejected(lambda:train.get(idx,'heldout_confirmation'));assert not train.access
        scale=PurposeDataset(split,'scale');rejected(lambda:scale.get(split['selection_dev_indices'][0],'selection_dev'));assert not scale.access
        export=PurposeDataset(split,'export');rejected(lambda:export.get(idx,'heldout_confirmation',True));assert not export.access
        dev=train.get(split['selection_dev_indices'][0],'selection_dev');assert dev['purpose']=='selection_dev'
        wrong=copy.deepcopy(split['members'][split['selection_dev_indices'][0]]);wrong['agent_ids']=['wrong']*len(wrong['agent_ids']);rejected(lambda:check_item(dev,wrong))
        wrong=copy.deepcopy(split['members'][split['selection_dev_indices'][0]]);wrong['scene_id']='wrong';rejected(lambda:check_item(dev,wrong))
        initial={}
        for seed in (3102,3103):
            initial[str(seed)]={}
            for kind,factory in [('bcep_flow',BCEP),('mdn8',MDNProvider)]:
                torch.manual_seed(seed);model=factory();initial[str(seed)][kind]=dict(model=state_digest(model.state_dict()),encoder=state_digest(model.encoder.state_dict()))
            assert initial[str(seed)]['bcep_flow']['encoder']==initial[str(seed)]['mdn8']['encoder']
        assert initial['3102']['bcep_flow']['encoder']!=initial['3103']['bcep_flow']['encoder']
        spec=importlib.util.spec_from_file_location('frozen_prepilot_tests',ROOT/'tests/test_bcep_prepilot.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        class Adapter:
            def __init__(self,ds,ss,hidden):self.ds=ds;self.ss=ss;self.confirmation_truth=hidden
            def get(self,i,purpose):
                assert i in self.ss['train_indices' if purpose=='train' else 'selection_dev_indices']
                return self.ds[i]
        traces={}
        for kind in ('bcep_flow','mdn8'):
            ds,ss,cfg,identity=mod.runner_fixture();ss['selection_dev_indices']=ss.pop('dev_indices');cfg.update(model_kind=kind,lr_gamma=1.,seed=3102)
            base=Path(assets)/'precheck'/kind;seq=[];res=[]
            wrapped=Adapter(ds,ss,torch.zeros(5))
            run_pilot(wrapped,ss,cfg,identity,base/'continuous',stop_after_updates=8,on_update=lambda i,u:seq.append((i,u)))
            run_pilot(wrapped,ss,cfg,identity,base/'resume',stop_after_updates=3,on_update=lambda i,u:res.append((i,u)))
            run_pilot(wrapped,ss,cfg,identity,base/'resume',resume=True,stop_after_updates=5,on_update=lambda i,u:res.append((i,u)))
            a=torch.load(base/'continuous/last.pt',weights_only=False);b=torch.load(base/'resume/last.pt',weights_only=False)
            assert seq==res
            for key in ['model','optimizer','scheduler','rng','order','epoch','completed_epoch','update','train_cursor','dev_cursor','best','stale','selected_epoch','train_accumulator','dev_accumulator']:mod.assert_tree_equal(a[key],b[key])
            assert b['elapsed_wall_seconds']>0 and b['completed_epoch']==2
            # Same actual updates/selector with a different inaccessible confirmation label.
            wrapped.confirmation_truth=torch.ones(5)*999
            run_pilot(wrapped,ss,cfg,identity,base/'truth_replaced',stop_after_updates=8)
            c=torch.load(base/'truth_replaced/last.pt',weights_only=False)
            for key in ['model','optimizer','scheduler','rng','best','selected_epoch']:mod.assert_tree_equal(a[key],c[key])
            traces[kind]=dict(continuous_resume_exact=True,confirmation_truth_replacement_exact=True,updates=8)
        x=np.random.default_rng(9).normal(size=(256,2));seed=local_seed('bcep_flow','s','a',2035)
        before=np.random.get_state();py=random.getstate();ka=kmeans_centers(x,20,seed);kb=kmeans_centers(x,20,seed)
        np.testing.assert_array_equal(ka['centers'],kb['centers']);np.testing.assert_array_equal(before[1],np.random.get_state()[1]);assert before[2:]==np.random.get_state()[2:] and py==random.getstate()
        r=dict(status='PASS',purpose_guards=True,recording_alias_negative=True,overlap_negative=True,scene_agent_swap_negative=True,train_only_scales_access=True,
               initial_hashes=initial,same_seed_encoder_equal=True,different_seed_encoder_different=True,recovery_and_truth_negative=traces,kmeans_repeat_exact=True,global_rng_unchanged=True,formal_optimizer_updates=0)
        write(Path(assets)/'reports/CPU_PREFLIGHT.json',r);return r

def gpu_checks(assets):
    with budget(assets,'GPU_PREFLIGHT',600):
        assert torch.cuda.is_available();results={}
        g=torch.Generator().manual_seed(17);obs=torch.randn(3,8,2,generator=g).cuda()*.1;future=torch.randn(3,12,2,generator=g).cuda()*.1
        maps=torch.zeros(3,6,256,256,device='cuda');maps[:,1]=1;valid=torch.ones(3,dtype=torch.bool,device='cuda')
        for kind,factory in [('bcep_flow',BCEP),('mdn8',MDNProvider)]:
            torch.manual_seed(17);m=factory().cuda();c=m.loss_components(obs,maps,valid,future,.25);c['loss_total'].backward()
            assert torch.isfinite(c['loss_total']) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            results[kind]=float(c['loss_total']);del m,c
        torch.cuda.synchronize();torch.cuda.empty_cache()
        r=dict(status='PASS',losses=results,synthetic_only=True,formal_optimizer_updates=0,optimizer_created=False,gpu=torch.cuda.get_device_name(0))
        write(Path(assets)/'reports/GPU_PREFLIGHT.json',r);return r
