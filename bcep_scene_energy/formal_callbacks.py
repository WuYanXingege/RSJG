"""Reuse the native grouped graph; bind verified, unchanged original bank slots."""
import time
import torch
from .formal_monitor import decision_statistics


class SlotCouplingView:
    def __init__(self,original,observer=None):
        self.original=original;self.observer=observer;self.last=None
    def __getattr__(self,name):return getattr(self.original,name)
    def assign(self,context,goals,scale,**kwargs):
        # The frozen native callback augments xy and compacts scenes/people only:
        # it NEVER permutes K. Thus each current column still has its ORIGINAL
        # verified bank slot. External shuffled inputs must supply transported IDs.
        kwargs.setdefault('candidate_ids',torch.arange(20,device=goals.device).expand(len(goals),context.valid.shape[1],20))
        requested=kwargs.pop('return_details',False)
        if goals.is_cuda:torch.cuda.synchronize(goals.device)
        start=time.perf_counter()
        result=self.original.assign(context,goals,scale,return_details=True,**kwargs)
        if goals.is_cuda:torch.cuda.synchronize(goals.device)
        elapsed=time.perf_counter()-start
        observations=decision_statistics(result[3],kwargs['straight_through'])
        self.last=dict(initial_pi=result[3]['initial_pi'].detach(),observations=observations,coupling_forward_seconds=elapsed)
        if self.observer is not None:
            self.observer.groups.append(dict(valid_people=context.valid.sum(1).tolist(),coupling_forward_seconds=elapsed,scenes=observations))
        return result if requested else result[:3]


class SlotModelView:
    def __init__(self,model,observer=None):self.original=model;self.coupling=SlotCouplingView(model.coupling,observer)
    def __getattr__(self,name):return getattr(self.original,name)
    def eval(self):self.original.eval();return self


def formal_callbacks(model,inputs,device,observer=None):
    from bcep_v3.formal_inputs import FormalInputs
    from bcep_v2.stages import joint_callbacks
    if type(inputs) is not FormalInputs:raise TypeError('verified official ZARA2 train/dev inputs required')
    if inputs.config['pair_objective']!='final_coordinate':raise ValueError('frozen coordinate objective required')
    if model.coupling.csee is not None:raise ValueError('Energy-only required')
    view=SlotModelView(model,observer)
    native,_=joint_callbacks(view,inputs.data,device,inputs.config,*inputs.banks)
    def backward(*args):
        if observer is not None:observer.groups=[]
        if str(device).startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();metrics=native(*args)
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        if observer is not None:
            observer.capture_backward(metrics,time.perf_counter()-start,torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
            scenes=[r for group in observer.groups for r in group['scenes']]
            for k in ['attempted','accepted','rejected','strict_ties','reliable_decreases','roundoff_uncertain','accepted_energy_delta']:
                metrics['ssec_'+k]=sum(r[k] for r in scenes)
        return metrics
    backward.view=view
    return backward
