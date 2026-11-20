"""Observe actual native optimizer steps without changing their update rule."""
import contextlib,json,time
import torch
from bcep_v2 import state as native
from .model import NEW_PREFIXES


class ModuleTelemetry:
    def __init__(self,model,path=None,start_update=0,domain='real'):
        self.model=model;self.path=path;self.update=start_update;self.domain=domain;self.rows=[]
        self.named={n:p for n,p in model.named_parameters() if n.startswith(NEW_PREFIXES)}
    def capture_backward(self):
        self.before={n:p.detach().clone() for n,p in self.named.items()}
        self.grad={}
        for prefix in NEW_PREFIXES:
            tensors=[p.grad for n,p in self.named.items() if n.startswith(prefix) and p.grad is not None]
            if any(not torch.isfinite(g).all() for g in tensors):raise FloatingPointError(prefix+' nonfinite gradient')
            self.grad[prefix]=dict(gradient_l2=sum(float(g.double().square().sum()) for g in tensors)**.5,
                                  present_tensors=len(tensors))
        return self.grad
    def step_done(self,*args):
        self.update+=1;row=dict(update=self.update,domain=self.domain,unix=time.time(),modules={})
        for prefix in NEW_PREFIXES:
            values=[(p.detach()-self.before[n]).double() for n,p in self.named.items() if n.startswith(prefix)]
            row['modules'][prefix]=dict(self.grad[prefix],parameter_delta_l2=sum(float(x.square().sum()) for x in values)**.5,
                                       parameter_delta_max=max(float(x.abs().max()) for x in values))
        self.rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.before={}
    @contextlib.contextmanager
    def native_optimizer_hook(self):
        original=native.optimizer
        def factory(*a,**kw):
            opt=original(*a,**kw);opt.register_step_post_hook(self.step_done);return opt
        native.optimizer=factory
        try:yield
        finally:native.optimizer=original


@contextlib.contextmanager
def no_optimizer_steps():
    original=torch.optim.AdamW.step
    def denied(*a,**kw):raise PermissionError('real-data readiness audit forbids optimizer.step')
    torch.optim.AdamW.step=denied
    try:yield
    finally:torch.optim.AdamW.step=original
