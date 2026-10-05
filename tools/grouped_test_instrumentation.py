"""CPU fixture accounting, separate from real train-data smoke attempts."""
from collections import Counter
import time
counts=Counter();optimizers=[]
def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**k):
        counts['cuda_lazy_init_attempts']+=1
        raise RuntimeError('CPU test session forbids CUDA initialization')
    torch.cuda._lazy_init=forbidden
    for cls in (torch.optim.Adam,torch.optim.SGD,torch.optim.AdamW):
        original=cls.step
        def wrapper(self,*a,_old=original,_name=cls.__name__,**k):
            counts['synthetic_optimizer_attempts']+=1
            n=sum(p.numel() for g in self.param_groups for p in g['params'])
            if n>10000:raise RuntimeError('real-model optimizer forbidden in CPU fixture suite')
            try:value=_old(self,*a,**k)
            except Exception:
                counts['synthetic_optimizer_failed']+=1;raise
            counts['synthetic_optimizer_successful']+=1;optimizers.append(dict(type=_name,parameters=n));return value
        cls.step=wrapper
def pytest_sessionfinish(session,exitstatus):
    from src.p2_protocol import ROOT
    from src.p2_observation import atomic_json
    d=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight'
    atomic_json(dict(scope='synthetic CPU pytest fixtures; not real train-label updates',counts=dict(counts),optimizers=optimizers,exit=int(exitstatus)),d/('CPU_FIXTURE_ACCOUNTING_'+str(time.time_ns())+'.json'))
