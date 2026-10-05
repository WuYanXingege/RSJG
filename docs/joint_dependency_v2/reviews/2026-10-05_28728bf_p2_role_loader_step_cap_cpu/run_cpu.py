"""Bounded CPU-only pytest recorder; explicit test paths, no full-suite discovery."""
import contextlib
from collections import Counter
import io
import json
from pathlib import Path
import sys
import time
import math
R=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(R),str(R/'tests')]
import torch
import pytest
torch.set_num_threads(1)
counts=Counter();reports=[]
def forbidden(*a,**kw):
    counts['forbidden_attempts']+=1
    raise AssertionError('CPU acceptance prohibits real network/CUDA')
for name in ('_lazy_init','init','is_available','device_count','synchronize','get_device_name','get_device_properties'):
    setattr(torch.cuda,name,forbidden)
# CPU Adam otherwise probes CUDA availability even for CPU-only parameters.
# Replace only this diagnostic with a checked CPU no-op, never the update math.
def cpu_health(self):
    assert all(p.device.type=='cpu' for group in self.param_groups for p in group['params'])
    counts['cpu_optimizer_health_noops']+=1
torch.optim.Optimizer._cuda_graph_capture_health_check=cpu_health
from src.models.model import GDTS
from src.models.goal_pretrain import Goal_Pretrain
GDTS.__init__=forbidden;GDTS.forward=forbidden;Goal_Pretrain.__init__=forbidden;Goal_Pretrain.forward=forbidden
for cls in (torch.optim.Adam,torch.optim.SGD,torch.optim.AdamW):
    old=cls.step
    def counted(self,*a,_old=old,**kw):
        counts['toy_optimizer_attempts']+=1
        result=_old(self,*a,**kw)
        counts['toy_optimizer_completed']+=1
        return result
    cls.step=counted
original_backward=torch.Tensor.backward
def backward(self,*a,**kw):
    counts['synthetic_backward_calls']+=1
    return original_backward(self,*a,**kw)
torch.Tensor.backward=backward
original_load=torch.load
def load(p,*a,**kw):
    if isinstance(p,(str,Path)) and not str(Path(p).resolve()).startswith('/tmp/'):
        raise AssertionError('No real checkpoint/payload load in CPU fixture runner')
    counts['synthetic_loads']+=1
    return original_load(p,*a,**kw)
torch.load=load
class Reporter:
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:
            reports.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome,
                seconds=report.duration,error=str(report.longrepr) if report.failed else None))
out=Path(sys.argv[1]);assert not out.exists()
started=time.monotonic();stream=io.StringIO()
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    code=pytest.main(['-q','-p','no:cacheprovider',*sys.argv[2:]],plugins=[Reporter()])
import mc_wiring_fixture as mc
p2=sys.modules.get('test_p2_role_loader_step_cap')
result=dict(exit_code=int(code),seconds=time.monotonic()-started,reports=reports,counts=dict(counts),
    fixture_counts=dict(mc.COUNTS),p2_counts=dict(p2.COUNTS) if p2 else {},bounded_traces=p2.TRACES if p2 else [],
    mc_measurements=getattr(sys.modules.get('test_jdv2_mc_wiring'),'MEASUREMENTS',{}),
    real_models=0,real_data_updates=0,protected_future_reads=0,cuda_execution=0,
    stdout=stream.getvalue(),command=sys.argv)
def safe(v):
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,list):return [safe(x) for x in v]
    return v
out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(exit_code=int(code),seconds=result['seconds'],outcomes=dict(Counter(x['outcome'] for x in reports)),
                     counts=dict(counts),record=str(out))))
raise SystemExit(code)
