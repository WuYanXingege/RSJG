"""CPU-only analysis of already saved small evidence; no loss/model calls."""
import hashlib
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
def load(name):return torch.load(HERE/'gpu_run'/name,map_location='cpu',weights_only=True)
def delta(a,b):
    return {'exact':torch.equal(a,b),'max_abs':float((a.double()-b.double()).abs().max()) if a.numel() else 0.}
raw=json.loads((HERE/'gpu_run/RESULTS.json').read_text())
r2,r3=load('R2.pt')['effective_inputs'],load('R3.pt')['effective_inputs']
result={'real_FP32_vs_BF16_effective_inputs':[{k:delta(x[k],y[k]) for k in ('u','q','C','z')} for x,y in zip(r2,r3)],
        'repeats':{},'synthetic_max_abs':{},'real_reference_max_abs':max(c['max_abs'] for r in raw['real'] for c in r['same_effective_input_references'])}
for mode in ('fp32','bf16'):
    x,y=load('asymmetric_'+mode+'.pt'),load('repeat_'+mode+'.pt')
    result['repeats'][mode]={'value':delta(x['value'],y['value']),'grads':{k:delta(x['grads'][k],y['grads'][k]) for k in x['grads']}}
for label in raw['synthetic'][0]['comparisons']:
    result['synthetic_max_abs'][label]=max(c['max_abs'] for row in raw['synthetic'] for c in row['comparisons'][label].values())
result['real_hashes_all_equal']=len({row[key] for row in raw['real'] for key in ('before_state_sha256','after_state_sha256')})==1
result['cuda_initialized']=torch.cuda.is_initialized()
assert not result['cuda_initialized']
print(json.dumps(result,indent=2))
