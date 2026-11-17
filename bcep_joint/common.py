import hashlib,json,os,random,signal,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import digest,canonical_hash
from bcep_confirmation.common import read,write,budget,state_digest
ROOT=Path(__file__).resolve().parents[1]
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def local_seed(*args):return int.from_bytes(hashlib.sha256(json.dumps(args,separators=(',',':')).encode()).digest()[:8],'little')%(2**63-1)
def save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def source_identity():
    paths=[p for folder in ['ebjd','bcep','selector_control','bcep_confirmation','bcep_joint'] for p in sorted((ROOT/folder).rglob('*.py'))]
    paths += [ROOT/'scripts'/name for name in ['run_bcep_joint.py','prepare_bcep_joint.py','bcep_pilot_runtime.py','run_zara1_unified_coupling_bridge.py']]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def rng_state(gen):return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=gen.get_state())
def restore_rng(r,gen):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);gen.set_state(r['loader'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
@contextmanager
def forbid_weight_reads():
    original=torch.load
    def denied(*a,**kw):raise RuntimeError('downstream builder may not read pretrained weights')
    torch.load=denied
    try:yield
    finally:torch.load=original
