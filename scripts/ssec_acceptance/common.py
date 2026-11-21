import csv,hashlib,inspect,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[2];ROOT=SOURCE.parent
sys.path.insert(0,str(SOURCE))
EXPERIMENTS=(ROOT/'RSJG_EBJD_assets/experiments').resolve()
OUT=Path(os.environ.get('RSJG_SSEC_ACCEPTANCE_OUT',str(EXPERIMENTS/'ssec_network_acceptance_20261121'))).resolve()
if OUT.parent!=EXPERIMENTS or not OUT.name.startswith('ssec_network_acceptance_'):
    raise ValueError('Acceptance output must be a separate ssec_network_acceptance_* directory under RSJG_EBJD_assets/experiments')
DOC=SOURCE/'docs/research/bcep_scene_energy/acceptance_20261121'
BASE='24feb26fd68b2f1a977d8de24020cc0f548851c8'
GUARD={'optimizer_step_calls':0}


def setup(cuda=False):
    def deny(*a,**kw):
        GUARD['optimizer_step_calls']+=1
        raise PermissionError('All real/synthetic optimizer.step forbidden in this task')
    for cls in vars(torch.optim).values():
        if inspect.isclass(cls) and issubclass(cls,torch.optim.Optimizer):cls.step=deny
    torch.set_num_threads(1)
    if cuda:
        from bcep_v2.runtime import configure_cuda
        configure_cuda()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state_hash(m):
    from bcep.contracts import canonical_hash
    return canonical_hash({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()})
def gradnorm(grads):return float(sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5)
def config():
    from bcep_v3.fixtures import joint_config
    return dict(joint_config(),allow_real_training=False)
def seed():
    from bcep_joint.common import seed_all
    seed_all(3201)
def snapshot():
    roots=['bcep_ebjd_v2_integrated','zara2_bcep_v3_stage2_20261120','zara2_costproxy_stage2_20261119',
           'zara2_sharedworld_stage2_onerun_20261119','zara2_v3_frozen_feedback_audit_20261121','joint_score_identifiability_20261121']
    result={}
    for root in roots:
        folder=OUT.parent/root
        for p in folder.rglob('*'):
            if p.is_file():result[root+'/'+str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return result
if __name__=='__main__':
    setup();assert not OUT.exists(),'never overwrite an acceptance run'
    OUT.mkdir(parents=True);DOC.mkdir(parents=True,exist_ok=True)
    before=snapshot();write(OUT/'PROTECTED_BEFORE.json',before)
    write(DOC/'FROZEN_CONFIG.json',dict(base=BASE,variant='energy_only',optional_variant='energy_csee',historical_reference='v3 full architecture ONLY',
        real_optimizer_updates=0,synthetic_optimizer_updates=0,allow_real_training=False,
        candidate_seed=2035,initialization_seed=3201,phi=[200,64,16],rho=[408,64,1],edge_chunk=64,
        sweeps=1,rejection_gradient='proposal_st',sinkhorn_iterations=20,sinkhorn_temperature=1.,pair_objective='final_coordinate',
        bridge_layers=6,ddim_steps=20,flow_layers=4,M=256,K=20,selection='official dev H-JADE; exact ties earlier epoch',
        unique_check='exact FP32 integer lattice minimum alternating cycle',energy_verification='detached FP64 complete scene with arithmetic-error bound',
        protected_before_files=len(before),protected_before_bytes=sum(v['bytes'] for v in before.values())))
    print('Protected',len(before),'files; frozen protocol written',flush=True)
