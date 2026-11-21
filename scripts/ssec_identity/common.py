"""Isolated zero-update identity audit; no historical files are written."""
import csv,hashlib,inspect,json,os,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[2];ROOT=SOURCE.parent
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'tests'))
BASE='54ef175c3cb2601c60ec178402831cee406eec6e'
DOC=SOURCE/'docs/research/bcep_scene_energy/candidate_identity_20261121'
EXPERIMENTS=(ROOT/'RSJG_EBJD_assets/experiments').resolve()
OUT=Path(os.environ.get('RSJG_SSEC_IDENTITY_OUT',str(EXPERIMENTS/'ssec_candidate_identity_20261121'))).resolve()
if OUT.parent!=EXPERIMENTS or not OUT.name.startswith('ssec_candidate_identity_'):
    raise ValueError('isolated ssec_candidate_identity_* output directory required')
GUARD={'optimizer_step_calls':0}
def setup(cuda=False):
    def deny(*a,**kw):
        GUARD['optimizer_step_calls']+=1;raise PermissionError('All optimizer.step forbidden')
    for c in vars(torch.optim).values():
        if inspect.isclass(c) and issubclass(c,torch.optim.Optimizer):c.step=deny
    torch.set_num_threads(1)
    if cuda:
        from bcep_v2.runtime import configure_cuda
        configure_cuda()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvwrite(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state_hash(m):
    from bcep.contracts import canonical_hash
    return canonical_hash({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()})
def old_class():
    mod=types.ModuleType('bcep_scene_energy._audit_base')
    source=subprocess.check_output(['git','show',BASE+':bcep_scene_energy/coupling.py'],cwd=SOURCE,text=True)
    exec(compile(source,BASE+':coupling.py','exec'),mod.__dict__)
    return mod.SceneEnergyCoupling

def protected():
    names=['bcep_ebjd_v2_integrated','zara2_bcep_v3_stage2_20261120','zara2_costproxy_stage2_20261119',
           'zara2_sharedworld_stage2_onerun_20261119','zara2_v3_frozen_feedback_audit_20261121','joint_score_identifiability_20261121',
           'ssec_network_acceptance_20261121']
    result={}
    for name in names:
        folder=OUT.parent/name
        assert folder.exists(),folder
        for p in folder.rglob('*'):
            if p.is_file():result[name+'/'+str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return result

def reordered(g,order):
    from ebjd.unified import gather_candidates
    return gather_candidates(g,order.transpose(1,2))
def order_for(n,device='cpu'):
    gen=torch.Generator().manual_seed(2035)
    return torch.stack([torch.randperm(20,generator=gen) for _ in range(n)])[None].to(device)
def map_pi(pi,order):return order.gather(2,pi.transpose(1,2)).transpose(1,2)
def relative_changes(a,b):
    a=a.detach().cpu()[0].numpy();b=b.detach().cpu()[0].numpy()
    return int(np.count_nonzero(a[np.argsort(a[:,0])]!=b[np.argsort(b[:,0])]))
def norm(gs):return float(sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5)

def prepare():
    setup();assert not OUT.exists(),'never overwrite an audit root';OUT.mkdir(parents=True)
    write(OUT/'PROTECTED_BEFORE.json',protected())
    from bcep_v3.formal_inputs import FormalInputs
    inputs=FormalInputs();rows=[]
    for bank,purpose in zip(inputs.banks,['train','selection_dev']):
        for i in inputs.split[purpose+'_indices']:
            g=bank.native.get(i).numpy().transpose(1,0,2)
            delta=np.abs(g[:,:,None]-g[:,None,:]).max(-1)
            delta[:,np.arange(20),np.arange(20)]=np.inf
            duplicates=int(np.count_nonzero(delta==0)//2)
            rows.append(dict(purpose=purpose,index=i,N=len(g),duplicate_pairs=duplicates,min_linf_separation_m=float(delta.min()),
                row_sha256=bank.native.rows[i]['sha256']))
    csvwrite(DOC/'BANK_IDENTITY_SCAN.csv',rows)
    write(DOC/'BANK_IDENTITY.json',dict(bank_hashes=[b.hash for b in inputs.banks],provider=inputs.provider,counts=inputs.counts,
        schema='member-identity-sha256-v2',candidate_field='no separate per-candidate ID array',
        immutable_occurrence_key=['bank hash','row member/digest','agent ID','original KMeans slot'],
        duplicates_total=sum(r['duplicate_pairs'] for r in rows),
        duplicate_windows=sum(r['duplicate_pairs']>0 for r in rows),optimizer_guard=GUARD))
    print('Prepared; duplicate candidate pairs:',sum(r['duplicate_pairs'] for r in rows),flush=True)
if __name__=='__main__':prepare()

def seed():
    from bcep_joint.common import seed_all
    seed_all(3201)
def config():
    from bcep_v3.fixtures import joint_config
    return dict(joint_config(),allow_real_training=False)

gradnorm=norm
