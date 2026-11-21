"""Shared identity/read-only helpers; controlled gradients allowed, optimizers denied."""
import csv,hashlib,inspect,io,json,os,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[2]
ROOT=SOURCE.parent
OUT=Path(os.environ.get('RSJG_JOINT_SCORE_AUDIT_OUT',str(ROOT/'RSJG_EBJD_assets/experiments/joint_score_identifiability_20261121'))).resolve()
assert OUT.parent==ROOT/'RSJG_EBJD_assets/experiments' and OUT.name.startswith('joint_score_identifiability_')
DOC=SOURCE/'docs/research/bcep_v3/joint_score_identifiability_20261121'
REPLAY=ROOT/'RSJG_EBJD_assets/experiments/zara2_v3_frozen_feedback_audit_20261121'
FORMAL=ROOT/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
V3RUN=ROOT/'RSJG_EBJD_assets/experiments/zara2_bcep_v3_stage2_20261120'
OLD_SOURCE=ROOT/'RSJG_BCEP_integrated_v2_20261118'
BASE='2c64c136438cac755b9a3d1723d5225a873dd707'
OLD_COMMIT='468c5eddde48bdeec47c4bb3c39418352be16bb9'
ARCHIVE='4c9eede551bebdd7634ab81b7b4bd790165c6bb0'
ARCHIVE_DIR='docs/research/bcep_v2/formal_fivefold_top5_20261118/20261119_zara2_joint_endpoint_dev/'
MODELS={
 'v2_e5':dict(epoch=5,path=FORMAL/'zara2/joint/epoch_005.pt',sha='9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317'),
 'v2_e26':dict(epoch=26,path=FORMAL/'zara2/joint/epoch_026.pt',sha='aac597735d1ac457357e9c1e5ddc74658ba073765055bd904e8d8dc9161d463a'),
 'v3_e12':dict(epoch=12,path=V3RUN/'joint/epoch_012.pt',sha='ae2ff4c81d0df7e1e42515a598208d2a2a7d6042513052bcb1fabd13bcf2be76')}
GUARD=dict(optimizer_constructions=0,optimizer_step_calls=0)


def setup(label=None):
    sys.path.insert(0,str(SOURCE))
    if label and label.startswith('v2'):
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD_SOURCE,text=True).strip()==OLD_COMMIT
        for n in ['bcep_v2/joint.py','bcep_v2/objective.py','ebjd/unified.py','bcep_v2/runtime.py']:
            assert (OLD_SOURCE/n).read_bytes()==subprocess.check_output(['git','show',OLD_COMMIT+':'+n],cwd=SOURCE)
        sys.path.insert(0,str(OLD_SOURCE))
    def init(*a,**kw):
        GUARD['optimizer_constructions']+=1;raise RuntimeError('NO OPTIMIZER permitted in this audit')
    def step(*a,**kw):
        GUARD['optimizer_step_calls']+=1;raise RuntimeError('NO optimizer.step permitted in this audit')
    torch.optim.Optimizer.__init__=init
    for cls in vars(torch.optim).values():
        if inspect.isclass(cls) and issubclass(cls,torch.optim.Optimizer):cls.step=step
    torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def array(t):return t.detach().cpu().contiguous().numpy()
def model_hash(model):
    from bcep.contracts import canonical_hash
    return canonical_hash({n:hashlib.sha256(array(t).tobytes()).hexdigest() for n,t in model.state_dict().items()})
def people(n):
    return '1' if n==1 else '2-4' if n<=4 else '5-8' if n<=8 else '9-12' if n<=12 else '13-20' if n<=20 else '21+'
def groups(n):return ['all']+(['single'] if n==1 else ['multi'])+(['N_ge_9'] if n>=9 else [])+['N_'+people(n)]
def center(s):
    s=np.asarray(s,dtype=np.float64)
    return s-s.mean(-1,keepdims=True)-s.mean(-2,keepdims=True)+s.mean((-1,-2),keepdims=True)
def effective_singular_values(s):
    # Work in the known K-1 dimensional zero-sum subspace. FP64 cancellation
    # of large gauge terms must not invent a twentieth nonzero singular value.
    c=center(s);k=c.shape[-1];q=np.zeros((k,k-1),np.float64)
    for j in range(1,k):
        q[:j,j-1]=1/np.sqrt(j*(j+1));q[j,j-1]=-j/np.sqrt(j*(j+1))
    nonzero=np.linalg.svd(q.T@c@q,compute_uv=False)
    return np.concatenate((nonzero,np.zeros(nonzero.shape[:-1]+(1,))),axis=-1)
def load_model(label):
    from bcep_joint.common import seed_all
    from bcep_v3.formal_inputs import FormalInputs
    from bcep_v2.runtime import configure_cuda
    configure_cuda();inputs=FormalInputs();seed_all(3201)
    spec=MODELS[label];assert sha(spec['path'])==spec['sha']
    state=torch.load(spec['path'],map_location='cpu',weights_only=False)
    assert state['completed_epoch']==spec['epoch']
    if label.startswith('v2'):
        from bcep_v2.joint import build_joint
        model=build_joint(inputs.config)
    else:
        from bcep_v3.model import build
        model,_=build(inputs.config)
    model.load_state_dict(state['model'],strict=True)
    assert {n for n,p in model.named_parameters() if p.requires_grad}==set(state['trainable'])
    model.cuda();model.eval()
    return model,inputs
def score_record(label,index):
    if label=='v3_e12':
        with np.load(REPLAY/'windows'/f'{index}.npz',allow_pickle=False) as z:
            return dict(S0=z['A_scores'][0],final=z['C_scores'][0],pi0=z['A_native_pi'][0],pi=z['C_pi'][0])
    with np.load(OUT/label/'windows'/f'{index}.npz',allow_pickle=False) as z:
        return {k:z[k] for k in ['S0','final','pi0','pi']}
def prepare():
    setup()
    assert not OUT.exists(),'never overwrite an audit output'
    OUT.mkdir(parents=True)
    for label,s in MODELS.items():assert sha(s['path'])==s['sha']
    before={}
    for label,root in [('v2',FORMAL),('v3',V3RUN),('candidate_cost',FORMAL.parent/'zara2_costproxy_stage2_20261119'),
                       ('sharedworld',FORMAL.parent/'zara2_sharedworld_stage2_onerun_20261119'),('previous_replay',REPLAY)]:
        for p in root.rglob('*'):
            if p.is_file():before[label+'/'+str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    write(OUT/'PROTECTED_BEFORE.json',before)
    blob=subprocess.check_output(['git','archive',ARCHIVE,ARCHIVE_DIR+'assignments/5',ARCHIVE_DIR+'assignments/26',
                                  ARCHIVE_DIR+'WINDOW_METRICS.csv'],cwd=SOURCE)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            rel=member.name.removeprefix(ARCHIVE_DIR);dest=OUT/'references'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(member).read())
    files=read(REPLAY/'WINDOW_FILES.json')
    for row in files:assert sha(REPLAY/row['file'])==row['sha256']
    write(OUT/'PROTOCOL.json',dict(base_commit=BASE,old_source_commit=OLD_COMMIT,models={k:dict(epoch=v['epoch'],sha256=v['sha']) for k,v in MODELS.items()},
        candidates=20,seed=2035,windows=590,near_gap=1e-6,sinkhorn_iterations=20,sinkhorn_score_temperature=1,
        no_temperature_search=True,optimizer_steps=0,
        reused_v3_score_windows=590,reused_v2_archived_permutations=1180,new_v2_score_windows=1180,
        new_features_windows=1770,DDIM_calls=0,
        gradient_sampling='for each N bin [1,2-4,5-8,9-12,13-20,21+], six evenly spaced positions in official dev order; same windows for all models',
        gradient_scope='native final_coordinate pair loss, canonical dev without augmentation; parameter VJPs, no optimizer',
        scores_source='old v2 native 468c5edd; v3 cached 2c64c136; no score recomputation for cached v3 inference',
        notes='gradient-enabled forwards are necessary new graph audits and are separately compared against frozen inference scores'))
    print('Prepared: weights, archived permutations, v3 cached scores, protections',flush=True)

if __name__=='__main__':prepare()


def feature_stats(x):
    x=np.asarray(x,np.float64);c=x-x.mean(-2,keepdims=True)
    raw=np.sqrt(np.mean(x*x));spread=np.sqrt(np.mean(c*c))
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    normal=np.divide(x,norms,out=np.zeros_like(x),where=norms!=0)
    cosine=np.einsum('...id,...jd->...ij',normal,normal)
    off=~np.eye(x.shape[-2],dtype=bool)
    return dict(raw_rms=float(raw),between_rms=float(spread),relative_between_rms=float(spread/raw) if raw else 0.,
                pair_cosine_distance=float((1-cosine[...,off]).mean()),pair_rms_l2=float(np.sqrt(2*x.shape[-2]/(x.shape[-2]-1)*np.mean(np.sum(c*c,axis=-1)))))
