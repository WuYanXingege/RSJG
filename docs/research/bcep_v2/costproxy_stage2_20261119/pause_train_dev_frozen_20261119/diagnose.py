"""One frozen train/dev comparison; no optimizer, augmentation, trajectories or DDIM.
Arguments are local paths; public outputs contain anonymous window indices only.
"""
import argparse,io,csv,json,hashlib,os,sys,subprocess,time
from pathlib import Path
import numpy as np
import torch
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
P='docs/research/bcep_v2/formal_fivefold_top5_20261118/'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def csvout(p,rows):
 t=p.with_suffix('.tmp')
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 t.replace(p)
def main(a):
 sys.path.insert(0,str(a.repo))
 from bcep_v2.joint import build_joint
 from bcep_v2.bank import Bank
 from bcep_v2.runtime import configure_cuda
 from bcep_formal.data import FormalData
 from bcep_joint.common import seed_all
 from bcep.contracts import canonical_hash
 from ebjd.unified import gather_candidates
 configure_cuda();torch.set_num_threads(1)
 up=a.workspace/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated';trial=a.workspace/'RSJG_EBJD_assets/experiments/zara2_costproxy_stage2_20261119';out=a.out;out.mkdir(exist_ok=True,parents=True)
 assert read(up/'STATUS.json')['status']=='PAUSED_RESUMABLE' and read(trial/'STATUS.json')['status'] in ['SESSION_BUDGET_PAUSED','COMPLETE']
 def inventory(r):return {str(p.relative_to(r)):[p.stat().st_size,p.stat().st_mtime_ns] for p in r.rglob('*') if p.is_file()}
 before={name:inventory(r) for name,r in [('old',up),('new',trial)]};write(out/'LOCAL_ASSETS_BEFORE.json',before)
 config=read(trial/'FROZEN_CONFIG.json');run=read(up/'FORMAL_RUN.json');plan=read(run['input_plan']);d=plan['folds']['zara2']['split'];assert sha(d['path'])==d['sha256'];split=read(d['path']);assert split['unit']=='m'
 def gitfile(commit,name):return subprocess.check_output(['git','show',commit+':'+P+name],cwd=a.repo)
 manifest=json.loads(gitfile('4c9eede551bebdd7634ab81b7b4bd790165c6bb0','20261119_zara2_joint_endpoint_dev/ARTIFACT_SHA256.json'))
 raw=gitfile('4c9eede551bebdd7634ab81b7b4bd790165c6bb0','20261119_zara2_joint_endpoint_dev/WINDOW_METRICS.csv');assert hashlib.sha256(raw).hexdigest()==manifest['WINDOW_METRICS.csv']
 olddev=list(csv.DictReader(io.StringIO(raw.decode())));oldlookup={(int(r['epoch']),int(r['index']),r['method']):float(r['JFDE_m']) for r in olddev}
 oldcp=list(csv.DictReader(io.StringIO(gitfile('4c9eede551bebdd7634ab81b7b4bd790165c6bb0','20261119_zara2_joint_endpoint_dev/CHECKPOINTS.csv').decode())))
 models={};checks=[]
 for label,root in [('old',up/'zara2'),('new',trial)]:
  for epoch in [1,3,5]:
   path=root/'joint'/f'epoch_{epoch:03d}.pt'
   if not path.exists():checks.append(dict(model=label,epoch=epoch,exists=False,sha256=None));continue
   ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['completed_epoch']==epoch and ck['cursor']==0 and ck['order'] is None
   digest=sha(path)
   if label=='old' and epoch in [3,5]:assert digest==next(r['sha256'] for r in oldcp if int(r['epoch'])==epoch)
   if label=='new':assert ck['identity']==config['identity']
   assert ck['identity']['dev_bank']==config['identity']['dev_bank'] and ck['identity']['train_bank']==config['identity']['train_bank']
   seed_all(3201);model=build_joint(config['config']);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False).cuda();model.coupling.capture_assignments=True;models[label,epoch]=model
   checks.append(dict(model=label,epoch=epoch,exists=True,sha256=digest,update=ck['update']));del ck
 assert len(models)==6,'Missing checkpoints; see inputs, do not substitute epochs'
 write(out/'CHECKPOINTS.json',checks)
 data=FormalData(split,'scoring',verify=False);rows=[];max_single=0.;max_cache=0.;bank_hashes={}
 for purpose in ['train','selection_dev']:
  folder=up/'zara2/refined_banks'/f'{purpose}_2035';bid=read(folder/'IDENTITY.json');bank=Bank(folder,bid,split)
  assert bid['seed']==2035 and bid['unit']=='m' and bid['K']==20
  assert bank.hash==config['identity']['train_bank' if purpose=='train' else 'dev_bank'];bank_hashes[purpose]=bank.hash
  ids=split[purpose+'_indices'];assert len(ids)==(2681 if purpose=='train' else 590)
  for pos,idx in enumerate(ids):
   assert sha(split['records'][idx]['path'])==split['records'][idx]['sha256']
   b=data.batch([idx],purpose);n=int(b.valid.sum());assert n==b.valid.shape[1];g=bank.batch([idx],b).cuda();x=b.to('cuda');truth=x.future[0,:,-1]
   err=(g[0]-truth[None]).norm(dim=-1);G=float(err.mean(-1).min());oracle=float(err.min(0).values.mean())
   for (label,epoch),model in models.items():
    if purpose=='selection_dev' and label=='old' and epoch in [3,5]:
     I=oldlookup[epoch,idx,'initial_Hungarian'];H=oldlookup[epoch,idx,'H'];max_cache=max(max_cache,abs(G-oldlookup[epoch,idx,'G']),abs(oracle-oldlookup[epoch,idx,'oracle']));origin='4c9eede_archive'
    elif purpose=='selection_dev' and label=='new':
     # immutable completed-epoch output, tied to checkpoint SHA and frozen identity
     cache=trial/'dev'/f'epoch_{epoch:03d}_windows.csv'
     if not hasattr(model,'cached_rows'):
      with cache.open() as f:model.cached_rows={int(r['index']):r for r in csv.DictReader(f)}
      assert set(model.cached_rows)==set(ids)
     r=model.cached_rows[idx];assert int(r['n_agents'])==n and int(r['epoch'])==epoch
     I=float(r['initial_JFDE']);H=float(r['H_JFDE']);max_cache=max(max_cache,abs(G-float(r['G_JFDE'])),abs(oracle-float(r['oracle'])));origin='frozen_trial_epoch_validation'
    else:
     with torch.inference_mode():
      context=model.encode_context(x.observed,x.semantic_maps,x.valid);h,pi,_=model.coupling.assign(context,g,model.bridge.representation.goal_scale,straight_through=False);pi0=model.coupling.last_initial_indices
      assert torch.equal(pi.sort(1).values,torch.arange(20,device='cuda')[None,:,None].expand_as(pi))
      initial=gather_candidates(g,pi0);I=float((initial[0]-truth[None]).norm(dim=-1).mean(-1).min());H=float((h[0]-truth[None]).norm(dim=-1).mean(-1).min());origin='frozen_hard_assignment_forward'
    if n==1:max_single=max(max_single,abs(G-I),abs(G-H))
    rows.append(dict(split=purpose,model=label,epoch=epoch,index=idx,n_agents=n,G_JFDE=G,I_JFDE=I,H_JFDE=H,feedback_gain=I-H,oracle=oracle,source=origin))
   if (pos+1)%100==0:print(purpose,pos+1,len(ids),flush=True)
  csvout(out/'WINDOW_METRICS.csv',rows)
 assert max_single<1e-6 and max_cache<1e-6
 lookup={(r['split'],r['epoch'],r['index'],r['model']):r for r in rows};paired=[];max_identity=0.
 for r in rows:
  if r['model']!='new':continue
  old=lookup[r['split'],r['epoch'],r['index'],'old'];di=r['I_JFDE']-old['I_JFDE'];dh=r['H_JFDE']-old['H_JFDE'];dg=r['feedback_gain']-old['feedback_gain'];max_identity=max(max_identity,abs(dh-(di-dg)))
  assert r['G_JFDE']==old['G_JFDE'] and r['oracle']==old['oracle']
  paired.append(dict(split=r['split'],epoch=r['epoch'],index=r['index'],n_agents=r['n_agents'],delta_initial=di,delta_final=dh,gain_new_minus_old=dg))
 summaries=[]
 for purpose in ['train','selection_dev']:
  for epoch in [1,3,5]:
   for group in ['all','N_ge_9']:
    rr=[r for r in paired if r['split']==purpose and r['epoch']==epoch and (group=='all' or r['n_agents']>=9)]
    s=dict(split=purpose,epoch=epoch,group=group,windows=len(rr))
    for label in ['old','new']:
     rs=[lookup[purpose,epoch,r['index'],label] for r in rr]
     for k in ['G_JFDE','I_JFDE','H_JFDE','feedback_gain','oracle']:s[label+'_'+k]=sum(r[k] for r in rs)/len(rs)
    for k in ['delta_initial','delta_final','gain_new_minus_old']:s[k]=sum(r[k] for r in rr)/len(rr)
    summaries.append(s)
 csvout(out/'PAIRED_WINDOWS.csv',paired);csvout(out/'GROUP_SUMMARY.csv',summaries)
 assert max_identity<1e-6
 for name,r in [('old',up),('new',trial)]:assert inventory(r)==before[name],'source asset changed during diagnostic'
 write(out/'QA.json',dict(status='PASS',single_person_max_difference_m=max_single,cached_vs_bank_max_difference_m=max_cache,decomposition_max_residual_m=max_identity,tolerance_m=1e-6,optimizer_updates=0,augmentation=False,DDIM_calls=0,test_data_read=False,source_assets_unchanged=True))
 write(out/'IDENTITY.json',dict(script_sha256=sha(Path(__file__)),bank_hashes=bank_hashes,split_sha256=d['sha256'],epochs=[1,3,5],window_counts=dict(train=2681,selection_dev=590),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),engineering_threshold_m=.01))
 print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
