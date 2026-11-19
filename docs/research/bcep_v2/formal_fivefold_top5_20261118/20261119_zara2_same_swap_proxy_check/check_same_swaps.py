"""Frozen same-swap audit. CPU FP64 only; no model forward, optimizer, or DDIM.
Run with --workspace <local RSJG root> --archive-repo <local git checkout> --output <dir>.
Raw weights, banks and asset paths are never copied into the public outputs.
"""
import argparse,ast,csv,hashlib,io,json,math,os,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch

COMMIT='4c9eede551bebdd7634ab81b7b4bd790165c6bb0'
SOURCE='468c5eddde48bdeec47c4bb3c39418352be16bb9'
ARCHIVE='docs/research/bcep_v2/formal_fivefold_top5_20261118/20261119_zara2_joint_endpoint_dev/'
TAU=.1; RECOMMEND_EPS=1e-6; NUMERIC_TOL=1e-10; FD_TOL=1e-8

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def dump(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvout(p,rows):
 assert rows
 tmp=p.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:("NA" if v is None else v) for k,v in row.items()} for row in rows)
 tmp.replace(p)
def cost(e):return -TAU*(torch.logsumexp(-e/TAU,0)-math.log(len(e)))
def action(s):return 'adopt' if s < -RECOMMEND_EPS else 'reject' if s > RECOMMEND_EPS else 'abstain'
def ratio(a,b):return a/b if b else None

def aggregate(rows,epoch,subgroup,threshold,window='all'):
 included=[r for r in rows if abs(r['true_delta_m'])>threshold]
 base=dict(epoch=epoch,subgroup=subgroup,truth_threshold_m=threshold,window=window,windows_total=len({r['index'] for r in rows}),swaps_total=len(rows),windows_eligible=len({r['index'] for r in included}),truth_eligible=len(included),truth_excluded_at_threshold=len(rows)-len(included))
 benefits=[r for r in included if r['true_delta_m']<0];harms=[r for r in included if r['true_delta_m']>0]
 total_gain=sum(-r['true_delta_m'] for r in benefits)
 summaries=[];conf=[]
 for proxy in ['coord','cost']:
  c=Counter(('benefit' if r['true_delta_m']<0 else 'harm',r[proxy+'_action']) for r in included)
  adopted=[r for r in included if r[proxy+'_action']=='adopt'];gain=sum(-r['true_delta_m'] for r in benefits if r[proxy+'_action']=='adopt')
  summaries.append(dict(**base,proxy=proxy,actual_benefit=len(benefits),actual_harm=len(harms),adopted=len(adopted),rejected=sum(r[proxy+'_action']=='reject' for r in included),abstained=sum(r[proxy+'_action']=='abstain' for r in included),harm_adopted=c['harm','adopt'],harm_suggestion_denominator=len(adopted),harm_suggestion_rate=ratio(c['harm','adopt'],len(adopted)),benefit_adopted=c['benefit','adopt'],benefit_recall_denominator=len(benefits),benefit_recall=ratio(c['benefit','adopt'],len(benefits)),captured_benefit_sum_m=gain,total_benefit_sum_m=total_gain,benefit_gain_capture=ratio(gain,total_gain)))
  for truth in ['benefit','harm']:
   conf.append(dict(**base,proxy=proxy,truth=truth,adopt=c[truth,'adopt'],reject=c[truth,'reject'],abstain=c[truth,'abstain']))
 oldgood=[r for r in benefits if r['coord_action']=='adopt'];retained=[r for r in oldgood if r['cost_action']=='adopt'];lost=[r for r in oldgood if r['cost_action']!='adopt']
 oldbad=[r for r in harms if r['coord_action']=='adopt'];newbad=[r for r in harms if r['cost_action']=='adopt']
 transition=dict(**base,old_harm_adopted=len(oldbad),new_harm_adopted=len(newbad),harm_adoptions_removed=sum(r['cost_action']!='adopt' for r in oldbad),harm_removed_to_reject=sum(r['cost_action']=='reject' for r in oldbad),harm_removed_to_abstain=sum(r['cost_action']=='abstain' for r in oldbad),harm_adoptions_introduced=sum(r['coord_action']!='adopt' for r in newbad),old_benefit_adopted=len(oldgood),old_benefit_retained=len(retained),old_benefit_retention=ratio(len(retained),len(oldgood)),old_benefit_lost=len(lost),old_benefit_lost_to_reject=sum(r['cost_action']=='reject' for r in lost),old_benefit_lost_to_abstain=sum(r['cost_action']=='abstain' for r in lost),benefit_adoptions_added=sum(r['coord_action']!='adopt' and r['cost_action']=='adopt' for r in benefits),old_benefit_gain_sum_m=sum(-r['true_delta_m'] for r in oldgood),retained_old_benefit_gain_m=sum(-r['true_delta_m'] for r in retained),retained_old_benefit_gain_ratio=ratio(sum(-r['true_delta_m'] for r in retained),sum(-r['true_delta_m'] for r in oldgood)))
 matrix=[]
 for truth in ['benefit','harm']:
  for old in ['adopt','reject','abstain']:
   for new in ['adopt','reject','abstain']:
    matrix.append(dict(**base,truth=truth,coord_action=old,cost_action=new,count=sum(('benefit' if r['true_delta_m']<0 else 'harm')==truth and r['coord_action']==old and r['cost_action']==new for r in included)))
 return summaries,conf,transition,matrix

def main(a):
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
 root=a.workspace/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated';src=a.workspace/'RSJG_BCEP_integrated_v2_20261118'
 assert read(root/'STATUS.json')['status']=='PAUSED_RESUMABLE'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==SOURCE
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=src,text=True)
 # Read the immutable git objects, not potentially changed working-tree copies.
 def gitbytes(name):return subprocess.check_output(['git','show',COMMIT+':'+ARCHIVE+name],cwd=a.archive_repo)
 hashes=json.loads(gitbytes('ARTIFACT_SHA256.json'));used={}
 def archived(name):
  b=gitbytes(name);h=hashlib.sha256(b).hexdigest();assert h==hashes[name],name;used[name]=h;return b
 original_script=archived('diagnose_gpu.py').decode();identity=json.loads(archived('IDENTITY.json'))
 tree=ast.parse(original_script);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='softcost')
 namespace={'torch':torch,'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<archived softcost>','exec'),namespace)
 old_softcost=namespace['softcost']
 old=list(csv.DictReader(io.StringIO(archived('SWAPS.csv').decode())));old_summary=list(csv.DictReader(io.StringIO(archived('GRADIENT_SUMMARY.csv').decode())))
 cps=list(csv.DictReader(io.StringIO(archived('CHECKPOINTS.csv').decode())))
 assert len(old)==3840 and Counter(int(r['epoch']) for r in old)=={3:1280,5:1280,26:1280}
 windows=sorted({int(r['index']) for r in old});assert len(windows)==32
 for ep in [3,5,26]:assert Counter(int(r['index']) for r in old if int(r['epoch'])==ep)=={i:40 for i in windows}
 assert len({tuple(r[k] for k in ['epoch','index','agent_id','world_u','world_v','candidate_u','candidate_v']) for r in old})==3840
 run=read(root/'FORMAL_RUN.json');plan=read(Path(run['input_plan']));descriptor=plan['folds']['zara2']['split'];assert sha(descriptor['path'])==descriptor['sha256']
 split=read(descriptor['path']);assert split['unit']=='m' and split['pred_length']==12 and set(windows)<=set(split['selection_dev_indices'])
 # Use exactly the canonical identity implementation from the source-bound checkout.
 import sys
 sys.path.insert(0,str(src));from bcep.contracts import canonical_hash
 bank=root/'zara2/refined_banks/selection_dev_2035';bid=read(bank/'IDENTITY.json');done=read(bank/'COMPLETE.json')
 assert bid==identity['bank_identity'] and canonical_hash(done)==identity['bank_hash']
 assert bid['split']==canonical_hash(split) and bid['purpose']=='selection_dev' and bid['seed']==2035 and bid['unit']=='m'
 assert done['identity_hash']==canonical_hash(bid)
 cpchecks=[]
 for row in cps:
  ep=int(row['epoch']);p=root/f'zara2/joint/epoch_{ep:03d}.pt';assert p.is_file() and sha(p)==row['sha256']
  ck=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['completed_epoch']==ep and ck['update']==int(row['update']) and ck['cursor']==0 and ck['order'] is None
  assert ck['identity']['dev_bank']==identity['bank_hash'] and all(ck['identity'][k]==v for k,v in identity['source'].items())
  cpchecks.append(dict(epoch=ep,sha256=row['sha256'],completed_epoch=ck['completed_epoch'],update=ck['update'],model_tensor_count=len(ck['model']),resumable_fields_present=all(k in ck for k in ['model','optimizer','scheduler','rng','stale'])))
  del ck
 cases={};input_rows=[];maxima=defaultdict(float);checks=[]
 for idx in windows:
  m=split['members'][idx];row=next(r for r in done['rows'] if r['index']==idx)
  assert row==read(bank/f'{idx}.json') and row['member']==m and row['identity_hash']==done['identity_hash'] and sha(bank/f'{idx}.npy')==row['sha256']
  record=split['records'][idx];assert sha(record['path'])==record['sha256']
  raw=np.load(bank/f'{idx}.npy',allow_pickle=False)
  with np.load(record['path'],allow_pickle=False) as z:truth=z['future'][:,-1].copy()
  n=len(m['agent_ids']);assert raw.shape==(20,n,2) and truth.shape==(n,2) and raw.dtype==np.float32 and truth.dtype==np.float32
  g=torch.from_numpy(raw).double();y=torch.from_numpy(truth).double();D=(g-y[None]).norm(dim=-1)
  assert torch.isfinite(D).all() and n>1
  input_rows.append(dict(index=idx,n_agents=n,bank_row_sha256=row['sha256'],data_record_sha256=record['sha256'],ordered_member_sha256=canonical_hash(m)))
  for ep in [3,5,26]:
   with np.load(io.BytesIO(archived(f'assignments/{ep}/{idx}.npz')),allow_pickle=False) as z:pi=torch.from_numpy(z['pi_H'][0].copy()).long();assert int(z['index'])==idx
   assert pi.shape==(20,n) and torch.equal(pi.sort(dim=0).values,torch.arange(20)[:,None].expand(20,n))
   X=torch.gather(g,0,pi[:,:,None].expand(-1,-1,2));eD=torch.gather(D,0,pi).mean(1);eX=(X-y[None]).norm(dim=-1).mean(1)
   harddiff=float(abs(eD.min()-eX.min()));softdiff=float(abs(cost(eD)-old_softcost(X,y)));worlddiff=float((eD-eX).abs().max())
   assert max(harddiff,softdiff,worlddiff)<=NUMERIC_TOL
   zz=X.detach().clone().requires_grad_(True);grad,=torch.autograd.grad(old_softcost(zz,y),zz)
   cases[ep,idx]=(g,y,D,pi,X,eD,torch.softmax(-eD/TAU,0),grad,m)
   checks.append(dict(epoch=ep,index=idx,n_agents=n,hard_JFDE_m=float(eD.min()),normalized_softmin_m=float(cost(eD)),world_error_difference=worlddiff,hard_JFDE_difference=harddiff,softmin_difference=softdiff))
 result=[]
 for rowid,r in enumerate(old):
  ep,idx,i,u,v,aa,bb=(int(r[k]) for k in ['epoch','index','agent_position','world_u','world_v','candidate_u','candidate_v'])
  g,y,D,pi,X,e,w,grad,m=cases[ep,idx];n=len(y)
  assert str(m['agent_ids'][i])==r['agent_id'] and int(pi[u,i])==aa and int(pi[v,i])==bb and u!=v and aa!=bb
  delta=(D[bb,i]-D[aa,i])/n;de=torch.zeros_like(e);de[u]=delta;de[v]=-delta
  true=float(cost(e+de)-cost(e));s_cost=float(torch.dot(w,de))
  Xswap=X.clone();Xswap[[u,v],i]=Xswap[[v,u],i]
  s_coord=float((grad*(Xswap-X)).sum());coordinate_true=float(old_softcost(Xswap,y)-old_softcost(X,y))
  eps=1e-5/max(1.,float(de.abs().max()));fd=float((cost(e+eps*de)-cost(e-eps*de))/(2*eps))
  # Independent P-space derivative at the exact hard permutation.
  P=torch.nn.functional.one_hot(pi.T,20).double().requires_grad_(True)
  eP=torch.einsum('nkj,jn->k',P,D)/n;pg,=torch.autograd.grad(cost(eP),P)
  pdir=float(pg[i,u,bb]-pg[i,u,aa]+pg[i,v,aa]-pg[i,v,bb])
  errors=dict(old_true_abs_error=abs(true-float(r['true_delta_m'])),old_coord_abs_error=abs(s_coord-float(r['coordinate_proxy_direction_m'])),lookup_vs_coordinate_true_abs_error=abs(true-coordinate_true),cost_fd_abs_error=abs(s_cost-fd),cost_P_autograd_abs_error=abs(s_cost-pdir))
  for k,val in errors.items():maxima[k]=max(maxima[k],val)
  assert errors['cost_fd_abs_error']<=FD_TOL and max(v for k,v in errors.items() if k!='cost_fd_abs_error')<=NUMERIC_TOL
  violation=true-s_cost;assert violation<=NUMERIC_TOL
  result.append(dict(sample_row=rowid,epoch=ep,index=idx,n_agents=n,agent_position=i,agent_id=r['agent_id'],world_u=u,world_v=v,candidate_u=aa,candidate_v=bb,true_delta_m=true,s_coord_m=s_coord,s_cost_m=s_cost,coord_action=action(s_coord),cost_action=action(s_cost),true_category='benefit' if true < -RECOMMEND_EPS else 'harm' if true>RECOMMEND_EPS else 'near_zero',large_true_change=abs(true)>.01,high_people=n>=9,candidate_error_delta_per_person_m=float(delta),weight_u=float(w[u]),weight_v=float(w[v]),cost_central_difference_m=fd,cost_P_direction_m=pdir,concavity_signed_violation_m=violation,**errors))
 assert len(result)==3840
 legacy=[]
 for ep in [3,5,26]:
  rr=[r for r in result if r['epoch']==ep and abs(r['true_delta_m'])>1e-6 and abs(r['s_coord_m'])>1e-6]
  disagree=sum(r['true_delta_m']*r['s_coord_m']<0 for r in rr);expected={3:(387,1083),5:(393,1067),26:(431,1093)}[ep];assert (disagree,len(rr))==expected
  archived_row=next(r for r in old_summary if int(r['epoch'])==ep and float(r['threshold_m'])==1e-6);assert int(archived_row['disagreements'])==disagree and int(archived_row['informative'])==len(rr)
  harm=sum(r['true_delta_m']>0 and r['coord_action']=='adopt' for r in rr);benefit_reject=sum(r['true_delta_m']<0 and r['coord_action']=='reject' for r in rr)
  if ep==26:assert (harm,benefit_reject)==(391,40)
  legacy.append(dict(epoch=ep,disagreements=disagree,denominator=len(rr),harm_adopted=harm,benefit_rejected=benefit_reject,matched_archive=True))
 csvout(out/'PAIRED_SWAPS.csv',result);csvout(out/'ANCHOR_CHECKS.csv',checks);csvout(out/'LEGACY_REPRODUCTION.csv',legacy)
 stats=[];conf=[];transitions=[];matrices=[];window_stats=[];window_trans=[]
 for ep in [26,3,5]:
  for subgroup in ['all','N_ge_9']:
   rr=[r for r in result if r['epoch']==ep and (subgroup=='all' or r['n_agents']>=9)]
   for threshold in [1e-6,.01]:
    st,co,tr,ma=aggregate(rr,ep,subgroup,threshold);stats+=st;conf+=co;transitions.append(tr);matrices+=ma
  for idx in windows:
   rr=[r for r in result if r['epoch']==ep and r['index']==idx]
   for threshold in [1e-6,.01]:
    st,co,tr,ma=aggregate(rr,ep,'window',threshold,idx);window_stats += [dict(x,n_agents=rr[0]['n_agents']) for x in st];window_trans.append(dict(tr,n_agents=rr[0]['n_agents']))
 for name,rr in [('SUMMARY.csv',stats),('CONFUSION_2x3.csv',conf),('TRANSITIONS.csv',transitions),('TRANSITION_MATRIX.csv',matrices),('WINDOW_SUMMARY.csv',window_stats),('WINDOW_TRANSITIONS.csv',window_trans)]:csvout(out/name,rr)
 missed=sorted([r for r in result if r['true_delta_m']< -1e-6 and r['coord_action']=='adopt' and r['cost_action']!='adopt'],key=lambda r:(r['epoch']!=26,-abs(r['true_delta_m']),r['sample_row']))
 csvout(out/'MISSED_BENEFICIAL_SWAPS.csv',missed)
 # Read previously archived endpoint records only; no trajectory inference.
 previous=list(csv.DictReader(io.StringIO(archived('WINDOW_METRICS.csv').decode())))
 deterioration=[]
 for group in ['all_590','all_590_N_ge_9','sample_32','sample_32_N_ge_9']:
  selected=[r for r in previous if r['method']=='H' and (not group.startswith('sample') or int(r['index']) in windows) and (not group.endswith('N_ge_9') or int(r['n_agents'])>=9)]
  for ep in [3,5,26]:
   rr=[r for r in selected if int(r['epoch'])==ep]
   deterioration.append(dict(group=group,epoch=ep,windows=len(rr),archived_H_JFDE_m=float(np.mean([float(r['JFDE_m']) for r in rr]))))
 csvout(out/'ARCHIVED_ENDPOINT_CONTEXT.csv',deterioration)
 recovery=root/'zara1/head/last.pt';saved=torch.load(recovery,map_location='cpu',weights_only=False)
 qa=dict(status='PASS',numeric_tolerance_m=NUMERIC_TOL,central_difference_tolerance_m=FD_TOL,statistics_abstention_threshold_m=RECOMMEND_EPS,maximum_errors=dict(maxima),anchor_count=len(checks),swaps=len(result),concavity_max_signed_violation_m=max(r['concavity_signed_violation_m'] for r in result),concavity_max_positive_violation_m=max(0.,max(r['concavity_signed_violation_m'] for r in result)),concavity_violations=sum(r['concavity_signed_violation_m']>NUMERIC_TOL for r in result),cost_gradient_P_checks=len(result),cost_central_difference_checks=len(result),legacy_reproduction=legacy,optimizer_updates=0,DDIM_calls=0,model_forward_calls=0,test_data_records_read=0)
 identityout=dict(source_commit=SOURCE,archive_commit=COMMIT,archive_directory=ARCHIVE,archive_input_hashes=used,script_sha256=sha(Path(__file__)),checkpoint_checks=cpchecks,bank_identity_sha256=canonical_hash(bid),bank_complete_sha256=identity['bank_hash'],split_sha256=descriptor['sha256'],window_inputs=input_rows,windows=windows,total_rows=3840,rows_per_epoch=1280,unit='m',K=20,temperature_m=TAU,precision='FP64',torch_version=str(torch.__version__),numpy_version=str(np.__version__),raw_weights_and_bank_published=False)
 dump(out/'INPUT_IDENTITY.json',identityout);dump(out/'QA.json',qa)
 dump(out/'RUN_STATUS.json',dict(status=read(root/'STATUS.json')['status'],experiment=run['experiment'],fold='zara1',stage='head',recovery_sha256=sha(recovery),saved_update=saved['update'],completed_epoch=saved['completed_epoch'],epoch=saved['epoch'],cursor=saved['cursor'],optimizer_updates=0,formal_source_modified=False,queue_resume_invoked=False,DDIM_calls=0))
 dump(out/'RESULT.json',dict(status='COMPLETE',summary=stats,transitions=transitions,qa=qa,archived_endpoint_context=deterioration))
 print(json.dumps(dict(qa=qa,epoch26=[r for r in stats if r['epoch']==26],transitions26=[r for r in transitions if r['epoch']==26]),ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
