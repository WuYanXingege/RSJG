"""Guarded future pilot. This package defaults to NO real training."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ebjd.data import SceneManifestDataset
from bcep.provider import BCEP

def atomic_save(value,path):
 temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
def identity(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text())
 if cfg.get('allow_real_training') is not True:raise SystemExit('REAL_TRAINING_DISABLED: prototype requires a separately authorized frozen pilot')
 workspace=Path(__file__).resolve().parents[1]
 split_path=(workspace/cfg['split_lock']).resolve();split=json.loads(split_path.read_text())
 assert identity(split['source_manifest'])==split['source_sha256'];assert split['scale_source']=='internal_train_only'
 ds=SceneManifestDataset(split['source_manifest']);device=torch.device(cfg['device']);torch.set_num_threads(2)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
 model=BCEP(split['pred_length'],split['dt']).to(device);model.scales.copy_(torch.tensor(split['scales'],device=device));model.scales_fitted.fill_(True)
 opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=cfg['lr_gamma']);gen=torch.Generator().manual_seed(cfg['seed'])
 output=(workspace/cfg['output']).resolve();output.mkdir(parents=True,exist_ok=True);lock={'config':identity(a.config),'split':identity(split_path),'source':{str(p.relative_to(Path(__file__).resolve().parents[1])):identity(p) for p in (Path(__file__).resolve().parents[1]/'bcep').rglob('*.py')},'entrypoint':identity(__file__),'weights_origin':'random_initialization','teacher_weights':False}
 start=0;update=0;best=float('inf');stale=0;elapsed=0.
 last=output/'last.pt'
 if a.resume:
  c=torch.load(last,map_location='cpu',weights_only=False);assert c['identity']==lock
  model.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler']);start=c['epoch'];update=c['update'];best=c['best'];stale=c['stale'];elapsed=c['elapsed']
  random.setstate(c['rng']['python']);np.random.set_state(c['rng']['numpy']);torch.set_rng_state(c['rng']['cpu']);gen.set_state(c['rng']['loader'])
  if device.type=='cuda':torch.cuda.set_rng_state_all(c['rng']['cuda'])
 elif last.exists():raise RuntimeError('existing last checkpoint requires --resume')
 t0=time.monotonic()
 def one(i,train):
  s=ds[i];obs=s['observed'].to(device);maps=s['semantic_maps'].to(device);future=s['future'].to(device);valid=torch.ones(len(obs),dtype=torch.bool,device=device)
  loss=model.loss(obs,maps,valid,future,cfg['lambda_aux']);assert torch.isfinite(loss)
  if train:
   opt.zero_grad(set_to_none=True);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
  return float(loss.detach())
 for epoch in range(start+1,cfg['max_epochs']+1):
  model.train();losses=[]
  for j in torch.randperm(len(split['train_indices']),generator=gen).tolist():losses.append(one(split['train_indices'][j],True));update+=1
  model.eval()
  with torch.no_grad():dev=np.mean([one(i,False) for i in split['dev_indices']])
  improved=dev<best;best=min(best,dev);stale=0 if improved else stale+1;scheduler.step()
  c={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'update':update,'best':best,'stale':stale,'identity':lock,'elapsed':elapsed+time.monotonic()-t0,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'loader':gen.get_state()}}
  atomic_save(c,last)
  if improved:atomic_save(c,output/'best.pt')
  with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'update':update,'train_nll':np.mean(losses),'dev_nll':dev,'elapsed':c['elapsed']})+'\n')
  print(f'Epoch {epoch} trainNLL={np.mean(losses):.6f} devNLL={dev:.6f}',flush=True)
  if stale>=cfg['patience'] or c['elapsed']>=cfg['budget_seconds']:break
if __name__=='__main__':main()
