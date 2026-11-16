"""Read-only six-layer restoration check; never constructs an optimizer."""
import argparse, csv, json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ebjd.config import build_model
from ebjd.unified import UnifiedCouplingBridge, gather_candidates
from ebjd.proposals import file_sha256
from ebjd.sampling import differentiable_sample
from run_zara1_unified_coupling_bridge import make_bank, evaluate, loader

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--full-eth',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.output=a.output.resolve()
    os.chdir(Path(__file__).resolve().parents[1])
    torch.set_num_threads(2);torch.manual_seed(2035)
    source=json.loads((Path(__file__).resolve().parents[1]/'docs/results/ebjd_eth_ucy_unified_fivefold_20261101/SOURCE_MANIFEST.json').read_text())
    result={'atol':2e-5,'rtol':2e-5,'optimizer_updates':0,'device':a.device,'folds':{}}
    a.output.mkdir(parents=True,exist_ok=True)
    if a.resume: result=json.loads((a.output/'BASELINE_REPLAY.json').read_text())
    with torch.no_grad():
      for fold,s in source['fold_sources'].items():
        if fold in result['folds']:continue
        cfg=json.loads(Path(s['resolved_config']['path']).read_text());root=Path(s['experiment_root'])
        ck=Path(s['selected_checkpoint']['path']);assert file_sha256(ck)==s['selected_checkpoint']['sha256']
        model=UnifiedCouplingBridge(build_model(cfg,proposal_conditioned=True))
        payload=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True);model.to(a.device).eval()
        rm=json.loads((root/'RUN_MANIFEST.json').read_text());cmd=rm.get('command',[])
        if '--validation-bank-pattern' in cmd: bankpath=Path(cmd[cmd.index('--validation-bank-pattern')+1].format(seed=2035))
        else: bankpath=root/'proposal_banks/validation_seed2035.npz'
        bank=make_bank(str(bankpath),cfg['paths']['matched_gdts_validation_checkpoint_sha256'],s['manifests']['validation']['sha256'],'validation')
        counts=[];seen=set()
        for cpu in loader(cfg,'validation',torch.Generator().manual_seed(99173)):
          n=int(cpu.valid.sum());group='singleton' if n==1 else 'multi'
          if group in seen:continue
          seen.add(group);b=cpu.to(a.device);g=bank.batch(cpu).to(a.device);ctx=model.encode_context(b.observed,b.semantic_maps,b.valid)
          h,idx,_=model.coupling.assign(ctx,g,model.bridge.representation.goal_scale,straight_through=False)
          assert torch.equal(h,gather_candidates(g,idx))
          for person in range(n):assert sorted(idx[0,:,person].tolist())==list(range(20))
          z=torch.randn(1,20,b.observed.shape[1],12,2,device=a.device)
          path,goal,_=differentiable_sample(model.bridge,ctx,z,20,proposal_goals=h,proposal_eta=0.)
          assert torch.isfinite(path).all() and torch.equal(goal,h)
          counts.append({'scene':cpu.scene_ids,'agents':n,'permutation_multiset_exact':True,'endpoint_exact':True})
          if len(seen)==2:break
        r={'strict_load':True,'epoch':payload['epoch'],'smoke':counts,'checkpoint_sha256':file_sha256(ck),'bank_path':str(bankpath),'bank_sha256':bank.sha256}
        if fold=='eth' and a.full_eth:
          measured=evaluate(model,cfg,bank,torch.device(a.device),2035)
          ref=next(x for x in csv.DictReader(open(s['raw_G_H_seed_results']['path'])) if x['seed']=='2035')
          diff={f'{method}_{key}': measured[method][key]-float(ref[f'{method}_{key}']) for method in ['G','H'] for key in ['minADE','minFDE','JADE','JFDE']}
          r.update(replay=measured,differences=diff,replay_pass=all(abs(v)<=2e-5+2e-5*abs(float(ref[k])) for k,v in diff.items()))
        result['folds'][fold]=r
        (a.output/'BASELINE_REPLAY.json').write_text(json.dumps(result,indent=2));print(fold,r,flush=True)
        del model,payload
    result['status']='PASS' if all(x.get('replay_pass',True) for x in result['folds'].values()) else 'BLOCKED_REPLAY'
    (a.output/'BASELINE_REPLAY.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
