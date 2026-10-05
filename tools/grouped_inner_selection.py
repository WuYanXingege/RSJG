#!/usr/bin/env python3
"""Qualified-only full inner interface check; no updates, no outer access.

Invoke under grouped_preflight_supervisor --gpu-seconds to share its cumulative
budget. Unqualified inputs exit before GPU enumeration or model construction.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_grouped_training import *
from tools.grouped_model_smoke import arguments

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=600);a=ap.parse_args();D=a.archive.resolve()
    if not 0<a.seconds<=1200:ap.error('bounded selection seconds required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
    issues=qualification_issues(reg,('train','inner_valid'))
    if issues:
        atomic_json(dict(status='EXTERNAL_EVIDENCE_BLOCKED',issues=issues,gpu_queries=0,
            model_constructs=0,inner_predictions=0,inner_metrics=0,optimizer_updates=0,
            coverage='explicit path exits before GPU enumeration/model constructor; import/native coverage remains UNKNOWN'),D/'INNER_SELECTION_RESULT.json')
        raise SystemExit(2)
    prior_path=D/'GPU_BUDGET_LEDGER.jsonl'
    prior=sum(json.loads(x)['charged_seconds'] for x in prior_path.read_text().splitlines()) if prior_path.exists() else 0.
    if prior+a.seconds+15>1800:raise RuntimeError('cumulative GPU selection budget')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    if q.returncode or q.stdout.strip():raise RuntimeError('RESOURCE_BLOCKED_DEVICE_BUSY_OR_QUERY_FAILED')
    torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();budget=min(10*1024**3,free-1024**3)
    if budget<2*1024**3:raise RuntimeError('RESOURCE_BLOCKED_MEMORY')
    torch.cuda.set_per_process_memory_fraction(budget/total,0);torch.cuda.reset_peak_memory_stats()
    source=json.loads((D/'MODEL_SMOKE_cuda.json').read_text())
    if source['status']!='PASS' or source['data_binding']!=data_binding(reg):raise RuntimeError('smoke input identity')
    results={};deadline=time.monotonic()+a.seconds
    from src.models.goal_pretrain import Goal_Pretrain
    from src.models.model import GDTS
    for kind,cls in [('goal',Goal_Pretrain),('joint',GDTS)]:
        ref=next(x['checkpoint'] for x in source['results'] if x['case']==kind+'_fp32')
        if file_hash(path(ref['path']))!=ref['sha256']:raise RuntimeError('smoke actual checkpoint SHA')
        value=torch.load(path(ref['path']),map_location='cpu',weights_only=True)
        if value['classification']!='SMOKE_ONLY':raise RuntimeError('selection smoke classification')
        seed_all(3101,'cuda:0');args=arguments(kind,'cuda:0',Path(ref['path']).parent)
        model=cls(args,torch.device('cuda:0'),dataset=geometry_dataset(reg)).cuda()
        model.load_state_dict(value['model_state_dict'],strict=True)
        reader=PackReader(reg,'inner_valid');metric=evaluate(model,reader,'cuda:0',kind=='goal',deadline)
        if metric.denominator!=24955:raise RuntimeError('incomplete inner selection')
        results[kind]=dict(value=metric.value(),denominator=metric.denominator,packs=len(reader),
            checkpoint_sha256=ref['sha256'],target_read_after_prediction=True)
        del model;torch.cuda.empty_cache()
    atomic_json(dict(status='PASS',results=results,optimizer_updates=0,outer_access=False,
        peak_reserved=torch.cuda.max_memory_reserved(),scope='SMOKE_ONLY engineering selection, not a performance result or parent'),D/'INNER_SELECTION_RESULT.json')
if __name__=='__main__':main()
