"""Apply only measured execution choices before the first formal update."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_joint.common import read,write,canonical_hash,digest
from run_bcep_jmm import sources

def main(path):
 plan=read(path);a=Path(plan['assets']);gate=read(a/'receipts/GPU_PREFLIGHT.json')
 assert not (a/'identity/EXECUTION_IDENTITY.json').exists(),'formal identity already frozen'
 assert gate['status']=='PASS' and gate['source_hash']==canonical_hash(sources())
 for row in plan['folds'].values():
  p=read(row['provider_config']);c=read(row['joint_config']);p['batched_horizons']=gate['recommendations']['provider']['adopt'];c['grouped_training']=gate['recommendations']['joint']['adopt']
  write(row['provider_config'],p);write(row['joint_config'],c)
  row['provider_config_sha256']=digest(row['provider_config']);row['joint_config_sha256']=digest(row['joint_config'])
 plan['runtime_selection_receipt']=dict(path=str(a/'receipts/GPU_PREFLIGHT.json'),sha256=digest(a/'receipts/GPU_PREFLIGHT.json'));write(path,plan)
 write(Path(plan['docs'])/'RESOLVED_PLAN.json',plan);write(a/'receipts/RUNTIME_FROZEN.json',dict(status='PASS',plan_sha256=digest(path),source_hash=gate['source_hash'],recommendations=gate['recommendations']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)
