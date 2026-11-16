"""Guarded future pilot: real training requires separate explicit authorization."""
import argparse,sys,time
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ebjd.data import SceneManifestDataset
from bcep.contracts import WORKSPACE,load_locked_split,checkpoint_identity,require
from bcep.pilot_runner import run_pilot

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    cfg=yaml.safe_load(a.config.read_text())
    if cfg.get('allow_real_training') is not True:raise SystemExit('REAL_TRAINING_DISABLED: separately authorized frozen pilot required')
    started=time.monotonic()
    split_path=(WORKSPACE/cfg['split_lock']).resolve();contract_path=(WORKSPACE/cfg['contract']).resolve()
    split,contract,_=load_locked_split(split_path,contract_path)
    require(cfg['max_updates']==len(split['train_indices'])*cfg['max_epochs'],'frozen update cap differs from membership')
    torch.set_num_threads(2)
    return run_pilot(SceneManifestDataset(split['source_manifest']),split,cfg,checkpoint_identity(contract_path,split_path),
                     WORKSPACE/cfg['output'],resume=a.resume,workflow_started=started)
if __name__=='__main__':main()
