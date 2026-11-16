"""Two explicitly bounded CPU stages; never constructs an optimizer."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure, ROOT
import torch
import yaml
from bcep.contracts import load_locked_split, validate_checkpoint as validate_flow, digest
from bcep.provider import BCEP
from bcep.pilot_runner import atomic_json
from ebjd.data import SceneManifestDataset
from bcep_control.mdn import MDNProvider
from bcep_control.contracts import validate_checkpoint as validate_mdn
from bcep_control.diagnostics import export_frozen
from bcep_control.budgets import readonly_budget


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--stage',choices=['flow-diagnostics','mdn-evaluation'],required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['assets']);configure()
    with readonly_budget(root,a.stage):
        cfg=yaml.safe_load(Path(plan['config']).read_text())
        contract_path=Path(plan['old_doc'])/'RUN_DATA_AND_EVAL_CONTRACT.json' if a.stage=='flow-diagnostics' else ROOT/cfg['contract']
        split_path=ROOT/cfg['split_lock'];split,contract,_=load_locked_split(split_path,contract_path)
        dataset=SceneManifestDataset(split['source_manifest'])
        if a.stage=='flow-diagnostics':
            checkpoint=Path(plan['frozen_bcep_best']['path']);assert digest(checkpoint)==plan['frozen_bcep_best']['sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_flow(state,contract_path,split_path)
            assert state['completed_epoch']==6 and state['update']==22740
            model=BCEP(split['pred_length'],split['dt']);kind='bcep_flow';output=root/'flow_diagnostics'
            old_banks=Path(plan['old_assets'])/'evaluation/banks'
        else:
            checkpoint=root/'mdn_training/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            validate_mdn(state,contract_path,split_path)
            model=MDNProvider(split['pred_length'],split['dt']);kind='mdn8';output=root/'mdn_evaluation';old_banks=None
        model.load_state_dict(state['model'],strict=True)
        summary=export_frozen(model,dataset,split,contract,checkpoint,output,kind,old_banks)
        if a.stage=='flow-diagnostics':
            atomic_json(dict(status='PASS',checkpoint_sha256=digest(checkpoint),epoch=6,updates=22740,
                             regression=summary['regression'],parameter_buffer_hash_unchanged=summary['state_hash_before']==summary['state_hash_after'],
                             optimizer_updates=0),root/'reports/FROZEN_BCEP_REGRESSION.json')


if __name__=='__main__':main()
