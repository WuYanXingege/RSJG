#!/usr/bin/env python3
"""Bind the completed selected goal to a new, independent fresh-joint protocol."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_protocol import ROOT,Registry,file_hash,data_binding,sealed
from src.p2_grouped_training import qualify_parent
from src.p2_observation import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--goal-archive',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();D=a.archive.resolve();G=a.goal_archive.resolve();parent=a.parent.resolve()
    D.mkdir(parents=True,exist_ok=False)
    source=json.loads((G/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());old=Registry(source,source['manifest_hash'])
    value=qualify_parent(dict(path=str(parent),sha256=a.sha256),old,'goal')
    parent_ref=dict(path=str(parent),sha256=a.sha256,artifact_role='p2_goal',classification='QUALIFIED',
        data_binding=data_binding(old),initial_state_sha256=value['initial_state_sha256'],state_sha256=value['state_sha256'],
        selected_epoch=value['progress']['epoch_completed'],selection_metric=value['metric'])
    manifest=copy.deepcopy(source);manifest['parents']['goal']=parent_ref;manifest=sealed(manifest)
    atomic_json(manifest,D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json');reg=Registry(manifest,manifest['manifest_hash'])
    assert data_binding(reg)==data_binding(old);qualify_parent(manifest['parents']['goal'],reg,'goal')
    cfg=json.loads((G/'FRESH_GOAL_CONFIG.yaml').read_text())
    run='grouped_fresh_joint_official_prior_seed3101_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    cfg.update(p2_manifest=str((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').relative_to(ROOT)),p2_manifest_hash=manifest['manifest_hash'],
        p2_mode='joint',phase='train',num_epochs=250,learning_rate=.0001,run_name=run,
        goal_pretrain_checkpoint=str(parent),pretrain_path=None,load_checkpoint=None)
    atomic_json(cfg,D/'FRESH_JOINT_CONFIG.yaml')
    atomic_json(dict(status='REGISTERED_NOT_READY',stage='fresh_joint',run_name=run,manifest_hash=manifest['manifest_hash'],
        data_binding=data_binding(reg),goal_parent=parent_ref,goal_archive=str(G.relative_to(ROOT)),
        formal_authority=dict(seed=3101,precision='fp32',epochs_max=250,updates_max=43750,wall_seconds=172800,
            no_auto_cache=True,no_auto_A0=True,no_auto_A1=True),formal_training_started=False),D/'REGISTRATION.json')
    print(json.dumps(dict(status='REGISTERED_NOT_READY',run_name=run,manifest_hash=manifest['manifest_hash'],data_binding=data_binding(reg))))
if __name__=='__main__':main()
