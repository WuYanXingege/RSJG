"""Readiness only. Every executable installs a zero-step guard first."""
import csv,hashlib,inspect,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE))
from bcep_scene_energy.formal_contract import paths,BASE,model_hash,file_sha
from bcep_joint.common import read,write,seed_all
from bcep.contracts import canonical_hash
ROOTS=paths();OUT=ROOTS['audit'];PUBLIC=SOURCE/'docs/research/bcep_scene_energy/stage2_ready_20261121'
GUARD={'optimizer_step_calls':0}


def setup(cuda=False):
    def denied(*a,**kw):GUARD['optimizer_step_calls']+=1;raise PermissionError('all real/synthetic optimizer.step forbidden in readiness')
    for c in vars(torch.optim).values():
        if inspect.isclass(c) and issubclass(c,torch.optim.Optimizer):c.step=denied
    torch.set_num_threads(1)
    if cuda:
        from bcep_v2.runtime import configure_cuda
        configure_cuda()
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def protected():
    names=['bcep_ebjd_v2_integrated','zara2_bcep_v3_stage2_20261120','zara2_costproxy_stage2_20261119',
        'zara2_sharedworld_stage2_onerun_20261119','zara2_v3_frozen_feedback_audit_20261121',
        'joint_score_identifiability_20261121','ssec_network_acceptance_20261121','ssec_candidate_identity_20261121']
    result={}
    for name in names:
        folder=OUT.parent/name;assert folder.exists()
        for p in folder.rglob('*'):
            if p.is_file():result[name+'/'+str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,sha256=digest(p))
    return result


def csvwrite(path,rows):
    from bcep_scene_energy.formal_validation import write_csv
    write_csv(path,rows)
