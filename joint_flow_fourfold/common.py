"""Fixed experiment constants and small atomic/provenance helpers."""
from pathlib import Path
import subprocess
import time

from bcep.contracts import digest
from bcep_joint.common import read, write

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
RUN = ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
DOC = SOURCE/'docs/research/continuous_joint_flow/fourfold_confirmation_20261122'
BASE_COMMIT = '171bcff3b6b2dbb6aad0becb04e8d7c086c8ffe3'
BRANCH = 'experiment/ethucy-joint-flow-fourfold-confirmation-20261122'
FOLDS = ('zara1', 'hotel', 'univ', 'eth')
SEEDS = tuple(range(2035, 2040))
PURPOSE = 'heldout_confirmation'


def immutable(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'immutable identity changed: '+str(path)
    else:
        write(path, value)


def source_hashes():
    packages = ('joint_flow_fourfold', 'joint_flow_dependency', 'joint_flow_frozen_test',
                'scene_pair_selector', 'bcep_v2', 'bcep_joint', 'bcep_formal', 'bcep',
                'ebjd', 'selector_control', 'bcep_confirmation')
    return {str(p.relative_to(SOURCE)): digest(p) for folder in packages
            for p in sorted((SOURCE/folder).rglob('*.py'))}


def status(folder, value, **details):
    write(Path(folder)/'STATUS.json', dict(status=value, unix=time.time(), **details))
    print(dict(status=value, **details), flush=True)


def wait_gpu(folder):
    while True:
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        if not pids:
            return
        status(folder, 'WAITING_FOR_GPU', compute_pids=pids)
        time.sleep(45)
