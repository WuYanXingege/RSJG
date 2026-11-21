"""External receipts only. The model runs from the separate fixed worktree."""
import csv, hashlib, json, subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
TRAIN_SOURCE = SOURCE.parent / 'RSJG_SSEC_formal_ready_20261121'
RUN = SOURCE.parent / 'RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_20261121'
DOC = SOURCE / 'docs/research/bcep_scene_energy/zara2_trial_20261121'
BASE = 'ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86'
RECOVERY_BASE = '9fad8f6dfbcefaebaa6a4944db3f5d35062ff91b'
FROZEN = '3ff36b653703307be4f38457d9cb9395c264ea7647c0df339913376f38b9d842'
BRANCH = 'experiment/ssec-zara2-energy-only-20261121'
PUBLIC_PATH = str(DOC.relative_to(SOURCE))
LIMITS = [0.4097670185111337, 0.7841791655426308, 1.2968434301886971]


def read(p): return json.loads(Path(p).read_text())


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def append(p, value):
    with Path(p).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def git(*args,cwd=SOURCE):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def csvwrite(p, rr):
    assert rr
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    t.replace(p)


def source_check():
    assert git('rev-parse','HEAD',cwd=TRAIN_SOURCE)==BASE
    assert not git('status','--porcelain',cwd=TRAIN_SOURCE)
    assert digest(RUN/'FROZEN_CONFIG.json')==FROZEN
    config=read(RUN/'FROZEN_CONFIG.json')
    for name,sha in config['implementation_files'].items():assert digest(TRAIN_SOURCE/name)==sha,name
    return config


def command():
    p=TRAIN_SOURCE/'docs/research/bcep_scene_energy/stage2_ready_20261121/CONTROLLED_COMMAND.txt'
    active=[line for line in p.read_text().splitlines() if line.strip() and not line.startswith('#')]
    assert len(active)==1
    expected=f'cd -- {TRAIN_SOURCE} && env RUN_ONE_ZARA2_SSEC=true CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {SOURCE.parent}/.conda/rsjg/bin/python -m bcep_scene_energy.formal train --frozen-sha {FROZEN}'
    assert active[0]==expected
    return active[0]
