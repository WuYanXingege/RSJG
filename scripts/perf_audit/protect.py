"""Read-only receipts; naturally changing live logs are explicitly separate."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
BASE = SOURCE.parent
OUTPUT = BASE/'RSJG_perf_outputs/bcep_v3_speed_audit_20261120'
ROOTS = {k: BASE/'RSJG_EBJD_assets/experiments'/v for k,v in dict(
    formal='bcep_ebjd_v2_integrated', candidate_cost='zara2_costproxy_stage2_20261119',
    sharedworld='zara2_sharedworld_stage2_onerun_20261119', live='zara2_bcep_v3_stage2_20261120').items()}
ROOTS['live_source'] = BASE/'RSJG_BCEP_v3_stage2_trial_20261120'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def live_receipt():
    root=ROOTS['live']; files={}
    for name in ['STATUS.json','SUPERVISOR_STATUS.json','joint/PROGRESS.json','joint/EPOCH_METRICS.json','joint/TOP5.json']:
        p=root/name
        if p.exists():
            raw=p.read_bytes()
            files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
    log=root/'MODULE_UPDATES.jsonl'
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-8192));lines=f.read().splitlines()
    complete=[]
    for line in lines:
        try:complete.append(json.loads(line))
        except json.JSONDecodeError:pass
    files['latest_logged_update']=complete[-1]['update']
    return dict(unix=time.time(),files=files)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['before','after']);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='before':
        assert not (OUTPUT/'PROTECTED_BEFORE.json').exists(), 'do not overwrite initial evidence'
        candidates={}
        for label in ['formal','candidate_cost','sharedworld']:
            for p in ROOTS[label].rglob('*'):
                if p.is_file() and p.suffix in {'.pt','.npy','.npz','.json','.csv','.md','.txt'}:
                    candidates[label+'/'+str(p.relative_to(ROOTS[label]))]=p
        root=ROOTS['live']
        for name in ['FROZEN_CONFIG.json','AUTHORIZATION.json','EXECUTION_AUTHORIZATION.json','INITIALIZATION_ORIGIN.json']:
            candidates['live/'+name]=root/name
        for p in (root/'joint').glob('epoch_*.pt'):candidates['live/'+str(p.relative_to(root))]=p
        names=subprocess.check_output(['git','ls-files'],cwd=ROOTS['live_source'],text=True).splitlines()
        for name in names:candidates['live_source/'+name]=ROOTS['live_source']/name
        before={name:dict(sha256=sha(p),size=p.stat().st_size) for name,p in sorted(candidates.items())}
        write(OUTPUT/'PROTECTED_BEFORE.json',before);write(OUTPUT/'LIVE_BEFORE.json',live_receipt())
        print('protected static files',len(before),'bytes',sum(v['size'] for v in before.values()),flush=True)
        return
    before=json.loads((OUTPUT/'PROTECTED_BEFORE.json').read_text());changed=[]
    for name,info in before.items():
        label,rel=name.split('/',1);p=ROOTS[label]/rel
        if not p.exists() or p.stat().st_size!=info['size'] or sha(p)!=info['sha256']:changed.append(name)
    after=live_receipt();write(OUTPUT/'LIVE_AFTER.json',after)
    prior=json.loads((OUTPUT/'LIVE_BEFORE.json').read_text())
    result=dict(status='PASS' if not changed else 'FAIL',static_files=len(before),static_manifest_sha256=canonical(before),
        static_files_unchanged=not changed,changed=changed,
        old_queue_states={k:json.loads((ROOTS[k]/'STATUS.json').read_text())['status'] for k in ['formal','candidate_cost','sharedworld']},
        live_status_before=prior['files']['STATUS.json']['value'],live_status_after=after['files']['STATUS.json']['value'],
        native_logged_update_before=prior['files']['latest_logged_update'],native_logged_update_after=after['files']['latest_logged_update'],
        audit_added_real_optimizer_updates=0,
        active_logs_and_new_checkpoints='Natural authorized training advancement; excluded from immutable hash equality; before/after receipts retained locally.',
        no_signal_no_restart_no_attach=True,no_live_worktree_edits=True,no_gpu_work=True)
    write(OUTPUT/'PROTECTION.json',result);assert not changed,changed
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
