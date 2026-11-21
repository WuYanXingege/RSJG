"""Read-only first-update/epoch receipts. No training or model constructor."""
import argparse, math, shutil, sys, time
from common import *
sys.path.insert(0,str(TRAIN_SOURCE))
import torch


def lines(p):
    with Path(p).open() as f:
        for line in f:
            try:yield json.loads(line)
            except json.JSONDecodeError:break # active writer may not have finished its final line


def modules(row):
    for prefix in ['coupling.energy.phi.','coupling.energy.rho.','bridge.encoder.']:
        m=row['modules'][prefix]
        assert all(math.isfinite(v) for v in m.values())
        assert m['gradient_l2']>0 and m['parameter_delta_l2']>0,prefix
    for g in row['physical_groups']:
        for s in g['scenes']:assert s['certified_monotone'] and math.isfinite(s['energy_after_fp64'])


def checkpoint(p):
    s=torch.load(p,map_location='cpu',weights_only=False)
    f=read(RUN/'FROZEN_CONFIG.json')
    assert s['identity']==f['identity'] and s['recipe']==f['recipe']
    assert {'model','optimizer','scheduler','rng','order','cursor','history','epoch_sums','epoch_counts','epoch_raw_sums'}<=s.keys()
    assert {'cpu','cuda','python','numpy','order','view'}<=s['rng'].keys()
    assert s['optimizer']['state'] and all(torch.isfinite(t).all() for t in s['model'].values())
    return s


def first_update():
    source_check()
    first=read(RUN/'joint/FIRST_UPDATE.json');assert first['update']==1
    row=next(lines(RUN/'MODULE_UPDATES.jsonl'));assert row['update']==1;modules(row)
    saved=checkpoint(RUN/'joint/last.pt');assert saved['update']>=1
    result=dict(passed=True,first_real_update=1,first_update_progress=first,
        modules=row['modules'],coordinated_windows=sum(len(g['scenes']) for g in row['physical_groups']),
        accepted_updates_reliably_monotone=True,finite_energy=True,checkpoint_update=saved['update'],
        complete_checkpoint_fields=True,source_commit=BASE,frozen_file_sha256=FROZEN,
        allocated_peak_bytes=row['allocated_peak_bytes'],disk_free_bytes=shutil.disk_usage(RUN).free,unix=time.time())
    write(RUN/'FIRST_UPDATE_VERIFICATION.json',result)
    print('SSEC FIRST REAL UPDATE VERIFIED',flush=True)


def first_epoch():
    source_check()
    if not (RUN/'FIRST_UPDATE_VERIFICATION.json').exists():first_update()
    p=RUN/'joint/epoch_001.pt';s=checkpoint(p)
    assert s['completed_epoch']==1 and s['update']==168 and s['cursor']==0 and s['order'] is None
    top=read(RUN/'joint/TOP5.json');assert top[0]['epoch']==1 and len(top)==1
    assert digest(p)==top[0]['sha256']==digest(RUN/'joint/best.pt')
    effective={r['update']:r for r in lines(RUN/'MODULE_UPDATES.jsonl') if r['update']<=168}
    assert set(effective)==set(range(1,169))
    for r in effective.values():modules(r)
    rr=rows(RUN/'dev/epoch_001_windows.csv');assert len(rr)==590
    assert sum(int(r['n_agents'])>=9 for r in rr)==127
    assert all(r['exact_endpoint']==r['exact_noise_identity']==r['certified_monotone']=='True' for r in rr)
    groups=read(RUN/'dev/epoch_001_groups.json');summary=read(RUN/'dev/epoch_001_training.json')
    assert summary['optimizer_updates']==168 and summary['journal_update_first']==1 and summary['journal_update_last']==168
    write(RUN/'FIRST_EPOCH_VERIFICATION.json',dict(passed=True,completed_epoch=1,real_updates=168,
        epoch_checkpoint_sha256=digest(p),retention='all 1 available completed epoch + best + last + TOP5 index',
        full_checkpoint_and_rng=True,windows=590,dense_windows=127,endpoint_noise_marginal_and_bijection_contract=True,
        modules_all_168_steps_finite_nonzero=True,energy_all_168_steps_certified=True,groups=groups,
        monitor=summary,disk_free_bytes=shutil.disk_usage(RUN).free,unix=time.time()))
    print('SSEC FIRST COMPLETE EPOCH VERIFIED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['first-update','first-epoch']);a=p.parse_args()
    torch.set_num_threads(1)
    (first_update if a.action=='first-update' else first_epoch)()
