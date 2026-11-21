"""Aggregate existing receipts, verify assets and issue engineering readiness (no training)."""
import subprocess,xml.etree.ElementTree as ET
from common import *


def csvread(name):
    with (DOC/name).open() as f:return list(csv.DictReader(f))


def main():
    setup();rows=csvread('GPU_BENCHMARK.csv');assert len(rows)==168
    summary=[]
    for n,dense in [(n,'False') for n in [1,2,9,20,42,57]]+[(57,'True')]:
        case=[r for r in rows if int(r['N'])==n and r['dense_all_tied_neighbors']==dense]
        assert len({r['input_sha256'] for r in case})==1
        for model in ['v2','v3','energy_only','energy_csee']:
            for kind in ['coupling_pair','full_native_objective']:
                a=[r for r in case if r['model']==model and r['kind']==kind];assert len(a)==3
                row=dict(model=model,N=n,dense_all_tied_neighbors=dense,kind=kind,repetitions=3)
                for metric in ['forward_seconds','backward_seconds','total_seconds','allocated_peak_MB','hungarian_calls']:
                    v=[float(r[metric]) for r in a]
                    row[metric+'_median']=float(np.median(v));row[metric+'_maximum']=max(v)
                row['edges']=a[0]['edges'];summary.append(row)
                assert int(float(a[0]['hungarian_calls']))==(0 if n==1 else 2*n)
    csvwrite(DOC/'PERFORMANCE_SUMMARY.csv',summary)
    uniqueness=csvread('UNIQUENESS_BENCHMARK.csv');u=[]
    for kind in ['random','strict_tie','near_tie']:
        a=[r for r in uniqueness if r['kind']==kind]
        new=float(np.median([float(r['new_seconds']) for r in a]));old=float(np.median([float(r['original_seconds']) for r in a]))
        u.append(dict(kind=kind,matrices=len(a),new_median_seconds=new,original_median_seconds=old,speedup=old/new,
                      additional_Hungarian_new=0,additional_Hungarian_original=20))
    csvwrite(DOC/'UNIQUENESS_SUMMARY.csv',u)
    from bcep_scene_energy.matching import solve,reference_gap
    cases=[]
    for name,s in [('subnormal',np.eye(20,dtype='f')*np.nextafter(np.float32(0),np.float32(1))),
                   ('wide_dynamic_range',np.array([[1e30,0,0],[0,1e-20,0],[0,0,1e-20]],dtype='f'))]:
        p,c=solve(s);cases.append(dict(name=name,integer_certificate=c,old_FP64_summed_gap=reference_gap(s,p),
            note='wide-range case validated against exact integer exhaustive enumeration; FP64 total subtraction can lose the positive gap'))
    write(DOC/'INTEGER_EDGE_CASES.json',cases)
    gates={}
    for file in ['CPU_RECEIPT.json','GPU_RECEIPT.json','MECHANISM_RECEIPT.json','INTERFACE_RECEIPT.json','DDIM_RECEIPT.json']:
        r=read(DOC/file);gates[file]=r['passed'] and r['optimizer_guard']['optimizer_step_calls']==0
    gpu=read(DOC/'GPU_RECEIPT.json')
    gates['GPU_receipt_matches_final_network_source']=all(sha(SOURCE/n)==h for n,h in gpu['network_source_sha256'].items())
    traces=csvread('GPU_ENERGY_TRACE.csv')+csvread('CPU_ENERGY_TRACE.csv')
    gates['certified_monotone_updates']=all(float(r['energy_after_fp64'])<=float(r['energy_before_fp64']) and
        (r['accepted']=='False' or (r['unique']=='True' and -float(r['energy_delta_fp64'])>float(r['arithmetic_error_bound']))) for r in traces)
    gates['native_Hungarian_exact_on_neural_proposals']=all(r['native_solution_exactly_optimal']=='True' for r in traces)
    gates['new_uniqueness_median_faster_in_each_measured_class']=all(r['speedup']>1 for r in u)
    permutation=csvread('PERMUTATION_AUDIT.csv')
    gates['carried_incumbent_candidate_world_relabel']=all(int(r['relative_changed'])==0 for r in permutation if r['operation'] in
        ['uniform_world_relabel','independent_candidate_relabel_carried_incumbent','people_relabel_same_physical_visit_order'])
    # Do not substitute a conditional/kernel test for the DEFAULT ENTRY check.
    gates['default_entry_candidate_relabel_stable']=all(int(r['relative_changed'])==0 for r in permutation if r['operation']=='candidate_relabel_native_S0_recomputed')
    before=read(OUT/'PROTECTED_BEFORE.json');after=snapshot()
    gates['historical_assets_unchanged']=before==after
    write(OUT/'PROTECTED_AFTER.json',after)
    native=subprocess.check_output(['git','diff',BASE,'--','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True)
    gates['native_source_unchanged']=native==''
    states={name:read(OUT.parent/name/'STATUS.json') for name in ['bcep_ebjd_v2_integrated','zara2_costproxy_stage2_20261119','zara2_sharedworld_stage2_onerun_20261119','zara2_bcep_v3_stage2_20261120']}
    gates['old_queue_states_unchanged']=[r['status'] for r in states.values()]==['PAUSED_RESUMABLE','SESSION_BUDGET_PAUSED','COMPLETE','COMPLETE']
    from bcep.contracts import canonical_hash
    write(DOC/'ASSET_RECEIPT.json',dict(protected_files=len(before),protected_bytes=sum(v['bytes'] for v in before.values()),
        unchanged=before==after,manifest_canonical_sha256=canonical_hash(before),queue_statuses={k:v['status'] for k,v in states.items()},
        v3_historical_epochs=len(list((OUT.parent/'zara2_bcep_v3_stage2_20261120/joint').glob('epoch_*.pt'))),
        v3_historical_updates=states['zara2_bcep_v3_stage2_20261120']['optimizer_updates'],native_source_unchanged=native=='',real_updates=0,synthetic_updates=0))
    cfg=read(DOC/'FROZEN_CONFIG.json');interfaces=read(DOC/'INTERFACE_RECEIPT.json')
    cfg.update(provider_identity=interfaces['provider'],bank_sha256=interfaces['bank_sha256'],official_counts=interfaces['official_counts'],
        native_recipe=dict(optimizer='AdamW',lr=1e-4,weight_decay=1e-4,warmup_epochs=5,scheduler_horizon=100,minimum=30,patience=15,max_epochs=100,
                           logical_scenes=16,updates_per_epoch=168,train_seed=3201),
        retention='unchanged native writer: all complete epochs, best, last, and H-JADE Top5 distinct epochs; exact ties earlier epoch',
        partial_epoch_never_claimed_complete=True,public_training_command_enabled=False,network_source_sha256=gpu['network_source_sha256'])
    write(DOC/'FROZEN_CONFIG.json',cfg)
    (DOC/'CONFIG_SHA256.txt').write_text('file_sha256 '+sha(DOC/'FROZEN_CONFIG.json')+'\ncanonical_sha256 '+canonical_hash(cfg)+'\n')
    tests=ET.parse(DOC/'CPU_TESTS.xml');count=sum(int(x.get('tests',0)) for x in tests.getroot().iter('testsuite'))
    # Public test receipt retains counts/results/times, omits workstation hostname.
    for e in tests.getroot().iter('testsuite'):e.attrib.pop('hostname',None)
    tests.write(DOC/'CPU_TESTS.xml',encoding='utf-8',xml_declaration=True)
    readiness=dict(status='READY_ENGINEERING_TRAINING_DISABLED' if all(gates.values()) else 'BLOCKED',gates=gates,CPU_tests=count,
        real_optimizer_updates=0,synthetic_optimizer_updates=0,training_authorized=False,primary_variant='energy_only',
        unresolved_engineering_blockers=[k for k,v in gates.items() if not v],
        limitations=['one-sweep physical visitation-order dependence','native S0 tie and candidate relabel dependence may remain',
                     'proposal-ST is a biased training-only adjoint; uniform-soft/hard discrepancy remains',
                     'no learned generalization, convergence or H-JADE/H-JFDE improvement established',
                     'measured slowdown versus v2/v3; no optimizer-state-allocation peak or real training executed',
                     'completed-epoch early-stop/Top5 controller unchanged, not reexecuted with forbidden optimizer steps'])
    write(DOC/'READINESS.json',readiness)
    print(json.dumps(readiness,ensure_ascii=False,indent=2))
    if not all(gates.values()):raise SystemExit(2)


if __name__=='__main__':main()
