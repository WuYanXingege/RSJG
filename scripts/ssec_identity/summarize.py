"""Do not remove or weaken any gate from the fixed acceptance baseline."""
from common import *
from bcep.contracts import canonical_hash


def rows(name):return list(csv.DictReader((DOC/name).open()))

def main():
    setup();parent=SOURCE/'docs/research/bcep_scene_energy/acceptance_20261121'
    old=read(parent/'READINESS.json');gates={k:False for k in old['gates']};evidence={}
    for name in ['CPU_RECEIPT.json','GPU_RECEIPT.json','MECHANISM_RECEIPT.json','INTERFACE_RECEIPT.json']:
        r=read(DOC/name);gates[name]=r['passed'] and r['optimizer_guard']['optimizer_step_calls']==0;evidence[name]='new run'
    end=rows('END_TO_END_DDIM.csv')
    gates['DDIM_RECEIPT.json']=len(end)==8 and all(float(r['G_max_abs'])==float(r['H_max_abs'])==float(r['H_endpoint_max_abs'])==0 and int(r['final_relative_changed'])==0 for r in end)
    evidence['DDIM_RECEIPT.json']='new END_TO_END_DDIM.csv: six N + duplicate + native augmentation'
    write(DOC/'DDIM_RECEIPT.json',dict(passed=gates['DDIM_RECEIPT.json'],optimizer_guard=GUARD,comparisons=len(end),steps=20,K=20,full_G_H_exact=True))
    full=read(DOC/'gpu_full_regression/GPU_RECEIPT.json')
    gates['GPU_RECEIPT.json']=gates['GPU_RECEIPT.json'] and full['passed'] and full['optimizer_guard']['optimizer_step_calls']==0
    evidence['GPU_RECEIPT.json']='new same-state before/after plus unchanged full prior suite: v2/v3/SSEC/CSEE, all N and dense N57'
    gpu=read(DOC/'GPU_RECEIPT.json')
    gates['GPU_receipt_matches_final_network_source']=all(sha(SOURCE/n)==v for n,v in gpu['network_sha256'].items()) and all(sha(SOURCE/n)==v for n,v in full['network_source_sha256'].items())
    traces=rows('ENERGY_TRACE.csv')+rows('gpu_full_regression/GPU_ENERGY_TRACE.csv')
    gates['certified_monotone_updates']=all(float(r['energy_after_fp64'])<=float(r['energy_before_fp64']) and (r['accepted']=='False' or (r['unique']=='True' and -float(r['energy_delta_fp64'])>float(r['arithmetic_error_bound']))) for r in traces)
    gates['native_Hungarian_exact_on_neural_proposals']=all(r['native_solution_exactly_optimal']=='True' for r in traces)
    original_matching=subprocess.check_output(['git','show',BASE+':bcep_scene_energy/matching.py'],cwd=SOURCE)
    assert original_matching==(SOURCE/'bcep_scene_energy/matching.py').read_bytes()
    old_bench=list(csv.DictReader((parent/'UNIQUENESS_SUMMARY.csv').open()))
    gates['new_uniqueness_median_faster_in_each_measured_class']=old['gates']['new_uniqueness_median_faster_in_each_measured_class'] and all(float(r['speedup'])>1 for r in old_bench)
    evidence['new_uniqueness_median_faster_in_each_measured_class']='retained fixed-base timing; matching.py byte-for-byte unchanged; exact kernel tests rerun; no new uniqueness speedup claimed'
    orders=rows('VISIT_ORDER_AUDIT.csv')
    carried=[r for r in orders if r['entry']=='carried' and r['operation'] in ['people_relabel_same_physical_visit','uniform_world_relabel']]
    gates['carried_incumbent_candidate_world_relabel']=len(carried)==20 and all(int(r['relative_changed'])==0 for r in carried) and read(DOC/'CPU_RECEIPT.json')['passed']
    cmp=rows('LAYER_AND_GRADIENT_COMPARISON.csv')
    gates['default_entry_candidate_relabel_stable']=len(cmp)==10 and {(r['device'],int(r['N'])) for r in cmp}=={(d,n) for d in ['cpu','cuda'] for n in [2,9,20,42,57]} and all(int(r['initial_changed'])==int(r['final_changed'])==int(r['final_relative_changed'])==0 and float(r['S0_max_abs'])==0 for r in cmp)
    before=read(OUT/'PROTECTED_BEFORE.json');after=protected();write(OUT/'PROTECTED_AFTER.json',after)
    gates['historical_assets_unchanged']=before==after
    native=subprocess.check_output(['git','diff',BASE,'--','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True)
    gates['native_source_unchanged']=not native
    names=['bcep_ebjd_v2_integrated','zara2_costproxy_stage2_20261119','zara2_sharedworld_stage2_onerun_20261119','zara2_bcep_v3_stage2_20261120']
    states={n:read(OUT.parent/n/'STATUS.json')['status'] for n in names}
    gates['old_queue_states_unchanged']=list(states.values())==['PAUSED_RESUMABLE','SESSION_BUDGET_PAUSED','COMPLETE','COMPLETE']
    unchanged=['energy.py','matching.py','verification.py','model.py','interface.py']
    for n in unchanged:assert subprocess.check_output(['git','show',BASE+':bcep_scene_energy/'+n],cwd=SOURCE)==(SOURCE/'bcep_scene_energy'/n).read_bytes()
    oldcoupling=subprocess.check_output(['git','show',BASE+':bcep_scene_energy/coupling.py'],cwd=SOURCE,text=True)
    newcoupling=(SOURCE/'bcep_scene_energy/coupling.py').read_text()
    class_body=lambda s:s[s.index('class CoordinatedAssignment'):s.index('class SceneEnergyCoupling')]
    assert class_body(oldcoupling)==class_body(newcoupling)
    for name in ['IDENTITY_INTERFACE_RECEIPT.json','NOISE_CONTRACT_RECEIPT.json','BASE_REPRODUCTION.json']:
        r=read(DOC/name);gates[name]=r['passed'] and r['optimizer_guard']['optimizer_step_calls']==0
    gates['full_objective_relabel_gradient_exact']=all(r['all_model_gradients_exact']=='True' and float(r['loss_abs_delta'])==0 for r in rows('FULL_OBJECTIVE_GRADIENTS.csv'))
    gates['old_tests_not_deleted_or_changed']=subprocess.check_output(['git','show',BASE+':tests/test_bcep_scene_energy.py'],cwd=SOURCE)==(SOURCE/'tests/test_bcep_scene_energy.py').read_bytes()
    assert set(old['gates']).issubset(gates)
    write(DOC/'ASSET_RECEIPT.json',dict(unchanged=before==after,protected_files=len(before),protected_bytes=sum(v['bytes'] for v in before.values()),
        manifest_canonical_sha256=canonical_hash(before),queue_states=states,native_source_unchanged=not native,optimizer_guard=GUARD))
    cfg=read(parent/'FROZEN_CONFIG.json');cfg.update(base=BASE,parent_config_sha256=sha(parent/'FROZEN_CONFIG.json'),
        protected_before_files=len(before),protected_before_bytes=sum(v['bytes'] for v in before.values()),
        identity_order='exact FP32 x, then y, then transported original-slot ID only for coincident coordinates',
        coordinate_rounding_or_epsilon=False,coincident_without_identity='raise CandidateIdentityAmbiguity',
        single_person_and_padding='unchanged native input-index identity; single-person relabel is a world relabel',
        G_definition='original frozen bank slot combination; restore by transported IDs before native DDIM',
        noise_key='unchanged scene, agent, ORIGINAL bank slot, seed',
        person_visit_order='unchanged supplied physical order, default valid-array order; no new person-ID ordering',
        allow_real_training=False,real_optimizer_updates=0,synthetic_optimizer_updates=0,network_source_sha256=gpu['network_sha256'])
    write(DOC/'FROZEN_CONFIG.json',cfg)
    (DOC/'CONFIG_SHA256.txt').write_text('file_sha256 '+sha(DOC/'FROZEN_CONFIG.json')+'\ncanonical_sha256 '+canonical_hash(cfg)+'\n')
    readiness=dict(base=BASE,status='READY_ENGINEERING_TRAINING_DISABLED' if all(gates.values()) else 'BLOCKED',gates=gates,
        inherited_gate_evidence=evidence,original_gate_names_preserved=True,CPU_tests=read(DOC/'CPU_RECEIPT.json')['tests'],
        real_optimizer_updates=0,synthetic_optimizer_updates=0,training_authorized=False,formal_training_execution_enabled=False,
        original_101_changed_slots_remaining=next(int(r['final_relative_changed']) for r in cmp if r['device']=='cpu' and r['N']=='57'),
        unresolved_engineering_blockers=[k for k,v in gates.items() if not v],
        admission_scope='fixed ZARA2 bank/membership order and original native protocol; identity/noise transport required for reordered inputs; no new train launch',
        limitations=['physical visitation order dependence remains',
            'default S0 can remain sensitive to PERSON-array reordering even with the same physical visit order; no unconditional person-equivariance claim',
            'coincident endpoints alone cannot recover original noise occurrence IDs; unlabelled coincidences fail closed',
            'proposal-ST bias, near-uniform Sinkhorn and hardware-dependent near ties remain',
            'no training/convergence/JADE/JFDE improvement was measured',
            'formal training command remains disabled; separate explicit authorization required'])
    write(DOC/'READINESS.json',readiness)
    print(json.dumps(readiness,ensure_ascii=False,indent=2),flush=True)
    if not all(gates.values()):raise SystemExit(2)
if __name__=='__main__':main()
