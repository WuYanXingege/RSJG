"""Bind all zero-update receipts to one frozen runtime and write readiness."""
import ast,shlex,subprocess
from common import *
from bcep_scene_energy.formal_contract import READY,source_manifest


def main():
    setup();frozen=read(ROOTS['run']/'FROZEN_CONFIG.json');sha=file_sha(ROOTS['run']/'FROZEN_CONFIG.json');gates={}
    assert read(PUBLIC/'FROZEN_CONFIG.json')==frozen
    gates['source_manifest_frozen']=source_manifest()==frozen['implementation_files']
    for file in ['CPU_RECEIPT.json','REAL_INPUTS.json','REAL_ACCEPTANCE.json','RECOVERY_RECEIPT.json','INITIAL_DEV.json','LOGICAL_GROUP_BACKWARD.json']:
        r=read(PUBLIC/file);gates[file]=r['passed'] and r['optimizer_guard']['optimizer_step_calls']==0
        if 'implementation_hash' in r:gates[file]=gates[file] and r['implementation_hash']==frozen['implementation_hash']
    initial=read(PUBLIC/'INITIAL_DEV.json')
    gates['590_windows_20_DDIM_initial_only']=initial['completed_windows']==590 and initial['DDIM_steps']==20 and not initial['selection'] and initial['model_unchanged']
    gates['initial_model_fingerprint']=initial['initial_model_sha256']==frozen['initial_model_sha256']==read(PUBLIC/'REAL_ACCEPTANCE.json')['initial_model_sha256']
    gates['exact_original_grouped_backward']=read(PUBLIC/'REAL_ACCEPTANCE.json')['native_vs_instrumented_backward_exact']
    windows=list(csv.DictReader((OUT/'dev/epoch_000_windows.csv').open()))
    gates['energy_monotone_and_finite']=len(windows)==590 and all(r['certified_monotone']=='True' and all(np.isfinite(float(r[k])) for k in ['H_JADE','I_JFDE','H_JFDE','sinkhorn_row_entropy','final_proposal_effective_score_RMS']) for r in windows)
    gates['frozen_marginal']=abs(initial['metrics']['minFDE']-.3520422351797007)<1e-6 and abs(initial['metrics']['scene_weighted_minFDE']-.33942601943420153)<1e-6
    expected_recipe=dict(seed=3201,lr=.0001,weight_decay=.0001,clip=5.,logical_scenes=16,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
    gates['original_optimizer_scheduler_selector']=frozen['recipe']==expected_recipe and frozen['config']['pair_objective']=='final_coordinate'
    # Original network files and original native/v2/v3 controllers are untouched.
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'bcep_scene_energy','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True).splitlines()
    gates['existing_architecture_and_controllers_unchanged']=all(subprocess.check_output(['git','show',BASE+':'+n],cwd=SOURCE)==(SOURCE/n).read_bytes() for n in tracked)
    before=read(OUT/'PROTECTED_BEFORE.json');after=protected();write(OUT/'PROTECTED_AFTER.json',after)
    gates['all_protected_assets_unchanged']=before==after
    states={k:read(ROOTS[k]/'STATUS.json')['status'] for k in ['formal','v3','candidate_cost','sharedworld']}
    gates['protected_queue_states']=states==dict(formal='PAUSED_RESUMABLE',v3='COMPLETE',candidate_cost='SESSION_BUDGET_PAUSED',sharedworld='COMPLETE')
    run=read(ROOTS['run']/'STATUS.json')
    gates['no_training_started']=run['status']=='PREPARED_NO_TRAINING_AUTHORIZATION' and run['optimizer_updates']==0 and not (ROOTS['run']/'joint/last.pt').exists() and not (ROOTS['run']/'AUTHORIZATION.json').exists()
    # Verify no hidden optimizer.step calls in the new implementation/audit paths.
    new=[p for p in (SOURCE/'bcep_scene_energy').glob('formal*.py')]+list((SOURCE/'scripts/ssec_ready').glob('*.py'))+[SOURCE/'tests/test_ssec_formal.py']
    gates['no_new_step_call']=all(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(ast.parse(p.read_text()))) for p in new)
    write(PUBLIC/'ASSET_RECEIPT.json',dict(unchanged=before==after,protected_files=len(before),protected_bytes=sum(v['bytes'] for v in before.values()),
        manifest_canonical_sha256=canonical_hash(before),states=states,new_run_status=run,optimizer_guard=GUARD))
    for p in (OUT/'dev').iterdir():
        if p.is_file():(PUBLIC/p.name).write_bytes(p.read_bytes())
    readiness=dict(status=READY if all(gates.values()) else 'BLOCKED',base=BASE,frozen_file_sha256=sha,
        source_manifest_sha256=frozen['implementation_hash'],initial_model_sha256=frozen['initial_model_sha256'],gates=gates,
        real_optimizer_updates=0,synthetic_optimizer_updates=0,training_authorized=False,completed_training_epochs=0,
        CPU_tests=read(PUBLIC/'CPU_RECEIPT.json')['tests'],unresolved_engineering_blockers=[k for k,v in gates.items() if not v],
        limitations=['zero-step recovery does not claim equivalence after positive AdamW updates',
            'Top5/stop rules tested via actual native writer/expressions with explicit fixtures; no trained synthetic epochs',
            'actual post-step parameter changes will be observed only after separate training authorization',
            'fixed person membership/order remains required; SSEC order dependence and soft/hard bias persist',
            'initial dev metrics do not establish trained model quality'])
    write(PUBLIC/'READINESS.json',readiness);write(OUT/'READINESS.json',readiness)
    command='cd -- '+shlex.quote(str(SOURCE))+' && env RUN_ONE_ZARA2_SSEC=true CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.quote(str(SOURCE.parent/'.conda/rsjg/bin/python'))+' -m bcep_scene_energy.formal train --frozen-sha '+sha
    text='# Prepared only; execute ONLY after separate explicit training authorization.\n'+command+'\n'
    (PUBLIC/'CONTROLLED_COMMAND.txt').write_text(text);(ROOTS['run']/'CONTROLLED_COMMAND.txt').write_text(text)
    print(json.dumps(readiness,ensure_ascii=False,indent=2))
    if not all(gates.values()):raise SystemExit(2)
if __name__=='__main__':main()
