"""Read-only source/evidence checks; never execute CONTROLLED_COMMAND.txt."""
import re,subprocess
from common import *
from bcep_scene_energy.formal_contract import READY,source_manifest


def main():
    setup();r=read(PUBLIC/'READINESS.json');assert r['status']==READY and all(r['gates'].values())
    assert r['real_optimizer_updates']==r['synthetic_optimizer_updates']==r['completed_training_epochs']==0
    assert not r['training_authorized'] and not r['unresolved_engineering_blockers']
    cfg=read(PUBLIC/'FROZEN_CONFIG.json')
    assert file_sha(PUBLIC/'FROZEN_CONFIG.json')==file_sha(ROOTS['run']/'FROZEN_CONFIG.json')==r['frozen_file_sha256']
    assert source_manifest()==cfg['implementation_files']
    assert cfg['variant']=='energy_only' and cfg['training_seed']==3201 and cfg['candidate_seed']==2035
    assert cfg['config']['pair_objective']=='final_coordinate' and cfg['recipe']['minimum']==30 and cfg['recipe']['patience']==15 and cfg['recipe']['max_epochs']==100
    assert (PUBLIC/'CONFIG_SHA256.txt').read_text().splitlines()[0].split()[1]==r['frozen_file_sha256']
    command=(PUBLIC/'CONTROLLED_COMMAND.txt').read_text()
    assert command.count(' -m bcep_scene_energy.formal train ')==1 and '--frozen-sha '+r['frozen_file_sha256'] in command and 'RUN_ONE_ZARA2_SSEC=true' in command
    assert not (ROOTS['run']/'joint/last.pt').exists() and not (ROOTS['run']/'AUTHORIZATION.json').exists()
    for file in ['CPU_RECEIPT.json','REAL_INPUTS.json','REAL_ACCEPTANCE.json','RECOVERY_RECEIPT.json','INITIAL_DEV.json','LOGICAL_GROUP_BACKWARD.json','ASSET_RECEIPT.json','RUNTIME.json']:
        assert read(PUBLIC/file)['optimizer_guard']['optimizer_step_calls']==0
    for md in PUBLIC.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if '://' not in target and not target.startswith('#'):assert (md.parent/target.split('#')[0]).exists(),(md,target)
    assert not subprocess.check_output(['git','diff',BASE,'--','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True)
    write(PUBLIC/'DELIVERY_CHECKS.json',dict(passed=True,status=READY,command_not_executed=True,source_and_config_hashes_match=True,
        report_links_valid=True,optimizer_guard=GUARD,real_optimizer_updates=0,synthetic_optimizer_updates=0))
    files=list((SOURCE/'bcep_scene_energy').glob('formal*.py'))+list((SOURCE/'scripts/ssec_ready').glob('*.py'))+[SOURCE/'tests/test_ssec_formal.py']
    files += [p for p in PUBLIC.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json']
    write(PUBLIC/'ARTIFACT_SHA256.json',{str(p.relative_to(SOURCE)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)})
    print('Delivery checks PASS; controlled command remains unexecuted',flush=True)
if __name__=='__main__':main()
