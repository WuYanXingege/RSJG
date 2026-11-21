"""Verify committed evidence and source constraints without executing a model."""
import ast,re
from common import *


def main():
    setup();r=read(DOC/'READINESS.json')
    assert r['status']=='READY_ENGINEERING_TRAINING_DISABLED' and all(r['gates'].values())
    assert r['real_optimizer_updates']==r['synthetic_optimizer_updates']==0 and not r['training_authorized']
    assert r['CPU_tests']==32 and r['original_101_changed_slots_remaining']==0
    parent=SOURCE/'docs/research/bcep_scene_energy/acceptance_20261121'
    assert set(read(parent/'READINESS.json')['gates']).issubset(r['gates'])
    assert not subprocess.check_output(['git','diff',BASE,'--','docs/research/bcep_scene_energy/acceptance_20261121','tests/test_bcep_scene_energy.py','scripts/ssec_acceptance','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True)
    files=list((SOURCE/'bcep_scene_energy').glob('*.py'))+list((SOURCE/'scripts/ssec_identity').glob('*.py'))+[SOURCE/'tests/test_ssec_identity.py']
    for p in files:
        tree=ast.parse(p.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(tree)),p
    gpu=read(DOC/'GPU_RECEIPT.json')
    for n,h in gpu['network_sha256'].items():assert sha(SOURCE/n)==h
    for p in DOC.rglob('*RECEIPT.json'):
        v=read(p)
        if 'optimizer_guard' in v:assert v['optimizer_guard']['optimizer_step_calls']==0
    for md in DOC.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if '://' not in link and not link.startswith('#'):assert (md.parent/link.split('#')[0]).exists(),(md,link)
    cfg=read(DOC/'FROZEN_CONFIG.json')
    assert not cfg['allow_real_training'] and cfg['pair_objective']=='final_coordinate' and cfg['candidate_seed']==2035 and cfg['initialization_seed']==3201
    assert sha(DOC/'FROZEN_CONFIG.json')==(DOC/'CONFIG_SHA256.txt').read_text().splitlines()[0].split()[1]
    write(DOC/'DELIVERY_CHECKS.json',dict(passed=True,engineering_status=r['status'],optimizer_guard=GUARD,
        original_gates_retained=True,old_tests_and_reports_unchanged=True,network_hashes_match=True,local_links_valid=True))
    files+=list(p for p in DOC.rglob('*') if p.is_file() and p.name!='ARTIFACT_SHA256.json')
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(SOURCE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
    print('Delivery integrity PASS; training remains disabled; optimizer updates 0',flush=True)
if __name__=='__main__':main()
