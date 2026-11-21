"""Delivery integrity, explicitly NOT a training-admission override."""
import ast,re,subprocess
from common import *


def main():
    setup();r=read(DOC/'READINESS.json')
    assert r['status']=='BLOCKED' and r['unresolved_engineering_blockers']==['default_entry_candidate_relabel_stable']
    assert r['CPU_tests']==19 and r['real_optimizer_updates']==r['synthetic_optimizer_updates']==0
    assert all(v for k,v in r['gates'].items() if k!='default_entry_candidate_relabel_stable')
    for file in ['GPU_RECEIPT.json','CPU_RECEIPT.json','MECHANISM_RECEIPT.json','INTERFACE_RECEIPT.json','DDIM_RECEIPT.json']:
        assert read(DOC/file)['optimizer_guard']['optimizer_step_calls']==0
    gpu=read(DOC/'GPU_RECEIPT.json')
    for name,value in gpu['network_source_sha256'].items():assert sha(SOURCE/name)==value
    cfg=read(DOC/'FROZEN_CONFIG.json')
    assert not cfg['allow_real_training'] and cfg['pair_objective']=='final_coordinate' and cfg['variant']=='energy_only'
    assert cfg['candidate_seed']==2035 and cfg['initialization_seed']==3201 and cfg['K']==20 and cfg['bridge_layers']==6
    for md in DOC.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if '://' not in target and not target.startswith('#'):assert (md.parent/target.split('#')[0]).exists(),(md,target)
    files=list((SOURCE/'bcep_scene_energy').glob('*.py'))+list((SOURCE/'scripts/ssec_acceptance').glob('*.py'))+[SOURCE/'tests/test_bcep_scene_energy.py']
    for p in files:
        tree=ast.parse(p.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(tree)),p
    assert subprocess.check_output(['git','diff',BASE,'--','bcep','ebjd','bcep_v2','bcep_v3','bcep_joint','bcep_formal','configs'],cwd=SOURCE,text=True)==''
    receipt=dict(delivery_integrity_passed=True,engineering_acceptance='BLOCKED',blocker='default_entry_candidate_relabel_stable',
        training_allowed=False,real_updates=0,synthetic_updates=0,network_hashes_match_GPU_receipt=True,
        no_optimizer_step_call_in_new_code=True,report_links_valid=True)
    write(DOC/'DELIVERY_CHECKS.json',receipt)
    files+=list(p for p in DOC.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json')
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(SOURCE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
