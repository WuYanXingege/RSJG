"""Cross-file consistency checks, without models, data loaders, or optimizer calls."""
import re
from common import *


def main():
    setup();checks=[]
    matrices=csvread(DOC/'SCORE_MATRICES.csv');sv=csvread(DOC/'SINGULAR_VALUES.csv');soft=csvread(DOC/'SOFT_HARD.csv')
    assert len(matrices)==len(sv)==len(soft)==3*2*4262
    assert len(csvread(DOC/'SOFT_ROW_COLUMN_DISTRIBUTIONS.csv'))==3*2*4262
    assert len(csvread(DOC/'FEATURES.csv'))==1770 and len(csvread(DOC/'RELABEL.csv'))==3*2*4173
    for r in matrices:
        assert int(r['effective_rank64'])<=19
        if int(r['n_agents'])==1:assert r['gap_class']=='singleton' and r['gap']==''
        else:assert float(r['gap'])>=0 and r['gap_class'] in ['strict','near']
    for r in sv:
        vals=[float(r['sigma_'+str(i)]) for i in range(1,21)]
        assert vals[-1]==0 and all(a>=b for a,b in zip(vals,vals[1:]))
    checks.append('25572 matrix/spectrum/soft rows; known zero-sum rank <=19 and gap classifications consistent')
    for r in soft:
        if int(r['n_agents'])==1:
            assert float(r['hard_soft_row_TV'])==0 and float(r['entropy_mean'])==0
        else:
            assert float(r['entropy_fraction_of_logK'])>.999999
            assert float(r['row_sum_max_error'])<1e-6 and float(r['column_sum_max_error'])<1e-6
    checks.append('singleton identity and multi-agent native Sinkhorn row/column marginals verified')
    indices=None
    for model in MODELS:
        rr=csvread(DOC/f'GRADIENTS_{model}.csv')
        assert len(rr)==36
        ix=[r['index'] for r in rr]
        if indices is None:indices=ix
        assert indices==ix
        assert all(sum(r['people_bin']==g for r in rr)==6 for g in ['1','2-4','5-8','9-12','13-20','21+'])
        for r in rr:
            assert int(r['initial_pi_changed_vs_inference'])==int(r['final_pi_changed_vs_inference'])==0
            assert float(r['initial_score_max_change_vs_inference'])==float(r['final_score_max_change_vs_inference'])==0
            if int(r['n_agents'])==1:assert float(r['g_final_score_norm'])==0
            else:
                assert float(r['g_final_score_norm'])>0
                assert abs(float(r['native_sinkhorn_gain'])-.05)<1e-5
        proof=read(DOC/f'GRADIENT_PROOF_{model}.json')
        assert proof['unchanged'] and proof['optimizer_guard']==GUARD
        cp=read(DOC/f'COLLECTION_PROOF_{model}.json')
        assert cp['weights_unchanged'] and cp['optimizer_guard']==GUARD
        if model.startswith('v2'):assert cp['historical_pi_exact_windows']==590 and cp['historical_pi_different_windows']==0
    checks.append('108 native gradient-mode forwards match cached scores and permutations exactly; state hashes unchanged; zero optimizer calls')
    for r in csvread(DOC/'RELABEL.csv'):
        assert float(r['original_minus_remapped_score'])==0
        if int(r['changed_choices']):assert r['gap_class']=='strict'
    checks.append('all relabel differences are equal-objective strict ties')
    p=read(DOC/'PROTOTYPE_CHECK.json');a=read(DOC/'ASSET_RECEIPT.json')
    assert p['N']==57 and p['edges']==1596 and p['optimizer_guard']==GUARD and p['final_energy']<=p['initial_energy']
    assert a['all_protected_assets_byte_identical'] and a['real_optimizer_steps']==a['synthetic_optimizer_steps']==0
    assert a['original_tracked_source_diff']==''
    for md in DOC.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if '://' not in target and not target.startswith('#'):assert (md.parent/target.split('#')[0]).exists(),(md,target)
    checks.append('57-person CPU prototype, protected assets, and report links checked')
    write(DOC/'DELIVERY_CHECKS.json',dict(passed=True,checks=checks,optimizer_guard=GUARD))
    files=[p for p in DOC.iterdir() if p.is_file() and p.name!='ARTIFACT_SHA256.json']
    files+=list((SOURCE/'scripts/joint_score_audit').glob('*.py'))+[SOURCE/'tests/test_joint_scene_energy_prototype.py']
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(SOURCE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
