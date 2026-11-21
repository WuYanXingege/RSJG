"""Report logic only: no model, optimizer, inference or input data loading."""
import copy, importlib.util, sys
from pathlib import Path
import pytest

FOLDER=Path(__file__).resolve().parents[1]/'scripts/ssec_trial'
sys.path.insert(0,str(FOLDER))
from analysis import decision, select_rank, paired, summarize
from common import LIMITS


def test_selection_uses_jade_and_earlier_ties_only():
    h=[dict(epoch=5,validation=.4,JFDE=.1),dict(epoch=2,validation=.4,JFDE=.9),dict(epoch=1,validation=.5,JFDE=.01)]
    assert select_rank(h)['epoch']==2


def test_gate_preserves_lte_and_strict_lt():
    assert decision(LIMITS)['passes']==[True,False,False]
    assert decision([LIMITS[0],LIMITS[1]-1e-10,LIMITS[2]-1e-10])['passes']==[True,True,True]


def fixture():
    old=[];new=[]
    for i,n in enumerate([1]*89+[2]*374+[9]*127):
        old.append(dict(index=i,n_agents=n,H_JADE=.4,I_JFDE=.8,H_JFDE=.7,feedback_gain=.1,minFDE=.3,oracle=.3))
        new.append(dict(index=i,n_agents=n,H_JADE=.39,I_JFDE=.79,H_JFDE=.72,feedback_gain=.07,G_JFDE=.9,minFDE_agent_micro=.3,minFDE_scene=.3))
    return new,old


def test_paired_identity_and_feedback_decomposition():
    n,o=fixture();p=paired(n,o);s=summarize(p)
    assert len(p)==590 and 'index' not in p[0] and p[0]['window']=='w0001'
    assert next(r for r in s if r['group']=='N_ge_9' and r['metric']=='H_JFDE')['regressed']==127
    assert next(r for r in s if r['group']=='all' and r['metric']=='feedback_gain')['regressed']==590


def test_changed_population_or_marginal_rejected():
    n,o=fixture();n[0]['minFDE_scene']=.4
    with pytest.raises(AssertionError):paired(n,o)
    n,o=fixture();n[0]['n_agents']=2
    with pytest.raises(AssertionError):paired(n,o)


def test_old_reference_hash_and_frozen_candidates():
    spec=importlib.util.spec_from_file_location('ssec_trial_report_only',FOLDER/'report.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=m.source_check();old,meta=m.reference(c)
    assert len(old)==590 and meta['identity']['candidate_seed']==2035
