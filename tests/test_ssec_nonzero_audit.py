"""Audit harness tests. These tests execute NO optimizer steps."""
import copy
import json
import numpy as np
import pytest
import torch
from ssec_nonzero.common import exact, frozen
from ssec_nonzero.audit import committed_journal, semantic_journal


def test_exact_detects_one_ulp_and_dtype_changes():
    a = {'tensor':torch.tensor([1.],dtype=torch.float32),'array':np.array([2.]),'nested':[1,2]}
    exact(a,copy.deepcopy(a))
    b=copy.deepcopy(a);b['tensor']=torch.nextafter(b['tensor'],torch.tensor([2.]))
    with pytest.raises(AssertionError):exact(a,b)
    b=copy.deepcopy(a);b['array']=b['array'].astype(np.float32)
    with pytest.raises(AssertionError):exact(a,b)


def test_exact_does_not_skip_unknown_checkpoint_fields():
    with pytest.raises(AssertionError):exact({'a':1,'unexpected_statistic':2},{'a':1,'unexpected_statistic':3})
    with pytest.raises(AssertionError):exact({'a':1},{'a':1,'extra':2})


def test_replay_preserves_attempts_and_committed_horizon():
    rows=[dict(update=1,session='a'),dict(update=2,session='a'),dict(update=3,session='crash'),
          dict(update=3,session='recovered'),dict(update=4,session='recovered')]
    assert [r['update'] for r in committed_journal(rows[:3],2)]==[1,2]
    result=committed_journal(rows,4)
    assert result[2]['session']=='recovered' and len(rows)==5
    with pytest.raises(AssertionError,match='missing'):committed_journal(rows[:3],4)


def test_numeric_journal_mismatch_cannot_be_hidden_as_timing():
    a=dict(update=1,epoch=1,modules={'phi':{'parameter_delta_l2':.1}},physical_groups=[],unix=1)
    b=copy.deepcopy(a);b['unix']=2
    exact(semantic_journal(a),semantic_journal(b))
    b['modules']['phi']['parameter_delta_l2']=.2
    with pytest.raises(AssertionError):exact(semantic_journal(a),semantic_journal(b))


def test_exact_frozen_network_config_and_recipe():
    c=frozen()
    assert c['config']['pair_objective']=='final_coordinate'
    assert c['recipe']['logical_scenes']==16 and c['recipe']['seed']==3201
    assert c['config']['batch_scenes']==4 and c['config']['gradient_accumulation']==4
    assert c['config']['group_max_slots']==32


def test_no_extra_step_phase_or_formal_entry():
    from ssec_nonzero.worker import PHASES
    assert sum(v[3] for v in PHASES.values())==11
    assert all(v[2] in [2,4] for v in PHASES.values())
    assert set(v[0] for v in PHASES.values())=={'continuous','split','replay'}
