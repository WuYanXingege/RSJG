"""Read-only combinatorial tests. No network, backward, or optimizer."""
import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/v3_frozen_audit'))
from matching import solve,value,second_best_gap,center,assignment_change,gap_bin


@pytest.mark.parametrize('seed',[0,4,21])
def test_second_best_matches_exhaustive_distinct_permutations(seed):
    s=np.random.default_rng(seed).normal(size=(4,4)).astype(np.float32)
    pi=solve(s);gap,runner=second_best_gap(s,pi)
    values=sorted((value(s,np.array(p)) for p in itertools.permutations(range(4))),reverse=True)
    assert gap==values[0]-values[1]
    assert value(s,runner)==values[1] and not np.array_equal(pi,runner)


def test_strict_tie_and_small_unique_gap_are_distinct():
    s=np.zeros((4,4),np.float32);pi=solve(s)
    assert second_best_gap(s,pi)[0]==0
    assert gap_bin(0)=='strict_tie_0'
    s=np.eye(4,dtype=np.float32)*1e-7
    gap,_=second_best_gap(s,solve(s))
    assert 0<gap<=1e-6 and gap_bin(gap)=='near_(0,1e-6]'


def test_common_world_relabel_vs_changed_correspondence():
    a=np.tile(np.arange(4)[:,None],(1,3));b=a[[2,0,3,1]]
    result=assignment_change(a,b)
    assert result['changed_choices']==12 and result['uniform_world_relabel_only']
    b=b.copy();b[[0,1],1]=b[[1,0],1]
    result=assignment_change(a,b)
    assert not result['uniform_world_relabel_only'] and result['relative_changed_choices']>0


def test_row_column_offsets_do_not_change_assignment_objective_differences():
    s=np.random.default_rng(9).normal(size=(4,4))
    p=np.arange(4);q=p[::-1]
    assert value(s,p)-value(s,q)==pytest.approx(value(center(s),p)-value(center(s),q),abs=1e-14)


def test_strict_ties_can_fail_permutation_equivariance():
    s=np.zeros((4,4),np.float32);p=solve(s)
    columns=np.array([1,2,3,0]);q=columns[solve(s[:,columns])]
    assert not np.array_equal(p,q) and value(s,p)==value(s,q)
