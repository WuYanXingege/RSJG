"""Algebra/invariance tests only; no optimizer or real data."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/joint_score_audit'))
from energy_prototype import SceneEnergy,synthetic


class EnergyTests(unittest.TestCase):
    def test_exact_candidate_replacement(self):
        model,p=synthetic(n=5,k=6);base=model.world_energy(p)
        for i in range(model.n):
            s=model.scores(p,i)
            for w in range(model.k):
                for a in range(model.k):
                    q=p.copy();q[i,w]=a
                    self.assertAlmostEqual(s[w,a],-(model.world_energy(q)[w]-base[w]),places=11)

    def test_low_rank_global_is_same_complete_pair_energy(self):
        model,p=synthetic(n=5,k=6);model.tables[:]=0;x=model.selected(p)
        brute=sum(np.square(x[i]-x[j]).sum(-1) for i in range(5) for j in range(i+1,5))/4
        np.testing.assert_allclose(model.world_energy(p),brute,atol=1e-12)

    def test_sweep_keeps_bijection_and_decreases_one_common_energy(self):
        model,p=synthetic();q,trace=model.sweep(p);model.validate(q)
        self.assertTrue(any(r['accepted'] for r in trace))
        self.assertTrue(all(r['energy_after']<=r['energy_before'] for r in trace))
        for i in range(model.n):np.testing.assert_array_equal(np.sort(model.z[i,p[i]],axis=0),np.sort(model.z[i,q[i]],axis=0))

    def test_scores_are_not_row_column_bias(self):
        model,p=synthetic();s=model.scores(p,0);c=s-s.mean(0)-s.mean(1)[:,None]+s.mean()
        self.assertGreater(np.linalg.norm(c),1.)
        # Row/column additions change all complete assignment totals equally.
        rng=np.random.default_rng(8);row=rng.normal(size=20);col=rng.normal(size=20);q=rng.permutation(20)
        self.assertAlmostEqual((s+row[:,None]+col)[np.arange(20),q].sum()-s[np.arange(20),q].sum(),row.sum()+col.sum(),places=11)

    def test_candidate_relabel_with_carried_incumbent(self):
        m,p=synthetic(n=5,k=6);rng=np.random.default_rng(9)
        orders=np.stack([rng.permutation(6) for _ in range(5)]);inverse=np.argsort(orders,axis=1)
        z=np.stack([m.z[i,orders[i]] for i in range(5)])
        tables=np.stack([m.tables[e][orders[i]][:,orders[j]] for e,(i,j) in enumerate(m.edges)])
        renamed=SceneEnergy(z,m.edges,tables);pp=np.stack([inverse[i,p[i]] for i in range(5)])
        a,_=m.sweep(p);b,_=renamed.sweep(pp);back=np.stack([orders[i,b[i]] for i in range(5)])
        np.testing.assert_array_equal(a,back)

    def test_global_world_relabel(self):
        m,p=synthetic(n=5,k=6);order=np.array([2,4,1,5,0,3]);a,_=m.sweep(p);b,_=m.sweep(p[:,order])
        np.testing.assert_array_equal(a[:,order],b)

    def test_strict_tie_abstains(self):
        m,p=synthetic(n=3,k=6);m.z[:]=0;m.tables[:]=0;q,trace=m.sweep(p)
        np.testing.assert_array_equal(p,q);self.assertTrue(all(r['strict_tie'] and not r['accepted'] for r in trace))

    def test_simultaneous_neighbor_updates_can_conflict(self):
        # Both agents individually propose swapping to anti-align; both together undo it.
        m=SceneEnergy(np.zeros((2,2,1)),np.array([[0,1]]),np.array([[[2.,0.],[0.,2.]]]))
        p=np.array([[0,1],[0,1]]);a,_=m.step(p,0);b,_=m.step(p,1)
        jacobi=np.stack([a[0],b[1]]);gauss,trace=m.sweep(p)
        self.assertEqual(m.world_energy(jacobi).sum(),m.world_energy(p).sum())
        self.assertLess(m.world_energy(gauss).sum(),m.world_energy(jacobi).sum())

    def test_singleton_identity_preserves_marginal(self):
        m,p=synthetic(n=1,k=20);q,trace=m.sweep(p)
        np.testing.assert_array_equal(p,q);self.assertEqual(trace,[])

    def test_known_gauge_null_space_is_not_numerical_rank_twenty(self):
        from common import effective_singular_values
        rng=np.random.default_rng(10);s=rng.normal(size=(20,1))*.05+rng.normal(size=(1,20))*.05+rng.normal(size=(20,20))*1e-7
        sv=effective_singular_values(s)
        self.assertEqual(sv[-1],0.)
        self.assertEqual(np.count_nonzero(sv),19)


if __name__=='__main__':unittest.main()
