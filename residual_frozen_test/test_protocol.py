import tempfile
import unittest
from pathlib import Path
import numpy as np
from .common import arrays,groups_for,validate_row
from .run import observation_guard
from bcep.contracts import digest


class ProtocolTest(unittest.TestCase):
    def test_observed_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.npz';np.savez(p,observed=np.zeros((1,8,2)),future=np.ones((1,12,2)))
            with observation_guard(),np.load(p) as z:
                self.assertEqual(z['observed'].shape,(1,8,2))
                with self.assertRaises(RuntimeError):z['future']
            with np.load(p) as z:self.assertEqual(z['future'].shape,(1,12,2))

    def test_shard_tamper_and_joint_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pair.npz'
            payload=dict(B=np.zeros((20,2,12,2),np.float32),Bstar=np.ones((20,2,12,2),np.float32),
                         raw_ids=np.arange(20,dtype=np.int64),valid=np.ones(2,dtype=bool))
            np.savez(p,**payload)
            expected=dict(file='pair.npz',n=2)
            row=dict(expected,sha256=digest(p),ddim_calls=1,ablation=False)
            validate_row(row,expected,Path(d))
            payload['raw_ids'][1]=0;np.savez(p,**payload)
            with self.assertRaises(AssertionError):validate_row(row,expected,Path(d))
            with self.assertRaises(AssertionError):arrays(p,2)

    def test_group_boundaries(self):
        for n,g in [(1,'N_eq_1'),(2,'N_2_to_8'),(8,'N_2_to_8'),(9,'N_ge_9')]:
            self.assertEqual(groups_for(n),('all',g))


class SeedProtocolTest(unittest.TestCase):
    def test_paths_and_pair_sd(self):
        from .common import expected_row
        from .statistics import paired_summary
        from statistics import stdev
        split={'members':[{'agent_ids':['a']}]}
        a=expected_row('zara1',2036,1,0,split,{})
        b=expected_row('zara1',2037,1,0,split,{})
        self.assertNotEqual(a['file'],b['file']);self.assertNotEqual(a['seed'],b['seed'])
        rows=[]
        for k in range(5):
            rows.append(dict(fold='x',group='all',fold_count=1,windows=1,agents=1,metric='JFDE',seed=2035+k,
                B_m=10.*k,Bstar_m=10.*k-(k+1),delta_cm=-100.*(k+1),four_metrics_improved=True))
        result=paired_summary(rows)
        self.assertEqual(result['delta_std_cm'],stdev([-100.,-200.,-300.,-400.,-500.]))
        self.assertEqual(result['improved'],5)

    def test_three_noise_streams_repeat_and_change(self):
        import torch
        from types import SimpleNamespace
        from .random_audit import regenerate,fingerprint
        obs=torch.zeros(1,2,8,2);obs[0,1,:,0]=1.
        batch=SimpleNamespace(observed=obs,scene_ids=['fixture'],agent_ids=[['a','b']])
        ids=torch.arange(20)
        a=regenerate(batch,ids,2036);b=regenerate(batch,ids,2037);repeat=regenerate(batch,ids,2036)
        for k in a:
            self.assertGreater(a[k].numel(),0)
            self.assertEqual(fingerprint(a[k]),fingerprint(repeat[k]))
            self.assertNotEqual(fingerprint(a[k]),fingerprint(b[k]))
        moved=regenerate(batch,ids.flip(0),2036)
        self.assertTrue(torch.equal(moved['bridge'],a['bridge'][:,torch.arange(19,-1,-1)]))

if __name__=='__main__':unittest.main()
