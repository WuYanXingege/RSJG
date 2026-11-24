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

if __name__=='__main__':unittest.main()
