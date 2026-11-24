import itertools
import math
import unittest
import numpy as np
import torch
from ebjd.metrics import hard_metric_statistics
from .math import (rank_minimum_weights, expected_uniform_min, expected_uniform_metrics,
                   endpoint_errors, finite_error_statistics, motion_groups)


class ExpectationTests(unittest.TestCase):
    def test_exhaustive_shared_scene_subsets(self):
        errors = np.array([[0., 4.], [1., 1.], [3., 0.], [2., 3.], [4., 2.]],dtype=np.float32)
        for k in range(1, 6):
            choices = list(itertools.combinations(range(5), k))
            minima = np.array([errors[list(s)].min(0) for s in choices],dtype=np.float64)
            joint = np.array([errors[list(s)].mean(1).min() for s in choices],dtype=np.float64)
            actual = expected_uniform_metrics(errors, k=k)
            np.testing.assert_allclose(actual['agent_min'], minima.mean(0), rtol=0., atol=2e-15)
            self.assertAlmostEqual(actual['L'], minima.mean(), places=14)
            self.assertAlmostEqual(actual['J'], joint.mean(), places=14)
            self.assertEqual(actual['G'], actual['J']-actual['L'])
        # Paired joint-cost decomposition must remain an identity.
        native = finite_error_statistics(errors[[0, 3]])
        uniform = expected_uniform_metrics(errors, k=2)
        self.assertAlmostEqual(native['J']-uniform['J'],
                               native['L']-uniform['L']+native['G']-uniform['G'], places=14)

    def test_rank_law_matches_exhaustive_and_formula(self):
        for m in range(1, 9):
            for k in range(1, m+1):
                counts = np.zeros(m)
                sets = list(itertools.combinations(range(m), k))
                for s in sets: counts[min(s)] += 1
                np.testing.assert_array_equal(rank_minimum_weights(m,k), counts/len(sets))
        w = rank_minimum_weights(256,20)
        self.assertAlmostEqual(w.sum(),1.,places=15)
        self.assertEqual(w[0],20/256)
        self.assertEqual(np.count_nonzero(w),237)
        for r in (1,2,100,237):
            self.assertEqual(w[r-1],math.comb(256-r,19)/math.comb(256,20))

    def test_edges_ties_and_axis(self):
        a=np.array([[3.,1.,1.,7.],[2.,2.,4.,6.]])
        np.testing.assert_allclose(expected_uniform_min(a,k=1,axis=1),a.mean(1),rtol=0,atol=1e-15)
        np.testing.assert_array_equal(expected_uniform_min(a,k=4,axis=1),a.min(1))
        with self.assertRaises(ValueError):rank_minimum_weights(4,5)

    def test_native_finite_endpoint_equivalence(self):
        gen=torch.Generator().manual_seed(712)
        goals=torch.randn(20,13,2,generator=gen)
        truth=torch.randn(13,2,generator=gen)
        valid=torch.tensor([True]*11+[False]*2)
        values=finite_error_statistics(endpoint_errors(goals,truth),valid)
        prediction=goals[:,:,None,:].expand(-1,-1,12,-1)[None].contiguous()
        target=truth[:,None,:].expand(-1,12,-1)[None].contiguous()
        ref=hard_metric_statistics(prediction,target,valid[None])
        np.testing.assert_array_equal(values['agent_min'],ref.agent_minfde[ref.agent_valid].numpy())
        self.assertEqual(values['J'],float(ref.scene_jfde[0]))
        self.assertEqual(values['L'],float(ref.agent_minfde[ref.agent_valid].double().mean()))
        self.assertEqual(values['agent_min_sum'],float(ref.agent_minfde[ref.agent_valid].double().sum()))

    def test_speed_path_length_groups_and_boundary(self):
        # End-to-end displacement alone is insufficient: the last person returns.
        x=torch.zeros(3,8,2,dtype=torch.float64)
        x[1,-1,0]=.2*2.8
        x[2,1:4,0]=1.
        result=motion_groups(x)
        self.assertEqual(result['low_count'],2)
        self.assertEqual(result['groups'],['all','mixed','mixed_N2_8'])
        self.assertAlmostEqual(result['speeds_m_s'][2],2./2.8)
        self.assertEqual(motion_groups(x[:2])['groups'],['all','all_low_speed'])
        moving=x[2:].repeat(2,1,1)
        self.assertEqual(motion_groups(moving)['groups'],['all','all_moving','all_moving_N2_8'])
        self.assertEqual(motion_groups(moving[:1])['groups'],['all','all_moving'])
        self.assertEqual(motion_groups(moving.repeat(5,1,1))['groups'],['all','all_moving'])


if __name__=='__main__':unittest.main()
