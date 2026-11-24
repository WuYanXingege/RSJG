import unittest
import numpy as np
from .math import coverage_cost, rank_positive_swaps, directed_distance


class SingleSwapTests(unittest.TestCase):
    @staticmethod
    def brute(raw, ids, valid=None):
        p0 = coverage_cost(raw, ids, valid)
        rows = []
        for slot, out_id in enumerate(ids):
            for in_id in sorted(set(range(len(raw))) - set(ids)):
                chosen = np.array(ids, dtype=np.int64)
                chosen[slot] = in_id
                p = coverage_cost(raw, chosen, valid)
                if p0-p > 0.:
                    rows.append(dict(slot=slot, out_id=int(out_id), in_id=in_id, p=p, gain=p0-p))
        rows.sort(key=lambda row: (-row['gain'], row['out_id'], row['in_id']))
        return dict(p0=p0, total_swaps=len(ids)*(len(raw)-len(ids)), swaps=rows)

    def test_every_swap_matches_direct_fp64_reduction(self):
        rng = np.random.default_rng(3201)
        for m, n, k in [(9, 3, 4), (7, 1, 1), (7, 4, 6), (5, 2, 5)]:
            raw = rng.normal(size=(m, n, 2)).astype(np.float32)
            ids = rng.permutation(m)[:k]
            valid = np.arange(n) != (n-1) if n > 1 else np.ones(n, dtype=bool)
            expected = self.brute(raw, ids, valid)
            for chunk in (1, 3, 64):
                self.assertEqual(rank_positive_swaps(raw, ids, valid, chunk), expected)

    def test_exact_ties_use_raw_id_order_not_slot(self):
        raw = np.zeros((4, 1, 2), dtype=np.float64)
        raw[:, 0, 0] = np.arange(4)
        ids = np.array([1, 0])
        result = rank_positive_swaps(raw, ids)
        self.assertEqual(result, self.brute(raw, ids))
        self.assertEqual(result['swaps'][0]['out_id'], 0)
        self.assertEqual(result['swaps'][0]['in_id'], 2)
        self.assertEqual(result['swaps'][0]['slot'], 1)
        self.assertEqual(result['swaps'][0]['gain'], .25)
        self.assertEqual(rank_positive_swaps(np.zeros((4, 2, 2)), ids)['swaps'], [])

    def test_duplicate_nearest_and_masked_nan(self):
        raw = np.array([[[0., 0.]], [[0., 0.]], [[1., 0.]], [[2., 0.]], [[4., 0.]]])
        raw = np.concatenate([raw, np.full_like(raw, np.nan)], axis=1)
        ids = np.array([1, 0, 4])
        valid = np.array([True, False])
        self.assertEqual(rank_positive_swaps(raw, ids, valid), self.brute(raw, ids, valid))

    def test_direction_and_mean_after_norm(self):
        old = np.array([[[0., 0.], [0., 0.]], [[1., 0.], [1., 0.]]])
        new = old[:1]
        self.assertEqual(directed_distance(old, new), 1.)
        self.assertEqual(directed_distance(new, old), 0.)
        paired = np.array([[[1., 0.], [-1., 0.]]])
        self.assertEqual(directed_distance(old[:1], paired), 1.)

    def test_one_sided_jfde_protection(self):
        rng = np.random.default_rng(2035)
        old = rng.normal(size=(20, 13, 2))
        new = old + rng.normal(size=old.shape)*.001
        new = new[rng.permutation(20)]
        h = directed_distance(old, new)
        self.assertLess(h, .005)
        for _ in range(100):
            truth = rng.normal(size=(13, 2))
            old_j = np.linalg.norm(old-truth, axis=-1).mean(axis=1).min()
            new_j = np.linalg.norm(new-truth, axis=-1).mean(axis=1).min()
            self.assertLessEqual(new_j-old_j, h+1e-14)
        self.assertEqual(directed_distance(old, old[::-1]), 0.)


if __name__ == '__main__':
    unittest.main()
