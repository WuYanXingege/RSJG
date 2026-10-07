#!/usr/bin/env python3
import unittest

import numpy as np

from cpu_math import decompose_uniform, reconstruct, sample_jensen_gap


class DiagnosticMathReferences(unittest.TestCase):
    def test_additive_cost_has_zero_interaction_and_reconstructs(self):
        a = np.array([-1.0, 0.5, 2.0])
        b = np.array([0.2, -0.3])
        cost = a[:, None] + b[None, :]
        interaction, row_term, col_term, _ = decompose_uniform(cost)
        self.assertLess(np.abs(interaction).max(), 1e-12)
        np.testing.assert_allclose(reconstruct(interaction, row_term, col_term), cost)

    def test_interaction_is_double_centered(self):
        cost = np.array([[0.0, 2.0], [3.0, -1.0]])
        interaction, a, b, _ = decompose_uniform(cost)
        np.testing.assert_allclose(interaction.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(interaction.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(reconstruct(interaction, a, b), cost)

    def test_common_logit_shift_has_zero_gap(self):
        base = np.array([0.1, -0.2, 1.3])
        shifts = np.array([-4.0, 0.0, 2.0, 7.0])
        logits = base[None, :] + shifts[:, None]
        self.assertAlmostEqual(sample_jensen_gap(logits), 0.0, places=12)

    def test_candidate_dependent_variation_has_positive_gap(self):
        logits = np.array([[4.0, 0.0], [0.0, 4.0]])
        self.assertGreater(sample_jensen_gap(logits), 0.0)


if __name__ == "__main__":
    unittest.main()
