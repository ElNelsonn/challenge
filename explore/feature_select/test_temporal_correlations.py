"""Pruebas de límites de secuencia y conteo de pares válidos.

Desde la raíz del proyecto:
    .\.venv\Scripts\python.exe -m unittest discover -s my_solution\explore\feature_select -p "test_*.py" -v
"""

import unittest

import numpy as np

from temporal_correlations import correlations, lag_within_sequences


class TemporalCorrelationTests(unittest.TestCase):
    def test_lag_never_crosses_sequence_boundary(self):
        values = np.array([1, 2, 3, 4, 100, 200, 300, 400], dtype=float)
        lagged = lag_within_sequences(values, 1, 4)
        np.testing.assert_allclose(lagged, [np.nan, 1, 2, 3, np.nan, 100, 200, 300],
                                   equal_nan=True)
        delta = values - lagged
        self.assertTrue(np.isnan(delta[0]) and np.isnan(delta[4]))
        self.assertEqual(delta[5], 100)

    def test_each_lag_has_its_own_valid_rows(self):
        values = np.arange(12, dtype=float)
        target = values.copy()
        for lag in (1, 2, 3):
            lagged = lag_within_sequences(values, lag, 6)
            _, _, n_valid = correlations(lagged, target)
            self.assertEqual(n_valid, 12 - 2 * lag)

    def test_nonfinite_values_are_excluded_without_imputation(self):
        x = np.array([np.nan, 1, 2, 3, np.inf, 5], dtype=float)
        y = np.array([0, 2, 4, np.nan, 8, 10], dtype=float)
        pearson, spearman, n_valid = correlations(x, y)
        self.assertEqual(n_valid, 3)
        self.assertAlmostEqual(pearson, 1.0)
        self.assertAlmostEqual(spearman, 1.0)


if __name__ == "__main__":
    unittest.main()
