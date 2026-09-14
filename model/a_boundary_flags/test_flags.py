"""Ejecutar manualmente: python -m unittest my_solution.model.a_boundary_flags.test_flags"""

import unittest

import numpy as np

from .flags import A_INDICES, BOUNDARY, FEATURE_NAMES, boundary_flags, model_inputs
from utils import FEATURE_COLUMNS


class BoundaryFlagTests(unittest.TestCase):
    def test_min_max_and_original_values(self):
        rows = np.zeros((3, len(FEATURE_COLUMNS)), dtype=np.float32)
        rows[0, A_INDICES[5]] = np.float32(-BOUNDARY)
        rows[0, A_INDICES[6]] = np.float32(BOUNDARY)
        rows[1, A_INDICES[5]] = -5.198
        original = rows.copy()
        flags = boundary_flags(rows)
        self.assertEqual(flags.shape, (3, 16))
        self.assertEqual(flags[0, FEATURE_NAMES.index("a5_is_min")], 1)
        self.assertEqual(flags[0, FEATURE_NAMES.index("a5_is_max")], 0)
        self.assertEqual(flags[0, FEATURE_NAMES.index("a6_is_max")], 1)
        self.assertEqual(flags[1, FEATURE_NAMES.index("a5_is_min")], 0)
        np.testing.assert_array_equal(rows, original)
        np.testing.assert_array_equal(flags, np.stack([boundary_flags(row) for row in rows]))
        inputs = model_inputs(rows, np.zeros(112, dtype=np.float32),
                              np.ones(112, dtype=np.float32))
        np.testing.assert_array_equal(inputs[:, :112], original)
        np.testing.assert_array_equal(inputs[:, 112:], flags)


if __name__ == "__main__":
    unittest.main()
