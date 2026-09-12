"""Tests for deterministic row-group and row sampling."""

import unittest

from profile_variables import sample_counts, select_row_groups


class SelectionTests(unittest.TestCase):
    def test_random_selection_is_reproducible(self):
        first = select_row_groups(100, 10, seed=2026, first=False)
        self.assertEqual(first, select_row_groups(100, 10, seed=2026, first=False))
        self.assertEqual(len(set(first)), 10)
        self.assertEqual(first, sorted(first))

    def test_first_n_and_bounds(self):
        self.assertEqual(select_row_groups(8, 3, seed=0, first=True), [0, 1, 2])
        with self.assertRaises(ValueError):
            select_row_groups(8, 9, seed=0, first=True)

    def test_balanced_row_budget(self):
        self.assertEqual(sample_counts(3, 10), [4, 3, 3])
        self.assertEqual(sample_counts(1, 100_000), [20_000])
        with self.assertRaises(ValueError):
            sample_counts(3, 2)


if __name__ == "__main__":
    unittest.main()
