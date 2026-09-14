"""Pruebas de selección de grupos nuevos para las cinco comparaciones."""

import unittest

from train_gru_five_group_pairs import choose_new_groups


class GroupSelectionTests(unittest.TestCase):
    def test_rounds_are_disjoint_and_avoid_previous_groups(self):
        used_train = {0, 1, 2, 3, 4}
        used_valid = {0, 1}
        rounds, valid = choose_new_groups(
            40, 15, used_train, used_valid, selection_seed=2027,
            rounds=3, train_per_round=4, valid_count=5,
        )
        self.assertEqual([len(groups) for groups in rounds], [4, 4, 4])
        all_train = [group for groups in rounds for group in groups]
        self.assertEqual(len(all_train), len(set(all_train)))
        self.assertFalse(set(all_train) & used_train)
        self.assertEqual(len(valid), 5)
        self.assertFalse(set(valid) & used_valid)
        repeat = choose_new_groups(
            40, 15, used_train, used_valid, selection_seed=2027,
            rounds=3, train_per_round=4, valid_count=5,
        )
        self.assertEqual((rounds, valid), repeat)

    def test_fails_if_not_enough_unused_groups(self):
        with self.assertRaises(ValueError):
            choose_new_groups(10, 10, set(range(7)), set(), 2027,
                              rounds=2, train_per_round=2, valid_count=5)
        with self.assertRaises(ValueError):
            choose_new_groups(10, 10, set(), set(range(8)), 2027,
                              rounds=2, train_per_round=2, valid_count=3)


if __name__ == "__main__":
    unittest.main()
