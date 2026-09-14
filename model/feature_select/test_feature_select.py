r"""Pruebas del ranking, causalidad y comparación.

Desde la raíz del proyecto:
    .\.venv\Scripts\python.exe -m unittest discover -s my_solution\model\feature_select -p "test_*.py" -v
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from features import (
    CausalFeatureBuilder, TemporalFeature, build_sequence, load_rankings,
    select_temporal_features,
)
from train_gru import comparison_row, normalize, select_variant_specs
from utils import FEATURE_COLUMNS, TARGET_COLUMNS


class FeatureSelectionTests(unittest.TestCase):
    def test_batch_and_online_match_and_reset_at_sequence_boundary(self):
        specs = [
            TemporalFeature("i0_p3", "lag", 1),
            TemporalFeature("i0_v4", "delta", 5),
            TemporalFeature("i0_p3", "lag", 50),
            TemporalFeature("i0_v4", "delta", 50),
        ]
        builder = CausalFeatureBuilder(specs)
        for seq_ix, offset in ((11, 0), (12, 1000)):
            raw = np.zeros((60, len(FEATURE_COLUMNS)), dtype=np.float32)
            raw[:, FEATURE_COLUMNS.index("i0_p3")] = np.arange(60) + offset
            raw[:, FEATURE_COLUMNS.index("i0_v4")] = 2 * np.arange(60) + offset
            batch = build_sequence(raw, specs)
            online = np.vstack([
                builder.push(seq_ix, step, raw[step]) for step in range(len(raw))
            ])
            np.testing.assert_allclose(batch, online, equal_nan=True)
            self.assertTrue(np.isnan(batch[0, len(FEATURE_COLUMNS):]).all())
            self.assertTrue(np.isnan(batch[:50, -2:]).all())
            self.assertTrue(np.isfinite(batch[50:, -2:]).all())

    def test_online_rejects_missing_or_out_of_order_steps(self):
        builder = CausalFeatureBuilder([TemporalFeature("i0_p3", "lag", 1)])
        row = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
        with self.assertRaises(ValueError):
            builder.push(1, 1, row)
        builder.push(1, 0, row)
        with self.assertRaises(ValueError):
            builder.push(1, 2, row)
        with self.assertRaises(ValueError):
            builder.push(2, 1, row)

    def test_rankings_are_per_target_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rank.csv"
            fields = ("rank", "feature", "base_feature", "transform", "lag", "target")
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for target in TARGET_COLUMNS:
                    for rank in range(1, 21):
                        base = FEATURE_COLUMNS[rank - 1] if target == "t0" else FEATURE_COLUMNS[rank]
                        spec = TemporalFeature(base, "lag", 1)
                        writer.writerow({"rank": rank, "feature": spec.name,
                                         "base_feature": base, "transform": "lag",
                                         "lag": 1, "target": target})
            ranking = load_rankings(path)
            self.assertEqual(select_temporal_features(ranking, 0), [])
            self.assertEqual(len(select_temporal_features(ranking, 3)), 4)
            self.assertEqual(len(select_temporal_features(ranking, 5)), 6)
            self.assertEqual(len(select_temporal_features(ranking, 10)), 11)
            self.assertEqual(len(select_temporal_features(ranking, 20)), 21)
            self.assertTrue(all(spec.name not in TARGET_COLUMNS
                                for spec in select_temporal_features(ranking, 20)))
            extra = ranking["t0"][4]
            selected = select_variant_specs(ranking, 3, (extra,))
            self.assertEqual(len(selected), 5)
            self.assertEqual(selected[-1], extra)
            self.assertEqual(select_variant_specs(ranking, 0, (extra,)), [extra])
            with self.assertRaises(ValueError):
                select_variant_specs(ranking, 3, (ranking["t0"][0],))

    def test_normalizer_only_replaces_missing_model_inputs(self):
        x = np.array([[np.nan, 5.0], [2.0, 7.0]], dtype=np.float32)
        normalized = normalize(x, np.array([2.0, 5.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.isnan(x[0, 0]))  # No modificar el array fuente.

    def test_comparison_delta_uses_same_base_for_each_target(self):
        base = {"t0": 0.2, "t1": 0.3, "weighted_pearson": 0.25}
        augmented = {"t0": 0.24, "t1": 0.28, "weighted_pearson": 0.26,
                     "eligible_blocks": 10}
        row = comparison_row("top_5", 5, 10, augmented, base, 10)
        self.assertAlmostEqual(row["delta_vs_base_t0"], 0.04)
        self.assertAlmostEqual(row["delta_vs_base_t1"], -0.02)
        self.assertAlmostEqual(row["delta_vs_base_mean"], 0.01)
        without_base = comparison_row("solo_modelo", 3, 7, augmented, None, 10)
        self.assertIsNone(without_base["delta_vs_base_mean"])


if __name__ == "__main__":
    unittest.main()
