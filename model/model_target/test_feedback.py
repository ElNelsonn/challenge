"""Pruebas unitarias; ejecutar manualmente: python -m unittest my_solution.model.model_target.test_feedback"""

from __future__ import annotations

import unittest

import numpy as np

from .feedback import FEATURE_NAMES, LAGS, PredictionHistory, feedback_from_predictions


class FeedbackTests(unittest.TestCase):
    def test_lags_and_deltas_use_only_earlier_predictions(self):
        scale = np.array([2.0, 4.0], dtype=np.float32)
        history = PredictionHistory(scale)
        predictions = np.column_stack(((np.arange(12) + 1) / 10,
                                       -(np.arange(12) + 1) / 20)).astype(np.float32)
        observed = []
        for prediction in predictions:
            observed.append(history.features())
            history.append(prediction)
        actual = np.vstack(observed)
        self.assertEqual(actual.shape, (12, 16))
        np.testing.assert_array_equal(actual[0], np.zeros(16))
        self.assertEqual(actual[1, FEATURE_NAMES.index("t0_pred_lag_1")], predictions[0, 0] / 2)
        self.assertEqual(actual[5, FEATURE_NAMES.index("t0_pred_lag_5")], predictions[0, 0] / 2)
        self.assertEqual(actual[10, FEATURE_NAMES.index("t1_pred_lag_10")], predictions[0, 1] / 4)
        self.assertEqual(actual[10, FEATURE_NAMES.index("t0_pred_delta_10")], 0)
        self.assertAlmostEqual(actual[11, FEATURE_NAMES.index("t0_pred_delta_10")],
                               float((predictions[10, 0] - predictions[0, 0]) / 2))
        np.testing.assert_allclose(actual, feedback_from_predictions(predictions, scale))

    def test_new_sequence_has_no_previous_history(self):
        history = PredictionHistory(np.ones(2, dtype=np.float32))
        history.append(np.array([1.0, -1.0]))
        fresh = PredictionHistory(np.ones(2, dtype=np.float32))
        np.testing.assert_array_equal(fresh.features(), np.zeros(len(FEATURE_NAMES)))
        self.assertEqual(LAGS, (1, 2, 5, 10))


if __name__ == "__main__":
    unittest.main()
