import unittest

import numpy as np

from src.functions.gradient_sync import (
    _cluster_channel_peaks,
    _find_local_periodic_grid_candidate,
    _persistent_activity_bounds,
    _track_connected_periodic_run_end,
)


class GradientSyncRegressionTests(unittest.TestCase):
    def test_peak_clustering_does_not_chain_unrelated_events(self) -> None:
        samples, votes, _ = _cluster_channel_peaks(
            [np.array([0, 10, 20]), np.array([5, 15, 25])],
            [np.ones(3), np.ones(3)],
            tolerance_samples=10,
        )

        np.testing.assert_array_equal(samples, np.array([5, 20]))
        np.testing.assert_array_equal(votes, np.array([2, 2]))

    def test_persistence_confirmation_returns_first_active_window(self) -> None:
        fs = 100.0
        true_onset = 600
        features = []
        for scale in (1.0, 1.2, 0.8):
            feature = np.full(1600, 0.01, dtype=float)
            feature[true_onset:] = scale * 10.0
            features.append(feature)

        t_ga, _, window_samples, *_ = _persistent_activity_bounds(
            features,
            fs=fs,
            tr_sec=2.0,
            calibration_seconds=5.0,
            threshold_sigma=3.0,
        )

        self.assertEqual(window_samples, 25)
        self.assertEqual(t_ga, true_onset)

    def test_persistence_end_stops_at_last_active_window(self) -> None:
        fs = 100.0
        true_onset = 600
        true_end = 1200
        features = []
        for scale in (1.0, 1.2, 0.8):
            feature = np.full(1800, 0.01, dtype=float)
            feature[true_onset:true_end] = scale * 10.0
            features.append(feature)

        _, detected_end, window_samples, *_ = _persistent_activity_bounds(
            features,
            fs=fs,
            tr_sec=2.0,
            calibration_seconds=5.0,
            threshold_sigma=3.0,
        )

        self.assertEqual(window_samples, 25)
        self.assertEqual(detected_end, true_end)

    def test_local_grid_tolerates_missing_slice_peaks(self) -> None:
        fs = 1000.0
        tr_sec = 2.0
        n_slices = 35
        slice_period_samples = tr_sec * fs / n_slices
        all_peaks = np.rint(6068 + np.arange(80) * slice_period_samples).astype(np.int64)
        peaks = np.delete(all_peaks, np.arange(9, 80, 10))

        index, fraction, valid = _find_local_periodic_grid_candidate(
            peaks,
            t_ga_sample=int(peaks[0]),
            fs=fs,
            tr_sec=tr_sec,
            expected_slice_period=tr_sec / n_slices,
            period_tolerance=(tr_sec / n_slices) * 0.2,
            min_cycle_peaks=4,
            min_match_fraction=0.7,
            search_trs=1.0,
            validation_trs=2.0,
        )

        self.assertEqual(index, 0)
        self.assertTrue(valid)
        self.assertGreaterEqual(fraction, 0.85)

    def test_local_grid_never_falls_back_to_a_late_block(self) -> None:
        early_peaks = np.array([1000, 1060, 1190, 1430], dtype=np.int64)
        late_peaks = 10000 + np.arange(80, dtype=np.int64) * 57
        peaks = np.concatenate([early_peaks, late_peaks])

        index, _, valid = _find_local_periodic_grid_candidate(
            peaks,
            t_ga_sample=1000,
            fs=1000.0,
            tr_sec=2.0,
            expected_slice_period=2.0 / 35.0,
            period_tolerance=(2.0 / 35.0) * 0.2,
            min_cycle_peaks=4,
            min_match_fraction=0.7,
            search_trs=1.0,
            validation_trs=2.0,
        )

        self.assertIsNotNone(index)
        self.assertLessEqual(int(peaks[index]), 3000)
        self.assertFalse(valid)

    def test_periodic_tracking_stops_before_unrelated_late_block(self) -> None:
        expected_period = 20.0
        first_run = 100 + np.arange(80) * 20
        isolated_noise = np.array([1750, 1880], dtype=np.int64)
        late_run = 2600 + np.arange(40) * 20
        peaks = np.concatenate([first_run, isolated_noise, late_run]).astype(np.int64)

        end_sample, fraction, valid = _track_connected_periodic_run_end(
            peaks,
            t0_sample=int(first_run[0]),
            expected_period_samples=expected_period,
            tolerance_samples=2.0,
            validation_samples=200,
            min_expected_peaks=4,
            min_match_fraction=0.7,
        )

        self.assertEqual(end_sample, int(first_run[-1]))
        self.assertTrue(valid)
        self.assertGreaterEqual(fraction, 0.7)


if __name__ == "__main__":
    unittest.main()
