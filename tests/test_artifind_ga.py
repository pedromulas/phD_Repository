import unittest

import numpy as np

from src.functions.artifind_ga import detect_artifind_gradient_onsets


class ArtifindGradientDetectionTests(unittest.TestCase):
    def test_detects_first_volume_peak_without_dummy_scans(self) -> None:
        fs = 100.0
        tr_sec = 1.0
        n_volumes = 12
        onsets = 50 + np.arange(n_volumes) * 100
        signal = np.zeros((3, 1300), dtype=float)
        signal[:, onsets] = np.array([[8.0], [10.0], [12.0]])

        result = detect_artifind_gradient_onsets(
            signal,
            fs,
            tr_sec,
            n_volumes,
            n_slices=5,
            n_dummy_scans=0,
        )

        self.assertEqual(result.trigger_type, "volume")
        self.assertEqual(result.t0_sample, int(onsets[0]))
        self.assertEqual(result.t_f_sample, int(onsets[-1]))
        self.assertEqual(result.t_f_end_sample, 1250)
        self.assertTrue(result.t_f_valid)
        np.testing.assert_array_equal(result.ga_onsets, onsets)
        self.assertTrue(result.meets_artifind_criteria)

    def test_removes_complete_dummy_volumes_from_slice_onsets(self) -> None:
        fs = 100.0
        tr_sec = 1.0
        n_slices = 5
        n_volumes = 3
        n_dummy_scans = 1
        all_onsets = 10 + np.arange((n_volumes + n_dummy_scans) * n_slices) * 20
        signal = np.zeros((2, 450), dtype=float)
        signal[:, all_onsets] = np.array([[5.0], [7.0]])

        result = detect_artifind_gradient_onsets(
            signal,
            fs,
            tr_sec,
            n_volumes,
            n_slices,
            n_dummy_scans=n_dummy_scans,
            trigger_mode="slice",
        )

        self.assertEqual(result.trigger_type, "slice")
        self.assertEqual(result.t0_sample, int(all_onsets[n_slices]))
        self.assertEqual(result.t_f_sample, int(all_onsets[-1]))
        self.assertEqual(result.t_f_end_sample, 410)
        self.assertTrue(result.t_f_valid)
        np.testing.assert_array_equal(result.ga_onsets, all_onsets[n_slices:])
        self.assertEqual(result.detected_onset_count, n_volumes * n_slices)

    def test_discards_isolated_peak_before_periodic_slice_run(self) -> None:
        fs = 100.0
        tr_sec = 1.0
        n_slices = 5
        n_volumes = 4
        true_onsets = 100 + np.arange(n_volumes * n_slices) * 20
        false_early_onset = 25
        signal = np.zeros((2, 550), dtype=float)
        signal[:, false_early_onset] = np.array([8.0, 10.0])
        signal[:, true_onsets] = np.array([[8.0], [10.0]])

        result = detect_artifind_gradient_onsets(
            signal,
            fs,
            tr_sec,
            n_volumes,
            n_slices,
            trigger_mode="slice",
        )

        self.assertEqual(result.t0_sample, int(true_onsets[0]))
        self.assertEqual(result.discarded_leading_onsets, 1)
        self.assertTrue(result.local_onset_valid)
        self.assertEqual(result.local_match_fraction, 1.0)
        np.testing.assert_array_equal(result.ga_onsets, true_onsets)

    def test_expected_end_ignores_isolated_trailing_peak(self) -> None:
        fs = 100.0
        tr_sec = 1.0
        n_slices = 5
        n_volumes = 4
        true_onsets = 100 + np.arange(n_volumes * n_slices) * 20
        false_trailing_onset = 525
        signal = np.zeros((2, 600), dtype=float)
        signal[:, true_onsets] = np.array([[8.0], [10.0]])
        signal[:, false_trailing_onset] = np.array([8.0, 10.0])

        result = detect_artifind_gradient_onsets(
            signal,
            fs,
            tr_sec,
            n_volumes,
            n_slices,
            trigger_mode="slice",
        )

        self.assertEqual(result.t_f_sample, int(true_onsets[-1]))
        self.assertEqual(result.t_f_end_sample, int(true_onsets[-1] + 20))
        self.assertTrue(result.t_f_valid)
        self.assertEqual(result.discarded_trailing_onsets, 1)


if __name__ == "__main__":
    unittest.main()
