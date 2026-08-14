import unittest
from types import SimpleNamespace

import numpy as np

from src.functions.preprocessing_window import (
    fixed_duration_sample_bounds,
    resolve_preprocessing_window,
)


class PreprocessingWindowTests(unittest.TestCase):
    def test_fixed_duration_is_measured_from_detected_t0(self) -> None:
        signal = np.zeros((2, 70_000))

        def detector(*args, **kwargs):
            return SimpleNamespace(t0_sec=7.25)

        window = resolve_preprocessing_window(
            signal,
            fs=100.0,
            tr_sec=2.0,
            n_slices=35,
            channel_names=("C3", "C4"),
            duration_sec=600.0,
            detector=detector,
        )

        self.assertEqual(window.start_sample, 725)
        self.assertEqual(window.stop_sample, 60_725)
        self.assertEqual(window.t0_source, "gradient_sync")

    def test_detector_error_uses_ten_second_fallback(self) -> None:
        signal = np.zeros((2, 70_000))

        def detector(*args, **kwargs):
            raise ValueError("no GA")

        window = resolve_preprocessing_window(
            signal,
            fs=100.0,
            tr_sec=2.0,
            n_slices=35,
            channel_names=("C3", "C4"),
            detector=detector,
        )

        self.assertEqual(window.start_sample, 1000)
        self.assertEqual(window.stop_sample, 61_000)
        self.assertEqual(window.t0_source, "fallback")
        self.assertIn("no GA", window.detection_error or "")

    def test_rejects_recording_shorter_than_requested_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains only"):
            fixed_duration_sample_bounds(1000, fs=100.0, start_sec=2.0, duration_sec=9.0)


if __name__ == "__main__":
    unittest.main()
