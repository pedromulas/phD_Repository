# Preprocessing pipelines

This directory contains named EEG-cleaning strategies that orchestrate the
reusable algorithms implemented in `src/functions`.

Add one script per strategy using the name specified for that strategy. New
algorithmic utilities belong in `src/functions`, not in this directory.

## Complete paired study

`run_batch_preprocessing.py` finds only complete raw/reference EEGLAB pairs
whose raw filename contains `fmri`, executes the selected strategies on each
complete recording, and evaluates their full reconstructed signals against
`derivatives/Gradient_artifact_corrected`. Results are stored under
`data/preprocessing/Dataset1/<STRATEGY>/` and
`data/evaluation/Dataset1/<STRATEGY>/`; `batch_manifest.csv` records any
recording/strategy that could not be completed.

Example (PowerShell):

```powershell
.\.venv\Scripts\python.exe src\preprocessing\run_batch_preprocessing.py
```

By default the batch detects `T0` once per recording with GradientSync
(`calibration_seconds=2`, `threshold_sigma=4`). If detection raises an error,
`T0=10 s` is used. Every strategy then processes the same 600-second interval
`[T0, T0 + 600 s]`, and evaluation crops the clean reference identically.
Disable this behaviour with `--no-use-gradient-sync-window`, or change the
duration with `--fmri-duration-seconds`.

The window is limited to the `fmrirestingec` task by default. Use
`--gradient-sync-window-scope all` to apply it to every fMRI task, or keep
`--gradient-sync-window-scope fmrirestingec` for only that modality.

Use `--strategies APPEAR BCGGAN DAR` to run a subset, or `--skip-existing` to
reuse already generated continuous results and regenerate only their metrics.
