"""Train BCGGAN from 5-second, single-channel raw/GA-corrected EEG windows."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

try:  # Supports both ``python -m functions...`` and direct script execution.
    from functions.bcggan import BCGGANConfig, BCGGANTrainer
    from functions.bcggan_training_data import (
        SingleChannelEEGWindowDataset,
        build_window_index,
        find_gradient_corrected_pairs,
        split_window_index,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution convenience
    from bcggan import BCGGANConfig, BCGGANTrainer
    from bcggan_training_data import SingleChannelEEGWindowDataset, build_window_index, find_gradient_corrected_pairs, split_window_index

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as error:  # pragma: no cover - depends on the training environment
    raise ImportError("BCGGAN training requires PyTorch. Install it with `pip install torch`.") from error


DEFAULT_EEG_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")
DEFAULT_CORRECTED_ROOT = DEFAULT_EEG_ROOT / "derivatives/Gradient_artifact_corrected"


def _ema(values: list[float], alpha: float) -> list[float]:
    """Exponentially smooth a display curve without changing recorded losses."""
    if not 0 < alpha <= 1:
        raise ValueError("loss EMA alpha must be in (0, 1].")
    output: list[float] = []
    for value in values:
        output.append(value if not output else alpha * value + (1 - alpha) * output[-1])
    return output


def _next_experiment_name(output_root: Path) -> str:
    """Choose the next ``trainingN`` directory without overwriting a prior run."""
    numbers = [
        int(path.name.removeprefix("training"))
        for path in output_root.glob("training*")
        if path.is_dir() and path.name.removeprefix("training").isdigit()
    ]
    return f"training{max(numbers, default=0) + 1}"


def _render_loss_plot(
    history: list[dict[str, float]],
    figure_path: Path,
    maximum_epochs: int,
    remaining_seconds: float | None = None,
    early_stopping_wait: int | None = None,
    patience: int | None = None,
    loss_ema_alpha: float = 0.3,
) -> None:
    """Write an updated graph showing losses, completed epochs and ETA."""
    if not history:
        return
    current_epoch = history[-1]["epoch"]
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5))
    train_loss = [row["train_generator"] for row in history]
    validation_loss = [row["validation_generator"] for row in history]
    plt.plot(epochs, train_loss, label="Training (raw)", color="tab:blue", alpha=0.25)
    plt.plot(epochs, validation_loss, label="Validation (raw)", color="tab:orange", alpha=0.25)
    plt.plot(epochs, _ema(train_loss, loss_ema_alpha), label=f"Training EMA (α={loss_ema_alpha:g})", color="tab:blue", lw=2)
    plt.plot(epochs, _ema(validation_loss, loss_ema_alpha), label=f"Validation EMA (α={loss_ema_alpha:g})", color="tab:orange", lw=2)
    plt.axvline(current_epoch, color="black", ls="--", lw=1, label=f"Current epoch: {current_epoch}")
    plt.xlim(1, maximum_epochs)
    title = f"BCGGAN progress: epoch {current_epoch}/{maximum_epochs}"
    if remaining_seconds is not None:
        title += f" | estimated max ETA: {remaining_seconds / 60:.1f} min"
    if early_stopping_wait is not None and patience is not None:
        title += f" | early stopping: {early_stopping_wait}/{patience}"
    plt.xlabel("Epoch"); plt.ylabel("BCGGAN generator loss")
    plt.title(title)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()


def _save_history(history: list[dict[str, float]], output_dir: Path, maximum_epochs: int, loss_ema_alpha: float) -> tuple[Path, Path]:
    """Persist numerical history and the latest train/validation loss plot."""
    csv_path = output_dir / "losses.csv"
    json_path = output_dir / "losses.json"
    keys = ["epoch", *sorted({key for row in history for key in row if key != "epoch"})]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    figure_path = output_dir / "generator_loss.png"
    _render_loss_plot(history, figure_path, maximum_epochs, loss_ema_alpha=loss_ema_alpha)
    return csv_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-channel BCGGAN using 5-second raw/GA-corrected EEG windows.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--task", default="fmrirestingec", help="BIDS EEG task name; omit with --task all.")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--max-time-windows", type=int, help="Optional cap on 5-second temporal windows for a fast smoke test; all EEG channels of each chosen window are retained.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--early-stopping", action="store_true", help="Opt in to early stopping; disabled by default.")
    parser.add_argument("--patience", type=int, default=8, help="Epochs without a meaningful validation improvement before stopping.")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum validation-loss decrease considered an improvement for early stopping.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Reduced GAN learning rate for more stable validation loss.")
    parser.add_argument("--loss-ema-alpha", type=float, default=0.3, help="EMA smoothing used only in saved loss figures.")
    parser.add_argument("--num-workers", type=int, default=0, help="Keep zero on Windows unless disk throughput is known to be sufficient.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"), help="Root containing one subfolder per training run.")
    parser.add_argument("--experiment-name", help="Run folder, e.g. training1. Defaults to the next available trainingN.")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume sequential training from a prior checkpoint.")
    args = parser.parse_args()
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if not 0 < args.loss_ema_alpha <= 1:
        raise ValueError("loss-ema-alpha must be in (0, 1].")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = None if args.task.lower() == "all" else args.task
    records = find_gradient_corrected_pairs(args.raw_root, args.corrected_root, task=task)
    if len({record.sfreq for record in records}) != 1:
        raise ValueError("All recordings must have the same sampling frequency for batched 5-second training.")
    index = build_window_index(records, args.window_seconds)
    if args.max_time_windows is not None:
        if args.max_time_windows < 2:
            raise ValueError("max-time-windows must be at least two to preserve an 80/20 split.")
        groups = np.unique([item.group_index for item in index])
        rng = np.random.default_rng(args.seed)
        rng.shuffle(groups)
        selected_groups = set(groups[: min(args.max_time_windows, groups.size)].tolist())
        index = [item for item in index if item.group_index in selected_groups]
    train_index, validation_index = split_window_index(index, args.validation_fraction, args.seed)
    datasets = {
        "train_clean": SingleChannelEEGWindowDataset(records, train_index, "clean"),
        "train_raw": SingleChannelEEGWindowDataset(records, train_index, "raw"),
        "validation_clean": SingleChannelEEGWindowDataset(records, validation_index, "clean"),
        "validation_raw": SingleChannelEEGWindowDataset(records, validation_index, "raw"),
    }
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    train_clean = DataLoader(datasets["train_clean"], shuffle=True, **loader_options)
    train_raw = DataLoader(datasets["train_raw"], shuffle=True, **loader_options)
    validation_clean = DataLoader(datasets["validation_clean"], shuffle=False, **loader_options)
    validation_raw = DataLoader(datasets["validation_raw"], shuffle=False, **loader_options)

    trainer = BCGGANTrainer.load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else BCGGANTrainer(
        BCGGANConfig(channels=1, learning_rate=args.learning_rate)
    )
    if trainer.config.channels != 1:
        raise ValueError("This workflow trains one channel at a time; the checkpoint must have channels=1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = args.experiment_name or _next_experiment_name(args.output_dir)
    experiment_dir = args.output_dir / experiment_name
    if experiment_dir.exists() and not args.resume_checkpoint:
        raise FileExistsError(f"Experiment directory already exists: {experiment_dir}. Choose another --experiment-name.")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    maximum_epoch_number = trainer.completed_epochs + args.epochs
    last_checkpoint = experiment_dir / "last.pt"
    best_checkpoint = experiment_dir / "best.pt"
    progress_figure = experiment_dir / "training_progress.png"

    print(f"Matched recordings: {len(records)} | complete 5 s channel-windows: {len(index)}")
    print(f"Train: {len(train_index)} | validation: {len(validation_index)} | maximum epochs: {args.epochs}")
    print(f"Experiment: {experiment_dir}")
    print(f"Device: {trainer.device} | learning rate: {trainer.config.learning_rate:g} | train batches per epoch: {len(train_clean)} | validation batches per epoch: {len(validation_clean)}")
    best_loss, best_early_stopping_loss, waiting = np.inf, np.inf, 0
    history: list[dict[str, float]] = []
    started_at = time.perf_counter()
    for _ in range(args.epochs):
        progress_bars: dict[str, tqdm] = {}

        def update_batch_progress(stage: str, current: int, total: int) -> None:
            if stage not in progress_bars:
                progress_bars[stage] = tqdm(total=total, desc=f"Epoch {trainer.completed_epochs + 1} {stage}", unit="batch")
            progress_bars[stage].update(1)

        try:
            metrics = trainer.fit(
                train_clean, train_raw, 1, validation_clean, validation_raw,
                progress_callback=update_batch_progress,
            )[0]
        finally:
            for progress_bar in progress_bars.values():
                progress_bar.close()
        row = {"epoch": trainer.completed_epochs, **metrics}
        history.append(row)
        trainer.save_checkpoint(last_checkpoint)
        validation_loss = row["validation_generator"]
        if validation_loss < best_loss:
            best_loss = validation_loss
            shutil.copy2(last_checkpoint, best_checkpoint)
        if validation_loss < best_early_stopping_loss - args.min_delta:
            best_early_stopping_loss, waiting = validation_loss, 0
        else:
            waiting += 1
        elapsed = time.perf_counter() - started_at
        mean_epoch_seconds = elapsed / len(history)
        remaining_seconds = mean_epoch_seconds * max(0, args.epochs - len(history))
        _render_loss_plot(
            history, progress_figure, maximum_epoch_number, remaining_seconds,
            waiting if args.early_stopping else None, args.patience if args.early_stopping else None,
            args.loss_ema_alpha,
        )
        print(
            f"Epoch {row['epoch']:03d}/{maximum_epoch_number}: train={row['train_generator']:.5f}, "
            f"validation={validation_loss:.5f} | elapsed={elapsed / 60:.1f} min, "
            f"estimated max remaining={remaining_seconds / 60:.1f} min"
        )
        if args.early_stopping and waiting >= args.patience:
            print(f"Early stopping after {waiting} epochs without validation improvement.")
            break

    csv_path, figure_path = _save_history(history, experiment_dir, maximum_epoch_number, args.loss_ema_alpha)
    metadata_path = experiment_dir / "metadata.json"
    metadata_path.write_text(json.dumps({
        "records": len(records), "windows_total": len(index), "windows_train": len(train_index),
        "windows_validation": len(validation_index), "window_seconds": args.window_seconds,
        "sampling_frequency": records[0].sfreq, "best_validation_generator_loss": best_loss,
        "raw_root": str(args.raw_root), "reference_root": str(args.corrected_root),
        "task": task, "seed": args.seed, "learning_rate": trainer.config.learning_rate,
        "early_stopping_enabled": args.early_stopping, "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta, "device": str(trainer.device),
        "loss_ema_alpha": args.loss_ema_alpha,
    }, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_checkpoint}\nLast checkpoint: {last_checkpoint}\nLoss history: {csv_path}\nLoss figure: {figure_path}\nLive progress figure: {progress_figure}")


if __name__ == "__main__":
    main()
