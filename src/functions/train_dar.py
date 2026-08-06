"""Train DAR on paired raw/GAC EEG windows and save reproducible run artefacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from functions.bcggan_training_data import find_gradient_corrected_pairs
    from functions.dar import DARConfig, DARTrainer
    from functions.dar_training_data import DARPairedWindowDataset, build_dar_window_index, split_dar_index
except ModuleNotFoundError:  # pragma: no cover
    from bcggan_training_data import find_gradient_corrected_pairs
    from dar import DARConfig, DARTrainer
    from dar_training_data import DARPairedWindowDataset, build_dar_window_index, split_dar_index


DEFAULT_ROOT = Path("data/raw/Dataset1/Simultaneous_EEG_fMRI/BIDS_dataset_EEG")


def _next_run(root: Path) -> str:
    values = [int(path.name.removeprefix("training")) for path in root.glob("training*") if path.is_dir() and path.name.removeprefix("training").isdigit()]
    return f"training{max(values, default=0) + 1}"


def _plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_l1"] for row in history], label="Training")
    plt.plot(epochs, [row["validation_l1"] for row in history], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("L1 loss"); plt.title("DAR training"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised DAR using paired raw and gradient-corrected EEG.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_ROOT / "derivatives/Gradient_artifact_corrected")
    parser.add_argument("--task", default="fmrirestingec")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("data/models/DAR"))
    parser.add_argument("--experiment-name")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.window_seconds <= 0 or args.stride_seconds <= 0 or args.stride_seconds > args.window_seconds:
        raise ValueError("window-seconds and stride-seconds must be positive, with stride no larger than window.")
    records = find_gradient_corrected_pairs(args.raw_root, args.corrected_root, task=args.task)
    index = build_dar_window_index(records, args.window_seconds, args.stride_seconds)
    train_index, validation_index = split_dar_index(index, seed=args.seed)
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(DARPairedWindowDataset(records, train_index), shuffle=True, **options)
    validation_loader = DataLoader(DARPairedWindowDataset(records, validation_index), shuffle=False, **options)
    trainer = DARTrainer.load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else DARTrainer(
        DARConfig(learning_rate=args.learning_rate, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.experiment_name or _next_run(args.output_root)
    output = args.output_root / run
    if output.exists() and not args.resume_checkpoint: raise FileExistsError(f"Run directory exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    best, waiting, history = np.inf, 0, []
    started_at = time.perf_counter()
    print(f"Run: {output} | device: {trainer.device} | train windows: {len(train_index)} | validation: {len(validation_index)}")
    for _ in range(args.epochs):
        bars: dict[str, tqdm] = {}
        def update(stage: str, current: int, total: int) -> None:
            if stage not in bars:
                bars[stage] = tqdm(total=total, desc=f"Epoch {trainer.completed_epochs + 1} {stage}", unit="batch")
            bars[stage].update(1)
        try:
            train_loss = trainer.train_epoch(train_loader, lambda current, total: update("train", current, total))
            validation_loss = trainer.validate(validation_loader, lambda current, total: update("validation", current, total))
        finally:
            for bar in bars.values(): bar.close()
        trainer.completed_epochs += 1
        row = {"epoch": trainer.completed_epochs, "train_l1": train_loss, "validation_l1": validation_loss}
        history.append(row); trainer.history.append(row); trainer.save_checkpoint(output / "last.pt")
        if validation_loss < best - args.min_delta:
            best, waiting = validation_loss, 0; shutil.copy2(output / "last.pt", output / "best.pt")
        else: waiting += 1
        _plot(history, output / "training_progress.png")
        elapsed = time.perf_counter() - started_at
        remaining = elapsed / len(history) * max(0, args.epochs - len(history))
        print(f"Epoch {trainer.completed_epochs}/{args.epochs}: train L1={train_loss:.6f}, validation L1={validation_loss:.6f}, patience={waiting}/{args.patience} | elapsed={elapsed / 60:.1f} min, estimated remaining={remaining / 60:.1f} min")
        if waiting >= args.patience: break
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_l1", "validation_l1"]); writer.writeheader(); writer.writerows(history)
    (output / "metadata.json").write_text(json.dumps({"records": len(records), "train_windows": len(train_index), "validation_windows": len(validation_index), "window_seconds": args.window_seconds, "stride_seconds": args.stride_seconds, "best_validation_l1": best, "task": args.task}, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output / 'best.pt'}\nLosses: {output / 'losses.csv'}\nFigure: {output / 'training_progress.png'}")


if __name__ == "__main__": main()
