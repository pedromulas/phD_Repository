"""DAR: channel-wise 1D denoising autoencoder for EEG-fMRI artefacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - environment dependent
    raise ImportError("DAR requires PyTorch. Install it with `pip install torch`.") from error


class DARAutoencoder(nn.Module):
    """Paper architecture: 1→128→64→64→128→1 temporal convolutions."""

    def __init__(self) -> None:
        super().__init__()
        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv1d(in_channels, out_channels, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(out_channels))
        self.encoder = nn.Sequential(block(1, 128), block(128, 64))
        self.decoder = nn.Sequential(block(64, 64), block(64, 128), nn.Conv1d(128, 1, 5, padding=2), nn.Tanh())

    def forward(self, values: Tensor) -> Tensor:
        return self.decoder(self.encoder(values))


@dataclass(frozen=True)
class DARConfig:
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    window_seconds: float = 2.0
    stride_seconds: float = 1.0


class DARTrainer:
    """Supervised L1 training and overlap-add inference for DAR."""

    def __init__(self, config: DARConfig = DARConfig()) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = DARAutoencoder().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.completed_epochs = 0
        self.history: list[dict[str, float]] = []

    def step(self, noisy: Tensor, clean: Tensor, train: bool) -> float:
        self.model.train(train)
        noisy, clean = noisy.to(self.device), clean.to(self.device)
        with torch.set_grad_enabled(train):
            loss = torch.nn.functional.l1_loss(self.model(noisy), clean)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        return float(loss.detach().cpu())

    def train_epoch(self, loader: torch.utils.data.DataLoader, progress_callback: object | None = None) -> float:
        losses = []
        for index, (noisy, clean) in enumerate(loader, start=1):
            losses.append(self.step(noisy, clean, True))
            if callable(progress_callback):
                progress_callback(index, len(loader))
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self, loader: torch.utils.data.DataLoader, progress_callback: object | None = None) -> float:
        losses = []
        for index, (noisy, clean) in enumerate(loader, start=1):
            losses.append(self.step(noisy, clean, False))
            if callable(progress_callback):
                progress_callback(index, len(loader))
        return float(np.mean(losses))

    @torch.no_grad()
    def clean_continuous(self, noisy: np.ndarray, fs: float, batch_size: int = 32) -> np.ndarray:
        """Denoise a single channel using 2-s windows and 1-s overlap-add."""
        values = np.asarray(noisy, dtype=np.float32).reshape(-1)
        window, stride = int(round(self.config.window_seconds * fs)), int(round(self.config.stride_seconds * fs))
        if values.size < window:
            raise ValueError("Recording is shorter than DAR's 2-second input window.")
        starts = np.arange(0, values.size - window + 1, stride)
        if starts[-1] != values.size - window:
            starts = np.append(starts, values.size - window)
        windows = np.stack([values[start : start + window] for start in starts])
        scales = np.maximum(np.max(np.abs(windows), axis=1, keepdims=True), 1e-8)
        normalized = (windows / scales)[:, None, :]
        output, weights = np.zeros(values.size, dtype=np.float32), np.zeros(values.size, dtype=np.float32)
        self.model.eval()
        for first in range(0, len(normalized), batch_size):
            batch = torch.as_tensor(normalized[first : first + batch_size], device=self.device)
            predicted = self.model(batch).cpu().numpy()[:, 0] * scales[first : first + batch_size]
            for segment, start in zip(predicted, starts[first : first + batch_size]):
                output[start : start + window] += segment
                weights[start : start + window] += 1
        covered = weights > 0
        output[covered] /= weights[covered]
        output[~covered] = values[~covered]
        return output

    def clean_recording(self, noisy: np.ndarray, fs: float, batch_size: int = 32) -> np.ndarray:
        values = np.asarray(noisy)
        if values.ndim != 2:
            raise ValueError("noisy must have shape (channels, samples).")
        return np.vstack([self.clean_continuous(channel, fs, batch_size) for channel in values])

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        payload = {"config": asdict(self.config), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "completed_epochs": self.completed_epochs, "history": self.history}
        try:
            # The legacy stream avoids intermittent ZIP writer failures seen on
            # some Windows filesystems. ``os.replace`` exposes only complete
            # checkpoints to a concurrent notebook or a resumed training run.
            torch.save(payload, temporary_path, _use_new_zipfile_serialization=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load_checkpoint(cls, path: Path, device: str | None = None) -> "DARTrainer":
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        config = dict(checkpoint["config"])
        config["device"] = target_device
        trainer = cls(DARConfig(**config))
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.completed_epochs = int(checkpoint.get("completed_epochs", 0))
        trainer.history = list(checkpoint.get("history", []))
        return trainer
