"""BCGGAN for unpaired BCG artefact removal in simultaneous EEG-fMRI.

This is a PyTorch implementation of the model in Lin et al. (2022).  It uses
two generators, two PatchGAN discriminators and the three losses described in
the paper: CycleGAN (CN1), autoencoder reconstruction (CN2), and feature
distribution alignment via RBF-MMD (CN3).  It intentionally trains on windows
rather than assuming aligned clean/corrupted examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import mne
import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover - exercised on installations without torch
    raise ImportError("BCGGAN requires PyTorch. Install it with `pip install torch`.") from error


def _window_data(
    data: np.ndarray, fs: float, window_s: float, stride_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment and normalize continuous EEG, retaining values needed to undo it."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("data must have shape (channels, samples).")
    window, stride = int(round(window_s * fs)), int(round(stride_s * fs))
    if window < 2 or stride < 1 or values.shape[1] < window:
        raise ValueError("Recording is too short for the requested window and stride.")
    starts = np.arange(0, values.shape[1] - window + 1, stride)
    windows = np.stack([values[:, start : start + window] for start in starts])
    means = windows.mean(axis=-1, keepdims=True)
    scales = np.maximum(windows.std(axis=-1, keepdims=True), 1e-6)
    return (windows - means) / scales, starts, means, scales


class EEGWindowDataset(Dataset[Tensor]):
    """Independent EEG windows for one of the unpaired BCGGAN domains."""

    def __init__(self, data: np.ndarray, fs: float, window_s: float = 1.0, stride_s: float = 1.0):
        self.windows, _, _, _ = _window_data(data, fs, window_s, stride_s)
        # Per-window channel normalization prevents the critic from learning
        # only amplitude differences between recording sessions.

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Tensor:
        return torch.from_numpy(self.windows[index])


def load_eeglab_windows(set_path: Path, window_s: float = 1.0, stride_s: float = 1.0) -> tuple[EEGWindowDataset, float, list[str]]:
    """Read an EEGLAB .set/.fdt file and create independent training windows."""
    raw = mne.io.read_raw_eeglab(Path(set_path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    return EEGWindowDataset(raw.get_data(), fs, window_s, stride_s), fs, list(raw.ch_names)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.relu(value + self.layers(value), inplace=True)


class BCGGANGenerator(nn.Module):
    """Five-module 1-D generator (M1--M5) used by BCGGAN."""

    def __init__(self, channels: int = 30, n_residual_blocks: int = 8) -> None:
        super().__init__()
        # M1: encoder/downsampling; M2-M4: feature transformation; M5: decoder.
        self.m1 = nn.Sequential(
            nn.Conv1d(channels, 64, 7, padding=3), nn.InstanceNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm1d(256), nn.ReLU(inplace=True),
        )
        split = max(1, n_residual_blocks // 2)
        self.m2 = nn.Sequential(*[ResidualBlock(256) for _ in range(split)])
        self.m3 = ResidualBlock(256)
        self.m4 = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks - split)])
        self.m5 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1), nn.InstanceNorm1d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.InstanceNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, channels, 7, padding=3),
        )

    def encode(self, value: Tensor) -> Tensor:
        return self.m1(value)

    def features(self, value: Tensor) -> Tensor:
        return self.m4(self.m3(self.m2(self.encode(value))))

    def decode(self, features: Tensor, output_length: int | None = None) -> Tensor:
        decoded = self.m5(features)
        return decoded if output_length is None else decoded[..., :output_length]

    def forward(self, value: Tensor) -> Tensor:
        return self.decode(self.features(value), value.shape[-1])


class PatchDiscriminator(nn.Module):
    """1-D PatchGAN critic; it returns scores without a final sigmoid."""

    def __init__(self, channels: int = 30) -> None:
        super().__init__()
        widths = (64, 128, 256, 512)
        layers: list[nn.Module] = []
        previous = channels
        for width in widths:
            layers.extend([nn.Conv1d(previous, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)])
            previous = width
        layers.append(nn.Conv1d(previous, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def rbf_mmd(first: Tensor, second: Tensor, bandwidth: float = 1.0) -> Tensor:
    """RBF-kernel MMD over pooled deep features, used for CN3."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    first = first.mean(dim=-1)
    second = second.mean(dim=-1)
    def kernel(left: Tensor, right: Tensor) -> Tensor:
        distance = torch.cdist(left, right).square()
        return torch.exp(-distance / (2 * bandwidth**2))
    return kernel(first, first).mean() + kernel(second, second).mean() - 2 * kernel(first, second).mean()


@dataclass(frozen=True)
class BCGGANConfig:
    channels: int = 30
    learning_rate: float = 2e-4
    lambda_cycle: float = 10.0
    lambda_ae: float = 5.0
    lambda_mmd: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BCGGANTrainer:
    """Joint optimisation of CN1, CN2 and CN3 with unpaired EEG windows."""

    def __init__(self, config: BCGGANConfig = BCGGANConfig()) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.clean_to_bcg = BCGGANGenerator(config.channels).to(self.device)
        self.bcg_to_clean = BCGGANGenerator(config.channels).to(self.device)
        self.clean_critic = PatchDiscriminator(config.channels).to(self.device)
        self.bcg_critic = PatchDiscriminator(config.channels).to(self.device)
        self.generator_optim = torch.optim.Adam(
            list(self.clean_to_bcg.parameters()) + list(self.bcg_to_clean.parameters()), lr=config.learning_rate, betas=(0.5, 0.999)
        )
        self.critic_optim = torch.optim.Adam(
            list(self.clean_critic.parameters()) + list(self.bcg_critic.parameters()), lr=config.learning_rate, betas=(0.5, 0.999)
        )
        self.completed_epochs = 0
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _gan_loss(scores: Tensor, real: bool) -> Tensor:
        return F.mse_loss(scores, torch.ones_like(scores) if real else torch.zeros_like(scores))

    def train_step(self, clean: Tensor, corrupted: Tensor) -> dict[str, float]:
        self.clean_to_bcg.train()
        self.bcg_to_clean.train()
        self.clean_critic.train()
        self.bcg_critic.train()
        clean, corrupted = clean.to(self.device), corrupted.to(self.device)
        # Train CN1 discriminators.
        with torch.no_grad():
            fake_bcg = self.clean_to_bcg(clean)
            fake_clean = self.bcg_to_clean(corrupted)
        critic_loss = (
            self._gan_loss(self.clean_critic(clean), True) + self._gan_loss(self.clean_critic(fake_clean), False)
            + self._gan_loss(self.bcg_critic(corrupted), True) + self._gan_loss(self.bcg_critic(fake_bcg), False)
        ) * 0.5
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        # CN1: adversarial + cycle consistency.
        fake_bcg = self.clean_to_bcg(clean)
        fake_clean = self.bcg_to_clean(corrupted)
        cycle_clean = self.bcg_to_clean(fake_bcg)
        cycle_bcg = self.clean_to_bcg(fake_clean)
        adversarial = self._gan_loss(self.bcg_critic(fake_bcg), True) + self._gan_loss(self.clean_critic(fake_clean), True)
        cycle_loss = F.l1_loss(cycle_clean, clean) + F.l1_loss(cycle_bcg, corrupted)
        # CN2: autoencoders formed by M1/M5 modules.
        ae_clean = self.clean_to_bcg.decode(self.clean_to_bcg.encode(clean), clean.shape[-1])
        ae_bcg = self.bcg_to_clean.decode(self.bcg_to_clean.encode(corrupted), corrupted.shape[-1])
        ae_loss = F.l1_loss(ae_clean, clean) + F.l1_loss(ae_bcg, corrupted)
        # CN3: preserve the deep feature distribution across domains and cycles.
        clean_features = self.clean_to_bcg.features(clean)
        bcg_features = self.bcg_to_clean.features(corrupted)
        mmd_loss = (
            rbf_mmd(clean_features, bcg_features)
            + rbf_mmd(clean_features, self.clean_to_bcg.features(fake_clean))
            + rbf_mmd(bcg_features, self.bcg_to_clean.features(fake_bcg))
        )
        generator_loss = adversarial + self.config.lambda_cycle * cycle_loss + self.config.lambda_ae * ae_loss + self.config.lambda_mmd * mmd_loss
        self.generator_optim.zero_grad(set_to_none=True)
        generator_loss.backward()
        self.generator_optim.step()
        return {"generator": float(generator_loss.detach()), "critic": float(critic_loss.detach()), "cycle": float(cycle_loss.detach()), "ae": float(ae_loss.detach()), "mmd": float(mmd_loss.detach())}

    @torch.no_grad()
    def validation_step(self, clean: Tensor, corrupted: Tensor) -> dict[str, float]:
        """Evaluate the same CN1/CN2/CN3 objectives without updating weights."""
        self.clean_to_bcg.eval()
        self.bcg_to_clean.eval()
        self.clean_critic.eval()
        self.bcg_critic.eval()
        clean, corrupted = clean.to(self.device), corrupted.to(self.device)
        fake_bcg = self.clean_to_bcg(clean)
        fake_clean = self.bcg_to_clean(corrupted)
        critic_loss = (
            self._gan_loss(self.clean_critic(clean), True) + self._gan_loss(self.clean_critic(fake_clean), False)
            + self._gan_loss(self.bcg_critic(corrupted), True) + self._gan_loss(self.bcg_critic(fake_bcg), False)
        ) * 0.5
        adversarial = self._gan_loss(self.bcg_critic(fake_bcg), True) + self._gan_loss(self.clean_critic(fake_clean), True)
        cycle_loss = F.l1_loss(self.bcg_to_clean(fake_bcg), clean) + F.l1_loss(self.clean_to_bcg(fake_clean), corrupted)
        ae_loss = (
            F.l1_loss(self.clean_to_bcg.decode(self.clean_to_bcg.encode(clean), clean.shape[-1]), clean)
            + F.l1_loss(self.bcg_to_clean.decode(self.bcg_to_clean.encode(corrupted), corrupted.shape[-1]), corrupted)
        )
        clean_features = self.clean_to_bcg.features(clean)
        bcg_features = self.bcg_to_clean.features(corrupted)
        mmd_loss = (
            rbf_mmd(clean_features, bcg_features)
            + rbf_mmd(clean_features, self.clean_to_bcg.features(fake_clean))
            + rbf_mmd(bcg_features, self.bcg_to_clean.features(fake_bcg))
        )
        generator_loss = adversarial + self.config.lambda_cycle * cycle_loss + self.config.lambda_ae * ae_loss + self.config.lambda_mmd * mmd_loss
        return {"generator": float(generator_loss), "critic": float(critic_loss), "cycle": float(cycle_loss), "ae": float(ae_loss), "mmd": float(mmd_loss)}

    def fit(
        self,
        clean_loader: DataLoader[Tensor],
        corrupted_loader: DataLoader[Tensor],
        epochs: int,
        validation_clean_loader: DataLoader[Tensor] | None = None,
        validation_corrupted_loader: DataLoader[Tensor] | None = None,
        progress_callback: object | None = None,
    ) -> list[dict[str, float]]:
        """Fit with independent clean/corrupted batches (no pair alignment is used)."""
        if epochs < 1:
            raise ValueError("epochs must be at least one.")
        if not len(clean_loader) or not len(corrupted_loader):
            raise ValueError("Both domains need at least one batch.")
        history: list[dict[str, float]] = []
        for _ in range(epochs):
            values: list[dict[str, float]] = []
            for batch_index, (clean, corrupted) in enumerate(zip(clean_loader, cycle(corrupted_loader)), start=1):
                values.append(self.train_step(clean, corrupted))
                if progress_callback is not None:
                    progress_callback("train", batch_index, len(clean_loader))
            epoch_history = {f"train_{key}": float(np.mean([value[key] for value in values])) for key in values[0]}
            if validation_clean_loader is not None and validation_corrupted_loader is not None:
                validation_values: list[dict[str, float]] = []
                for batch_index, (clean, corrupted) in enumerate(zip(validation_clean_loader, cycle(validation_corrupted_loader)), start=1):
                    validation_values.append(self.validation_step(clean, corrupted))
                    if progress_callback is not None:
                        progress_callback("validation", batch_index, len(validation_clean_loader))
                epoch_history.update({f"validation_{key}": float(np.mean([value[key] for value in validation_values])) for key in validation_values[0]})
            history.append(epoch_history)
            self.history.append(epoch_history)
            self.completed_epochs += 1
        return history

    @torch.no_grad()
    def clean(self, corrupted: np.ndarray) -> np.ndarray:
        """Infer clean EEG for a batch shaped ``(windows, channels, samples)``."""
        self.bcg_to_clean.eval()
        tensor = torch.as_tensor(corrupted, dtype=torch.float32, device=self.device)
        return self.bcg_to_clean(tensor).cpu().numpy()

    def clean_continuous(self, corrupted: np.ndarray, fs: float, window_s: float = 5.0, stride_s: float = 5.0, batch_size: int = 32) -> np.ndarray:
        """Clean continuous EEG with sequential 5-s training windows.

        The default is non-overlapping 5-s windows, so inferred outputs are
        concatenated chronologically and match the representation used in
        training. A final fragment shorter than a full window is retained.
        """
        normalized, starts, means, scales = _window_data(corrupted, fs, window_s, stride_s)
        output = np.zeros_like(corrupted, dtype=np.float32)
        weights = np.zeros(corrupted.shape[1], dtype=np.float32)
        for start_index in range(0, len(normalized), batch_size):
            stop_index = min(start_index + batch_size, len(normalized))
            cleaned = self.clean(normalized[start_index:stop_index]) * scales[start_index:stop_index] + means[start_index:stop_index]
            for window, sample_start in zip(cleaned, starts[start_index:stop_index]):
                sample_stop = sample_start + window.shape[-1]
                output[:, sample_start:sample_stop] += window
                weights[sample_start:sample_stop] += 1
        covered = weights > 0
        output[:, covered] /= weights[covered]
        output[:, ~covered] = corrupted[:, ~covered]
        return output

    def clean_recording(self, corrupted: np.ndarray, fs: float, window_s: float = 5.0, stride_s: float = 5.0, batch_size: int = 32) -> np.ndarray:
        """Clean a recording, applying a one-channel checkpoint independently per channel."""
        corrupted = np.asarray(corrupted)
        if corrupted.ndim != 2:
            raise ValueError("corrupted must have shape (channels, samples).")
        if self.config.channels == corrupted.shape[0]:
            return self.clean_continuous(corrupted, fs, window_s, stride_s, batch_size)
        if self.config.channels == 1:
            return np.vstack([
                self.clean_continuous(corrupted[channel : channel + 1], fs, window_s, stride_s, batch_size)
                for channel in range(corrupted.shape[0])
            ])
        raise ValueError("Input channel count does not match the trained checkpoint.")

    def save_checkpoint(self, path: Path) -> None:
        """Save models, optimizers and progress required for sequential training."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.__dict__,
                "completed_epochs": self.completed_epochs,
                "history": self.history,
                "bcg_to_clean": self.bcg_to_clean.state_dict(),
                "clean_to_bcg": self.clean_to_bcg.state_dict(),
                "clean_critic": self.clean_critic.state_dict(),
                "bcg_critic": self.bcg_critic.state_dict(),
                "generator_optimizer": self.generator_optim.state_dict(),
                "critic_optimizer": self.critic_optim.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path, device: str | None = None) -> "BCGGANTrainer":
        """Restore a checkpoint so training can continue on newly acquired EEG."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        config_values = checkpoint["config"].copy()
        if device is not None:
            config_values["device"] = device
        trainer = cls(BCGGANConfig(**config_values))
        trainer.bcg_to_clean.load_state_dict(checkpoint["bcg_to_clean"])
        trainer.clean_to_bcg.load_state_dict(checkpoint["clean_to_bcg"])
        trainer.clean_critic.load_state_dict(checkpoint["clean_critic"])
        trainer.bcg_critic.load_state_dict(checkpoint["bcg_critic"])
        trainer.generator_optim.load_state_dict(checkpoint["generator_optimizer"])
        trainer.critic_optim.load_state_dict(checkpoint["critic_optimizer"])
        trainer.completed_epochs = int(checkpoint.get("completed_epochs", 0))
        trainer.history = list(checkpoint.get("history", []))
        return trainer


def train_from_eeglab(
    clean_set: Path,
    corrupted_set: Path,
    output_model: Path,
    epochs: int = 1000,
    batch_size: int = 16,
    window_s: float = 1.0,
    resume_checkpoint: Path | None = None,
) -> list[dict[str, float]]:
    """Train or continue BCGGAN from unpaired EEGLAB/.fdt recordings."""
    clean_data, _, clean_channels = load_eeglab_windows(clean_set, window_s)
    corrupted_data, _, corrupted_channels = load_eeglab_windows(corrupted_set, window_s)
    if clean_channels != corrupted_channels:
        raise ValueError("Clean and corrupted recordings must have equal channels in the same order.")
    trainer = BCGGANTrainer.load_checkpoint(resume_checkpoint) if resume_checkpoint else BCGGANTrainer(BCGGANConfig(channels=len(clean_channels)))
    if trainer.config.channels != len(clean_channels):
        raise ValueError("Checkpoint channel count does not match the supplied recordings.")
    history = trainer.fit(DataLoader(clean_data, batch_size=batch_size, shuffle=True), DataLoader(corrupted_data, batch_size=batch_size, shuffle=True), epochs)
    trainer.save_checkpoint(output_model)
    return history
