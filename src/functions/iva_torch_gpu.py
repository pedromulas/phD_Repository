"""PyTorch/CUDA IVA-Laplace optimiser used by the accelerated EEG-fMRI path.

This is a GPU-native, differentiable IVA objective.  It is intentionally kept
separate from the third-party NumPy IVA-G/IVA-L-SOS implementation: the latter
cannot execute on CUDA.  The returned arrays follow the repository's IVA
interface so that component scoring and back-projection stay unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError as error:  # pragma: no cover - environment dependent
    raise ImportError("The GPU IVA backend requires PyTorch.") from error


def cuda_available() -> bool:
    """Return whether the installed PyTorch runtime can use CUDA."""
    return bool(torch.cuda.is_available())


def run_torch_iva_laplace(
    X: np.ndarray,
    max_iter: int = 200,
    learning_rate: float = 1e-2,
    tolerance: float = 1e-5,
    device: str = "cuda",
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Fit a real-valued IVA-Laplace model using batched CUDA linear algebra.

    ``X`` has the repository convention ``(sources, samples, channels)``.
    The objective couples the same source across channels through its vector
    norm and uses a log-determinant term to retain invertible demixing matrices.
    It is an accelerated IVA approximation, not a bitwise port of IVA-L-SOS.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] < 2:
        raise ValueError("X must have shape (n_sources, n_samples, n_channels) with at least two channels.")
    if max_iter < 1 or learning_rate <= 0 or tolerance <= 0:
        raise ValueError("max_iter, learning_rate and tolerance must be positive.")
    if device.startswith("cuda") and not cuda_available():
        raise RuntimeError("CUDA was requested for IVA but is not available in this PyTorch installation.")

    if random_state is not None:
        torch.manual_seed(random_state)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(random_state)
    torch_device = torch.device(device)
    # K x N x T permits one batched N x N unmixing matrix per EEG channel.
    x = torch.as_tensor(np.transpose(X, (2, 0, 1)), dtype=torch.float32, device=torch_device)
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    scale = centered.std(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = centered / scale
    n_channels, n_sources, _ = normalized.shape
    identity = torch.eye(n_sources, dtype=torch.float32, device=torch_device).expand(n_channels, -1, -1)
    W = torch.nn.Parameter(identity.clone() + 1e-3 * torch.randn_like(identity))
    optimizer = torch.optim.Adam([W], lr=learning_rate)
    costs: list[float] = []
    previous = float("inf")
    eps = torch.finfo(torch.float32).eps

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        Y = torch.matmul(W, normalized)
        # Multivariate Laplace IVA prior: one norm couples an SCV across K EEG
        # channels at every time sample.
        prior = torch.sqrt(torch.sum(Y.square(), dim=0) + eps).mean(dim=-1).sum()
        _, logabsdet = torch.linalg.slogdet(W)
        loss = prior - logabsdet.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("GPU IVA diverged; reduce --iva-gpu-learning-rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W], max_norm=10.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        costs.append(value)
        if abs(previous - value) / max(abs(previous), 1.0) < tolerance:
            break
        previous = value

    with torch.no_grad():
        Y = torch.matmul(W, normalized)
        # Map sources back to the original, centred channel scale.
        A = torch.diag_embed(scale.squeeze(-1)) @ torch.linalg.pinv(W)
        covariance = torch.einsum("knt,lnt->kln", Y, Y) / Y.shape[-1]
    y_np = np.transpose(Y.detach().cpu().numpy(), (1, 2, 0)).astype(np.float64)
    a_np = np.transpose(A.detach().cpu().numpy(), (1, 2, 0)).astype(np.float64)
    mean_np = np.transpose(mean.detach().cpu().numpy(), (1, 2, 0)).astype(np.float64)
    sigma_np = covariance.detach().cpu().numpy().astype(np.float64)
    w_np = np.transpose(W.detach().cpu().numpy(), (1, 2, 0)).astype(np.float64)
    return {
        "W_g": w_np,
        "W_l": w_np,
        "A_l": a_np,
        "Y": y_np,
        "cost_g": np.asarray(costs, dtype=np.float64),
        "cost_l": np.asarray(costs, dtype=np.float64),
        "sigma_g": sigma_np,
        "sigma_l": sigma_np,
        "data_mean": mean_np,
        "library_used": "torch_cuda_iva_laplace",
        "backend": "gpu",
        "device": str(torch_device),
        "iterations": len(costs),
    }


def run_blockwise_torch_iva_ga(
    signal: np.ndarray,
    TR: float,
    fs: float,
    epochs_per_block: int = 40,
    overlap_epochs: int = 4,
    periodic_frequency_hz: float = 14.0,
    max_iter: int = 200,
    learning_rate: float = 1e-2,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Apply CUDA IVA-Laplace to overlapping TR blocks and remove GA per block."""
    from functions.iva_ga import (
        _as_2d, epochs_to_iva_datasets, identify_ga_component,
        iva_datasets_to_epochs, reconstruct_without_component, segment_signal, tr_to_samples,
    )

    if epochs_per_block < 2 or not 0 <= overlap_epochs < epochs_per_block - 1:
        raise ValueError("Invalid GPU IVA block/overlap configuration.")
    data, was_1d = _as_2d(signal)
    t_samples = tr_to_samples(TR, fs)
    epochs = segment_signal(data, t_samples)
    n_epochs = epochs.shape[1]
    step = epochs_per_block - overlap_epochs
    accumulated = np.zeros_like(epochs)
    counts = np.zeros(n_epochs, dtype=np.int32)
    blocks: list[dict[str, int]] = []
    for block_index, start in enumerate(range(0, n_epochs, step)):
        stop = min(start + epochs_per_block, n_epochs)
        if stop - start < 2:
            break
        block_epochs = epochs[:, start:stop]
        X = epochs_to_iva_datasets(block_epochs)
        iva = run_torch_iva_laplace(
            X, max_iter=max_iter, learning_rate=learning_rate,
            random_state=None if random_state is None else random_state + block_index,
        )
        component = identify_ga_component(
            block_epochs, iva["Y"], iva["sigma_l"], fs, periodic_frequency_hz,
        )
        reconstruction = reconstruct_without_component(X, iva["Y"], iva["A_l"], component["ga_component"])
        reconstruction["X_clean"] += iva["data_mean"]
        accumulated[:, start:stop] += iva_datasets_to_epochs(reconstruction["X_clean"])
        counts[start:stop] += 1
        blocks.append({
            "start_epoch": int(start), "stop_epoch": int(stop),
            "ga_component": int(component["ga_component"]), "iterations": int(iva["iterations"]),
        })
        if stop == n_epochs:
            break
    cleaned = data.copy()
    usable = n_epochs * t_samples
    cleaned[:, :usable] = (accumulated / np.maximum(counts[None, :, None], 1)).reshape(data.shape[0], usable)
    return {
        "cleaned_signal": cleaned[0] if was_1d else cleaned,
        "library_used": "torch_cuda_iva_laplace",
        "backend": "gpu", "device": "cuda", "n_epochs": int(n_epochs),
        "epochs_per_block": int(epochs_per_block), "overlap_epochs": int(overlap_epochs), "blocks": blocks,
    }
