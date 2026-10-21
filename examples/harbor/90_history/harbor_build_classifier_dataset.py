#!/usr/bin/env python3
"""Harbor Sub-Task 4: build labelled mel-spectrogram dataset for the CNN classifier.

Extracts per-vessel per-channel sliding windows from the deterministic FDTD unit gathers,
then augments each window with Gaussian noise across N_seeds draws to build a large corpus.
Minority classes are oversampled to match the ship class count.

Usage::

    python examples/harbor/90_history/harbor_build_classifier_dataset.py \
        [--config examples/configs/harbor.scenario.json] \
        [--out output/harbor] \
        [--n-seeds 8] \
        [--win-s 4.0] \
        [--hop-s 2.0] \
        [--n-mels 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.ml.features import compute_mel_spectrogram  # noqa: E402

# ---------------------------------------------------------------------------
# Label mapping — static_scatter has level_pa=0 so it is excluded here.
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, int] = {
    "ferry":  0,  # ship
    "tug":    0,  # ship
    "moored": 0,  # ship
    "uuv1":   1,  # uuv
    "uuv2":   1,  # uuv
    "diver":  2,  # diver
}
LABEL_NAMES = ["ship", "uuv", "diver"]


def _extract_windows(
    traces: np.ndarray,   # (N_recv, N_time)
    fs: float,
    win_samples: int,
    hop_samples: int,
    n_mels: int,
    label: int,
    aug_seed: int,
    n_mels_param: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Extract mel windows from all channels; return (X_list, y_list, seed_list)."""
    X_out: list[np.ndarray] = []
    y_out: list[int] = []
    s_out: list[int] = []
    noise_scale = 0.05 * float(rng.uniform(0.5, 2.0))
    for ch in range(traces.shape[0]):
        trace = traces[ch].astype(np.float64)
        starts = range(0, len(trace) - win_samples + 1, hop_samples)
        for start in starts:
            window = trace[start: start + win_samples].copy()
            # Noise augmentation: Gaussian noise proportional to the window RMS.
            window += rng.standard_normal(window.shape) * noise_scale * np.std(window)
            mel = compute_mel_spectrogram(
                window, fs,
                n_fft=512, hop_length=128,
                n_mels=n_mels_param,
                f_min=10.0, f_max=fs / 2.0,
            )
            X_out.append(mel[np.newaxis])   # (1, n_mels, T)
            y_out.append(label)
            s_out.append(aug_seed)
    return X_out, y_out, s_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/harbor")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--win-s", type=float, default=4.0)
    ap.add_argument("--hop-s", type=float, default=2.0)
    ap.add_argument("--n-mels", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = HarborScenarioConfig.load(args.config)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}

    # Render unit gathers once — deterministic, cached after first run.
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache")

    fs = cfg.fdtd.fs
    win_samples = int(args.win_s * fs)
    hop_samples = int(args.hop_s * fs)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    seed_list: list[int] = []

    # For each vessel in the label map, augment with n_seeds independent noise draws.
    for vessel_id, label in LABEL_MAP.items():
        if vessel_id not in gathers:
            # Vessel not present in this scenario config — skip silently.
            continue
        traces = gathers[vessel_id].traces  # (N_recv, N_time) float32
        for aug_seed in range(args.n_seeds):
            rng = np.random.default_rng(aug_seed)
            X_new, y_new, s_new = _extract_windows(
                traces, fs, win_samples, hop_samples,
                args.n_mels, label, aug_seed, args.n_mels, rng,
            )
            X_list.extend(X_new)
            y_list.extend(y_new)
            seed_list.extend(s_new)

    y_arr = np.array(y_list, dtype=np.int64)

    # -----------------------------------------------------------------------
    # Class-balanced oversampling: repeat minority-class windows at random
    # to match the ship class count.
    # -----------------------------------------------------------------------
    counts = [int(np.sum(y_arr == i)) for i in range(len(LABEL_NAMES))]
    max_count = max(counts)
    print("Before oversampling: " + ", ".join(
        f"{ln}={counts[i]}" for i, ln in enumerate(LABEL_NAMES)))

    extra_X: list[np.ndarray] = []
    extra_y: list[int] = []
    extra_s: list[int] = []
    rng_os = np.random.default_rng(999)
    for cls_idx in range(len(LABEL_NAMES)):
        deficit = max_count - counts[cls_idx]
        if deficit <= 0:
            continue
        indices = np.where(y_arr == cls_idx)[0]
        chosen = rng_os.choice(indices, size=deficit, replace=True)
        for idx in chosen:
            extra_X.append(X_list[idx])
            extra_y.append(y_list[idx])
            extra_s.append(seed_list[idx])

    X_list.extend(extra_X)
    y_list.extend(extra_y)
    seed_list.extend(extra_s)

    X = np.array(X_list, dtype=np.float32)       # (N, 1, n_mels, T)
    y = np.array(y_list, dtype=np.int64)
    seeds = np.array(seed_list, dtype=np.int32)

    np.savez_compressed(
        out / "classifier_dataset.npz",
        X=X,
        y=y,
        label_names=np.array(LABEL_NAMES),
        seed=seeds,
    )
    print(f"Saved {len(y_list)} examples: " + ", ".join(
        f"{ln}={int(np.sum(y == i))}" for i, ln in enumerate(LABEL_NAMES)))
    print(f"Array shape: X={X.shape}, y={y.shape}")
    print(f"Written to {out / 'classifier_dataset.npz'}")


if __name__ == "__main__":
    main()
