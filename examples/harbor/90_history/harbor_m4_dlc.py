#!/usr/bin/env python3
"""Harbor M4: end-to-end detect–locate–classify pipeline.

Runs the full pipeline on the extended M2 dataset:
  1. CFAR calibration on noise-only seeds.
  2. Full scenario detection (AIS-aware or bare matched-field).
  3. Per-detection mel-spectrogram extraction from the nearest DAS channel.
  4. CNN coarse label + CVAE anomaly score via HarborClassifier.
  5. Evaluation: per-target Pd/RMSE and per-class CNN accuracy.
  6. Outputs: m4_track_map.png, m4_roc.png, m4_summary.json.

The script runs end-to-end even if the CNN/CVAE checkpoints do not exist yet;
in that case classification is skipped and all detections are labelled "unclassified".

Usage::

    python examples/harbor/90_history/harbor_m4_dlc.py \\
        [--config examples/configs/harbor.scenario.json] \\
        [--out output/studies/history/harbor_m4_dlc] \\
        [--cnn output/studies/history/harbor_train_classifier/cnn_classifier.pt] \\
        [--cvae output/studies/history/harbor_train_anomaly/cvae_anomaly.pt] \\
        [--pfa 0.01] \\
        [--cal-seeds 4] \\
        [--no-suppress]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from acoustic_sim.contract import annotate_figure  # noqa: E402
from acoustic_sim.water import (  # noqa: E402
    HarborScenarioConfig,
    assemble_dataset,
    build_environment,
    build_layout,
    build_track,
    render_vessel_gathers,
)
from acoustic_sim.water.detection import (  # noqa: E402
    ProcessingConfig,
    build_grid,
    run_detection,
    threshold_from_maxima,
)
from acoustic_sim.water.evaluate import score_detections, truth_labels  # noqa: E402
from acoustic_sim.plotting import new_figure, save_figure  # noqa: E402
from acoustic_sim.water.plotting import plot_harbor  # noqa: E402
from acoustic_sim.water.suppression import run_ais_aware  # noqa: E402
from acoustic_sim.ml.features import compute_mel_spectrogram  # noqa: E402

# ---------------------------------------------------------------------------
# Label colours for the track map
# ---------------------------------------------------------------------------
_LABEL_COLOUR = {
    "ship":          "green",
    "uuv":           "red",
    "diver":         "orange",
    "unclassified":  "grey",
}

# Labels that are ship-class (AIS vessels) — used for ROC construction
_SHIP_LABELS = {"ferry", "tug", "moored", "ship"}


# ---------------------------------------------------------------------------
# Step 4 helper: mel-spectrogram extraction for one detection
# ---------------------------------------------------------------------------

def _extract_mel(
    traces: np.ndarray,
    fs: float,
    layout,
    t_det: float,
    xy_det: np.ndarray,
    win_s: float = 4.0,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 10.0,
) -> np.ndarray:
    """Return a ``(1, n_mels, T)`` float32 mel-spectrogram for one detection.

    The nearest channel on the first cable is selected; if no cables exist
    the function falls back to the first receiver.
    """
    # Find the nearest live DAS channel on the first cable.
    if layout.cables:
        cable = layout.cables[0]
        cable_start = 0  # cable rows begin at index 0 in the receiver layout
        positions = cable.positions                          # (n_ch, 2)
        diff = positions - xy_det[None, :]                  # (n_ch, 2)
        ch_local = int(np.argmin(np.hypot(diff[:, 0], diff[:, 1])))
        ch_idx = cable_start + ch_local
    else:
        ch_idx = 0

    # Slice a win_s-second window centred on t_det.
    win_samples = int(win_s * fs)
    half = win_samples // 2
    start = max(0, int(t_det * fs) - half)
    end = start + win_samples
    n_total = traces.shape[1]
    if end > n_total:
        start = max(0, n_total - win_samples)
        end = n_total

    window = traces[ch_idx, start:end].copy().astype(np.float32)

    # Zero-pad on the right if the window is shorter than win_s (near the edge).
    if len(window) < win_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[: len(window)] = window
        window = padded

    mel = compute_mel_spectrogram(
        window.astype(np.float64), fs,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min,
    ).astype(np.float32)
    return mel[np.newaxis]   # (1, n_mels, T)


# ---------------------------------------------------------------------------
# Step 6 helper: per-class CNN accuracy
# ---------------------------------------------------------------------------

def _per_class_accuracy(
    clf_results: list[dict],
    assoc_labels: list[str | None],
) -> dict[str, float]:
    """Compute per-class CNN accuracy from associated ground-truth vessel IDs.

    Ground-truth vessel kinds are inferred from the label name:
      - ``ferry``, ``tug``, ``moored`` → "ship"
      - ``uuv1``, ``uuv2`` (starts with "uuv") → "uuv"
      - ``diver``          → "diver"

    Detections without a ground-truth association (``None``) are skipped.
    """
    def _gt_class(vid: str) -> str:
        if vid in ("ferry", "tug", "moored"):
            return "ship"
        if vid.startswith("uuv"):
            return "uuv"
        if vid == "diver":
            return "diver"
        return "unknown"

    buckets: dict[str, list[bool]] = {}
    for r, assoc in zip(clf_results, assoc_labels):
        if assoc is None:
            continue
        gt = _gt_class(assoc)
        pred = r["label"]
        # Map CVAE/CNN label to the same 3-class scheme for comparison.
        pred_coarse = pred if pred in ("ship", "uuv", "diver") else "unknown"
        buckets.setdefault(gt, []).append(pred_coarse == gt)

    return {cls: float(np.mean(hits)) for cls, hits in buckets.items() if hits}


# ---------------------------------------------------------------------------
# Step 7b helper: manual ROC + AUC (no sklearn)
# ---------------------------------------------------------------------------

def _compute_roc(
    scores: np.ndarray,
    binary_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC curve and AUC using anomaly_score as discriminant.

    Returns ``(fpr, tpr, auc)``.  Scores that are NaN are excluded.
    """
    valid = ~np.isnan(scores)
    s = scores[valid]
    y = binary_labels[valid]

    if len(np.unique(s)) < 2 or len(np.unique(y)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    thresholds = np.unique(s)[::-1]   # descending
    tpr_list, fpr_list = [], []
    for thresh in thresholds:
        tp = int(np.sum((s >= thresh) & (y == 1)))
        fp = int(np.sum((s >= thresh) & (y == 0)))
        fn = int(np.sum((s < thresh) & (y == 1)))
        tn = int(np.sum((s < thresh) & (y == 0)))
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapz(tpr[::-1], fpr[::-1]))
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "harbor.scenario.json"))
    ap.add_argument("--out", default="output/studies/history/harbor_m4_dlc")
    ap.add_argument("--cnn", default="output/studies/history/harbor_train_classifier/cnn_classifier.pt")
    ap.add_argument("--cvae", default="output/studies/history/harbor_train_anomaly/cvae_anomaly.pt")
    ap.add_argument("--pfa", type=float, default=0.01,
                    help="per-frame false-alarm probability for CFAR threshold")
    ap.add_argument("--cal-seeds", type=int, default=4,
                    help="number of noise-only seeds used for CFAR calibration")
    ap.add_argument("--no-suppress", dest="suppress", action="store_false", default=True,
                    help="Disable AIS-aware ship suppression (run bare matched-field scan)")
    ap.add_argument("--cuda", action="store_true", default=False,
                    help="Run MFP beamforming on the GPU via CuPy; PyTorch classifier uses CUDA too")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Setup
    # ------------------------------------------------------------------
    cfg = HarborScenarioConfig.load(args.config)
    env = build_environment(cfg)
    layout = build_layout(cfg, env)
    tracks = {v.id: build_track(v, cfg.fdtd.total_time, env) for v in cfg.vessels}
    gathers = render_vessel_gathers(cfg, env, layout, tracks, cache_dir=out / "cache")
    chart = env.public()          # what the processing knows of the environment

    base_pc = ProcessingConfig(use_cuda=args.cuda)
    grid = build_grid(chart, base_pc)
    print(f"scan grid: {len(grid)} cells, {layout.n_receivers} receivers", flush=True)

    zero_levels = {v.id: 0.0 for v in cfg.vessels}
    pc_detect = ProcessingConfig(line_halfwidth=4, use_cuda=args.cuda)

    # ------------------------------------------------------------------
    # Step 2 — CFAR calibration (noise-only, no vessels)
    # ------------------------------------------------------------------
    print(f"CFAR calibration: {args.cal_seeds} noise-only seeds …", flush=True)
    maxima = []
    for s in range(args.cal_seeds):
        ds_cal = assemble_dataset(
            cfg, layout, tracks, gathers,
            seed=1000 + s, levels=zero_levels, keep_clean=False,
        )
        if args.suppress:
            run_cal = run_ais_aware(
                ds_cal.traces, layout, ds_cal.fs, chart, ds_cal.ais,
                ProcessingConfig(line_halfwidth=4, use_cuda=args.cuda),
                floor=1e18, grid=grid,
            )
        else:
            run_cal = run_detection(
                ds_cal.traces, layout, ds_cal.fs, chart,
                ProcessingConfig(use_cuda=args.cuda),
                floor=1e18, grid=grid,
            )
        maxima.append(run_cal.scan.frame_max)

    thr = threshold_from_maxima(np.concatenate(maxima), args.pfa)
    print(f"CFAR threshold: {thr:.1f}  (noise max median {np.median(np.concatenate(maxima)):.1f})",
          flush=True)

    # ------------------------------------------------------------------
    # Step 3 — Full scenario detection (all vessels at nominal levels)
    # ------------------------------------------------------------------
    print("Running full scenario detection …", flush=True)
    ds = assemble_dataset(cfg, layout, tracks, gathers, seed=42, keep_clean=False)

    if args.suppress:
        run = run_ais_aware(
            ds.traces, layout, ds.fs, chart, ds.ais,
            pc_detect, floor=0.5 * thr, grid=grid,
        )
    else:
        run = run_detection(
            ds.traces, layout, ds.fs, chart,
            base_pc, floor=0.5 * thr, grid=grid,
        )

    dets = run.dets.above(thr)
    print(f"Detections above threshold: {len(dets)} / {len(run.dets)}", flush=True)

    fs = float(ds.fs)

    # ------------------------------------------------------------------
    # Step 4 — Per-detection mel-spectrogram extraction
    # ------------------------------------------------------------------
    print("Extracting mel-spectrograms for each detection …", flush=True)
    mel_list: list[np.ndarray] = []
    for i in range(len(dets)):
        mel = _extract_mel(
            ds.traces, fs, layout,
            t_det=float(dets.t[i]),
            xy_det=dets.xy[i],
        )
        mel_list.append(mel)

    # ------------------------------------------------------------------
    # Step 5 — Classify all detections (skip gracefully if CNN missing)
    # ------------------------------------------------------------------
    cnn_path = Path(args.cnn)
    clf_results: list[dict] = []
    classifier_available = cnn_path.exists()

    if classifier_available:
        try:
            from acoustic_sim.water.classifier import HarborClassifier  # noqa: E402
            device = "cuda" if args.cuda else "cpu"
            clf = HarborClassifier.load(str(cnn_path), str(args.cvae), device=device)
            if mel_list:
                mel_batch = np.stack(mel_list)          # (D, 1, n_mels, T)
                clf_results = clf.classify_batch(mel_batch)
            print(f"Classified {len(clf_results)} detections with CNN.", flush=True)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Classifier failed: {exc} — skipping classification.")
            classifier_available = False

    if not classifier_available:
        warnings.warn(
            f"CNN checkpoint not found at {cnn_path!r}. "
            "Skipping classification; all detections labelled 'unclassified'."
        )
        clf_results = [
            {"label": "unclassified", "label_idx": -1,
             "cnn_probs": np.array([]), "anomaly_score": float("nan")}
            for _ in mel_list
        ]

    # ------------------------------------------------------------------
    # Step 6 — Evaluate: score_detections + per-class CNN accuracy
    # ------------------------------------------------------------------
    print("Scoring detections …", flush=True)
    truth = ds.truth
    sc = score_detections(dets, truth, layout, run.wh.times, thr,
                          targets=list(tracks))

    # Associate each detection with the nearest truth vessel (for accuracy).
    assoc_labels, _ = truth_labels(dets, truth, gate=100.0)

    per_class_acc = _per_class_accuracy(clf_results, assoc_labels) if classifier_available else {}

    # ------------------------------------------------------------------
    # Step 7a — m4_track_map.png
    # ------------------------------------------------------------------
    print("Plotting track map …", flush=True)
    fig, ax = new_figure((12, 10))
    plot_harbor(env, layout, ax=ax)

    # Truth tracks.
    for vid, tr in ds.tracks.items():
        if tr.level_pa <= 0:
            continue
        xy_active = tr.xy[tr.active]
        if len(xy_active) == 0:
            continue
        kind = tr.kind
        if kind == "ship" or vid in ("ferry", "tug", "moored"):
            style, colour = "-", "blue"
        elif kind == "uuv" or vid.startswith("uuv"):
            style, colour = "--", "red"
        else:
            style, colour = "--", "orange"
        ax.plot(xy_active[:, 0], xy_active[:, 1], style,
                color=colour, lw=1.5, label=f"truth {vid}")

    # Detection scatter coloured by CNN label, sized by anomaly score.
    if len(dets) > 0:
        scores_raw = np.array([r["anomaly_score"] for r in clf_results], dtype=float)
        valid_scores = scores_raw[~np.isnan(scores_raw)]
        if len(valid_scores) >= 2:
            s_min, s_max = valid_scores.min(), valid_scores.max()
            denom = max(s_max - s_min, 1e-9)
            sizes = np.where(
                np.isnan(scores_raw),
                60.0,
                20.0 + 180.0 * np.clip((scores_raw - s_min) / denom, 0.0, 1.0),
            )
        else:
            sizes = np.full(len(dets), 60.0)

        colours = [_LABEL_COLOUR.get(r["label"], "grey") for r in clf_results]
        ax.scatter(dets.xy[:, 0], dets.xy[:, 1],
                   c=colours, s=sizes,
                   edgecolors="k", linewidths=0.4, zorder=5,
                   label="detections (colour=CNN label, size∝anomaly)")

    ax.set_title(f"harbor (5 m): detect, locate, classify (Pfa={args.pfa}, suppress={args.suppress})")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")
    # Deduplicate legend entries.
    handles, labels_leg = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    ax.legend(list(seen.values()), list(seen.keys()), loc="upper left", fontsize=11)
    fig.axes[0].set_title(fig.axes[0].get_title() + '\n' + run.provenance.label())
    save_figure(fig, out / "m4_track_map.png", recording="harbor")
    print(f"  → {out / 'm4_track_map.png'}", flush=True)

    # ------------------------------------------------------------------
    # Step 7b — m4_roc.png (skip if fewer than 2 unique scores)
    # ------------------------------------------------------------------
    roc_auc: float | None = None
    roc_skip_reason: str | None = None

    if classifier_available:
        anomaly_scores = np.array([r["anomaly_score"] for r in clf_results], dtype=float)

        # Build binary labels: 1 = non-ship (uuv/diver/unknown), 0 = ship.
        binary_labels = np.array([
            0 if (lb is not None and lb in _SHIP_LABELS) else 1
            for lb in assoc_labels
        ], dtype=int)

        valid_mask = ~np.isnan(anomaly_scores)
        if valid_mask.sum() < 2 or len(np.unique(anomaly_scores[valid_mask])) < 2:
            roc_skip_reason = "fewer than 2 unique anomaly scores — ROC not computed"
            warnings.warn(roc_skip_reason)
        else:
            fpr, tpr, roc_auc = _compute_roc(anomaly_scores, binary_labels)
            if np.isnan(roc_auc):
                roc_skip_reason = "only one class present in labels — ROC not computed"
            else:
                fig_roc, ax_roc = new_figure((11, 9))
                ax_roc.plot(fpr, tpr, "b-", lw=2, label=f"CVAE anomaly score (AUC = {roc_auc:.3f})")
                ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
                ax_roc.set_xlabel("False Positive Rate (ship labelled non-ship)")
                ax_roc.set_ylabel("True Positive Rate (non-ship correctly identified)")
                ax_roc.set_title("harbor (5 m): ROC of the CVAE anomaly score, non-ship (uuv, diver) against ship")
                ax_roc.legend(loc="lower right")
                ax_roc.set_xlim(0, 1)
                ax_roc.set_ylim(0, 1.02)
                ax_roc.grid(alpha=0.3)
                fig_roc.axes[0].set_title(fig_roc.axes[0].get_title() + '\n' + run.provenance.label())
                save_figure(fig_roc, out / "m4_roc.png", recording="harbor")
                print(f"  → {out / 'm4_roc.png'}  AUC = {roc_auc:.3f}", flush=True)
    else:
        roc_skip_reason = "classifier not available"

    # ------------------------------------------------------------------
    # Step 7c — m4_summary.json
    # ------------------------------------------------------------------
    score_dict: dict[str, dict] = {}
    for vid in tracks:
        pd_val = sc.pd(vid)
        rmse_val = sc.rmse(vid)
        if not np.isnan(pd_val):
            score_dict[vid] = {"pd": float(pd_val), "rmse_m": float(rmse_val)}

    summary: dict = {
        "pfa": args.pfa,
        "cfar_threshold": float(thr),
        "n_detections": int(len(run.dets)),
        "n_above_threshold": int(len(dets)),
        "suppression_enabled": bool(args.suppress),
        "provenance": run.provenance.to_dict(),
        "score": score_dict,
        "classifier": {
            "available": classifier_available,
            "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
            "roc_auc": float(roc_auc) if roc_auc is not None and not np.isnan(roc_auc) else None,
            "roc_skip_reason": roc_skip_reason,
        },
    }

    (out / "m4_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"  → {out / 'm4_summary.json'}", flush=True)
    print(f"\nDone. Outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
