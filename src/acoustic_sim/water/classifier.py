"""Harbor detect–locate–classify: combined CNN classifier + CVAE anomaly scorer.

Wraps :class:`~acoustic_sim.ml.acoustic_classifier.AcousticClassifier` and
:class:`~acoustic_sim.ml.anomaly_detector.ConvVAE` into a single
:class:`HarborClassifier` that runs both networks on a mel-spectrogram and
returns a structured result dict.

Example
-------
::

    clf = HarborClassifier.load(
        cnn_path="output/harbor/cnn_classifier.pt",
        cvae_path="output/harbor/cvae_anomaly.pt",
    )
    result = clf.classify(mel_spec)
    # result = {
    #     "label": "uuv",
    #     "label_idx": 1,
    #     "cnn_probs": array([0.05, 0.90, 0.05]),
    #     "anomaly_score": 3.7,
    # }

If the CVAE checkpoint does not exist (not yet trained), ``load`` sets the
internal CVAE to ``None`` and :meth:`classify` returns ``anomaly_score=float("nan")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from acoustic_sim.ml.acoustic_classifier import AcousticClassifier
from acoustic_sim.ml.anomaly_detector import ConvVAE


class HarborClassifier:
    """Combined CNN + CVAE classifier for the harbor DLC pipeline.

    Parameters
    ----------
    cnn : AcousticClassifier
        Trained CNN in eval mode.
    cvae : ConvVAE or None
        Trained CVAE in eval mode, or ``None`` if not available.
    label_names_ : list[str]
        Human-readable class names in label-index order
        (e.g. ``["ship", "uuv", "diver"]``).
    device : torch.device
        Device on which both models live.
    """

    def __init__(
        self,
        cnn: AcousticClassifier,
        cvae: ConvVAE | None,
        label_names_: list[str],
        device: torch.device,
    ) -> None:
        self._cnn = cnn
        self._cvae = cvae
        self._label_names = label_names_
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def label_names(self) -> list[str]:
        """Human-readable class names in label-index order."""
        return list(self._label_names)

    @classmethod
    def load(
        cls,
        cnn_path: str | Path,
        cvae_path: str | Path,
        device: str = "cpu",
    ) -> "HarborClassifier":
        """Load both checkpoints and return a ready-to-use classifier.

        Parameters
        ----------
        cnn_path : str or Path
            Path to the CNN checkpoint saved by ``harbor_train_classifier.py``.
            Must exist.
        cvae_path : str or Path
            Path to the CVAE checkpoint saved by ``harbor_train_anomaly.py``.
            If the file does not exist the CVAE is set to ``None`` and
            :meth:`classify` will return ``anomaly_score=float("nan")``.
        device : str
            PyTorch device string (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        HarborClassifier
        """
        dev = torch.device(device)

        # -- CNN (required) --
        cnn_path = Path(cnn_path)
        if not cnn_path.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {cnn_path}\n"
                "Run harbor_train_classifier.py first."
            )
        cnn_ckpt: dict[str, Any] = torch.load(
            str(cnn_path), map_location=dev, weights_only=True
        )
        label_names: list[str] = list(cnn_ckpt["label_names"])
        n_classes: int = int(cnn_ckpt["n_classes"])
        cnn = AcousticClassifier(n_classes=n_classes)
        cnn.load_state_dict(cnn_ckpt["model_state"])
        cnn.to(dev).eval()

        # -- CVAE (optional) --
        cvae_path = Path(cvae_path)
        cvae: ConvVAE | None = None
        if cvae_path.exists():
            cvae_ckpt: dict[str, Any] = torch.load(
                str(cvae_path), map_location=dev, weights_only=True
            )
            latent_dim: int = int(cvae_ckpt.get("latent_dim", 32))
            cvae = ConvVAE(latent_dim=latent_dim)
            cvae.load_state_dict(cvae_ckpt["model_state"])
            cvae.to(dev).eval()

        return cls(cnn=cnn, cvae=cvae, label_names_=label_names, device=dev)

    def classify(self, mel_spec: np.ndarray) -> dict[str, Any]:
        """Classify a single mel-spectrogram.

        Parameters
        ----------
        mel_spec : (1, n_mels, T) float32 numpy array
            Single mel-spectrogram window as produced by the dataset builder.

        Returns
        -------
        dict with keys:
            ``label`` : str — predicted class name.
            ``label_idx`` : int — predicted class index.
            ``cnn_probs`` : (n_classes,) float32 array — softmax probabilities.
            ``anomaly_score`` : float — CVAE reconstruction error, or
                ``float("nan")`` if the CVAE is not loaded.
        """
        # Ensure shape (1, 1, n_mels, T).
        x = _to_tensor(mel_spec, self._device)

        # -- CNN forward --
        with torch.no_grad():
            logits = self._cnn(x)                          # (1, n_classes)
            probs = F.softmax(logits, dim=1).squeeze(0)    # (n_classes,)
        label_idx = int(probs.argmax().item())
        label = self._label_names[label_idx]
        cnn_probs = probs.cpu().numpy().astype(np.float32)

        # -- CVAE anomaly score --
        if self._cvae is not None:
            score, _ = self._cvae.compute_anomaly_score(x)
        else:
            score = float("nan")

        return {
            "label": label,
            "label_idx": label_idx,
            "cnn_probs": cnn_probs,
            "anomaly_score": float(score),
        }

    def classify_batch(self, mel_specs: np.ndarray) -> list[dict[str, Any]]:
        """Classify a batch of mel-spectrograms.

        Parameters
        ----------
        mel_specs : (B, 1, n_mels, T) float32 numpy array
            Batch of mel-spectrogram windows.

        Returns
        -------
        list of dicts, one per example, with the same keys as
        :meth:`classify`.
        """
        # Ensure shape (B, 1, n_mels, T).
        if mel_specs.ndim == 3:
            # Treat as single example (1, n_mels, T) → add batch dim.
            mel_specs = mel_specs[np.newaxis]
        x_batch = torch.from_numpy(mel_specs.astype(np.float32)).to(self._device)

        # -- Batched CNN forward --
        with torch.no_grad():
            logits = self._cnn(x_batch)                            # (B, n_classes)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy()   # (B, n_classes)

        results: list[dict[str, Any]] = []
        for i in range(len(mel_specs)):
            probs_i = probs_batch[i].astype(np.float32)
            label_idx = int(np.argmax(probs_i))
            label = self._label_names[label_idx]

            # CVAE scores one sample at a time (uses compute_anomaly_score).
            if self._cvae is not None:
                x_i = x_batch[i : i + 1]
                score_i, _ = self._cvae.compute_anomaly_score(x_i)
            else:
                score_i = float("nan")

            results.append({
                "label": label,
                "label_idx": label_idx,
                "cnn_probs": probs_i,
                "anomaly_score": float(score_i),
            })
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(mel_spec: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a (1, n_mels, T) or (n_mels, T) array to a (1,1,n_mels,T) tensor."""
    arr = np.asarray(mel_spec, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis]    # (n_mels, T) → (1, 1, n_mels, T)
    elif arr.ndim == 3:
        arr = arr[np.newaxis]                # (1, n_mels, T) → (1, 1, n_mels, T)
    # arr is now (1, 1, n_mels, T)
    return torch.from_numpy(arr).to(device)
