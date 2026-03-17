"""FNO inference wrapper — drop-in replacement for FDTD forward model.

Loads a trained :class:`~acoustic_sim.ml.fno.AcousticFNO` checkpoint and
provides the same interface as the FDTD solver so it can be used directly
in existing pipelines.

Usage
-----
    from acoustic_sim.ml.fno_inference import FNOForwardModel

    fno = FNOForwardModel("checkpoints/fno/fno_final.pt")
    traces = fno.predict(
        velocity_field=vel,       # (ny, nx) numpy array
        grid_x=gx, grid_y=gy,    # 1-D coordinate arrays
        receiver_positions=recv,  # (n_recv, 2) or (n_recv, 3)
        source_x=10.0,
        source_y=20.0,
        source_freq=50.0,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from acoustic_sim.ml.fno import AcousticFNO, prepare_fno_input


class FNOForwardModel:
    """Acoustic forward model backed by a trained Fourier Neural Operator.

    Parameters
    ----------
    checkpoint : str or Path
        Path to the ``.pt`` state-dict file.
    modes : int
        Number of Fourier modes (must match training config).
    width : int
        Channel width (must match training config).
    n_layers : int
        Number of Fourier layers (must match training config).
    n_time_steps : int
        Trace length (must match training config).
    device : str or None
        ``"cuda"`` or ``"cpu"``. Auto-detected if *None*.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        n_time_steps: int = 1000,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_time_steps = n_time_steps

        self.model = AcousticFNO(
            modes1=modes,
            modes2=modes,
            width=width,
            n_layers=n_layers,
            n_time_steps=n_time_steps,
        )
        state = torch.load(
            str(checkpoint), map_location=self.device, weights_only=True,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        velocity_field: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        receiver_positions: np.ndarray,
        source_x: float,
        source_y: float,
        source_freq: float,
    ) -> np.ndarray:
        """Run the FNO forward model.

        Parameters
        ----------
        velocity_field : (ny, nx) numpy array
            Sound speed field in m/s.
        grid_x : (nx,) 1-D array of x-coordinates.
        grid_y : (ny,) 1-D array of y-coordinates.
        receiver_positions : (n_recv, 2) or (n_recv, 3)
            Receiver positions; only x, y are used.
        source_x, source_y : float
            Source position in physical coordinates.
        source_freq : float
            Dominant source frequency [Hz].

        Returns
        -------
        traces : (n_recv, n_time_steps) numpy array
        """
        recv = np.asarray(receiver_positions, dtype=np.float32)
        if recv.ndim == 1:
            recv = recv.reshape(1, -1)
        recv_xy = recv[:, :2]

        # Build 4-channel input.
        field = prepare_fno_input(
            torch.tensor(velocity_field, dtype=torch.float32),
            torch.tensor(grid_x, dtype=torch.float32),
            torch.tensor(grid_y, dtype=torch.float32),
            source_x, source_y, source_freq,
        )  # (4, ny, nx)

        # Add batch dimension and move to device.
        field = field.unsqueeze(0).to(self.device)
        recv_t = torch.tensor(recv_xy, dtype=torch.float32).unsqueeze(0).to(self.device)
        gx = torch.tensor(grid_x, dtype=torch.float32).to(self.device)
        gy = torch.tensor(grid_y, dtype=torch.float32).to(self.device)

        pred = self.model(field, recv_t, gx, gy)  # (1, n_recv, n_time_steps)
        return pred.squeeze(0).cpu().numpy()

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.model.parameters())
        return (
            f"FNOForwardModel(n_time_steps={self.n_time_steps}, "
            f"device={self.device}, params={params:,})"
        )
