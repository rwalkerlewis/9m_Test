"""Shared by the dataset scripts: the gathers read back from the FDTD cache (``--figures-only``, no FDTD run) and the
figures of the cable components, one figure per component."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt

from acoustic_sim.plotting import new_figure, save_figure
from acoustic_sim.water import render_vessel_gathers


def gathers_from_cache(cfg, env, layout, tracks, cache: Path, combined_bodies: bool = False) -> dict:
    """Every gather of the scenario read from the FDTD cache; raises if one is missing (no FDTD run)."""
    from acoustic_sim.water import fdtd_runner

    def no_fdtd(*a, **k):
        raise RuntimeError(f"an FDTD gather is missing from {cache}; --figures-only runs no FDTD")

    saved = fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run
    fdtd_runner.HarborFDTDBatch.run = fdtd_runner.HarborFDTDSolver.run = no_fdtd
    try:
        return render_vessel_gathers(cfg, env, layout, tracks, cache_dir=cache, combined_bodies=combined_bodies)
    finally:
        fdtd_runner.HarborFDTDBatch.run, fdtd_runner.HarborFDTDSolver.run = saved


def component_figures(layout, ds, sos, f_max: float, keys: list[tuple[str, str]], out: Path, rec: str) -> list[str]:
    """One figure per component ``(key in ds.components, file name)`` on the first cable: the band limited power of
    every channel in 50 ms windows, over 40 dB below its own 99.5th percentile.  Writes ``components_<name>.png``."""
    cable = layout.cables[0]
    sl = layout.slices[cable.name]
    unit = "strain" if getattr(cable, "das_mode", "pressure") == "strain" else "pressure"
    nwin = max(int(round(0.05 * ds.fs)), 1)
    written = []
    for key, name in keys:
        x = sosfiltfilt(sos, ds.components[key][sl].astype(float), axis=-1)
        m = x.shape[1] // nwin
        env_db = 10 * np.log10(np.maximum(np.mean(x[:, :m * nwin].reshape(len(x), m, nwin) ** 2, axis=-1), 1e-30))
        top = np.percentile(env_db, 99.5)
        fig, ax = new_figure((12.0, 8.0))
        im = ax.imshow(env_db, aspect="auto", origin="lower", cmap="viridis", vmin=top - 40, vmax=top,
                       extent=(0, m * nwin / ds.fs, 0, cable.length), interpolation="nearest")
        fig.colorbar(im, ax=ax, label=f"band limited {unit} power in 50 ms [dB re 1 trace unit squared]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance along cable [m]")
        ax.set_title(f"{rec}: cable component {key}, {0.25 * f_max:.0f} to {0.95 * f_max:.0f} Hz "
                     f"(rms {np.sqrt(np.mean(x ** 2)):.3g})")
        save_figure(fig, out / f"components_{name.replace(':', '_')}.png", recording=rec)
        written.append(name)
    return written
