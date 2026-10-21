#!/usr/bin/env python3
"""Step 1, the recording: what the system gets.

The chart (water and land), the sensors (a DAS cable and four hydrophones; the cable is in strain mode, the second
difference over its gauge length, on the bistatic and multistatic recordings and in pressure mode on the simple
and the 5 m harbor ones, as the suptitle says) and the traces: the DAS
gather (every channel against time) and the spectrogram of one hydrophone.  Ships, boats and bodies are
somewhere in the water; the dashed lines show where (from the simulation, for comparison only).

    python examples/harbor/06_walkthrough/01_recording.py [--dataset multistatic]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram

import _common as c

a = c.args(__doc__)
obs, tru = c.observation(a.dataset), c.truth(a.dataset)
lay, fs = obs.geometry, obs.fs
band = c.config(a.dataset).band
cab = lay.cables[0]
mode = f"strain DAS, {cab.gauge_length:g} m gauge" if cab.is_strain else "pressure DAS"
F = c.Figures(a, "01_recording", f"Step 1 - the recording ({a.dataset}): {lay.n_das} channels of {mode} + {lay.n_hydrophones} "
              f"hydrophones, {fs:.0f} Hz, {obs.traces.shape[1] / fs:.0f} s", c.NOTE)
fig, ax = F.new("chart")
c.chart(ax, obs)
c.truth_overlay(ax, tru)
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Chart and sensors (dashed: true tracks, blue radiating, orange silent)", fontsize=10)

sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
das = lay.slices[lay.cables[0].name]
t0, t1 = (8.0, 10.0) if obs.traces.shape[1] / fs > 12 else (2.0, 4.0)
seg = sosfiltfilt(sos, obs.traces[das, int(t0 * fs):int(t1 * fs)].astype(float), axis=1)
fig, ax = F.new("gather")
v = np.percentile(np.abs(seg), 99)
ax.imshow(seg.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, origin="upper",
          extent=(0, seg.shape[0], t1, t0))
ax.set_xlabel("DAS channel")
ax.set_ylabel("time [s]")
ax.set_title(f"DAS gather, {band[0]:.0f}-{band[1]:.0f} Hz, {t1 - t0:.0f} s\n(ship noise arriving along the cable)", fontsize=10)

h = lay.slices["hydrophones"].start
f, t, S = spectrogram(obs.traces[h].astype(float), fs=fs, nperseg=int(0.5 * fs), noverlap=int(0.25 * fs))
m = (f >= band[0] * 0.5) & (f <= band[1] * 1.2)
fig, ax = F.new("spectrogram", size=(10, 5))
pm = ax.pcolormesh(t, f[m], 10 * np.log10(S[m] + 1e-30), shading="auto", cmap="magma")
fig.colorbar(pm, ax=ax, label="dB")
ax.set_xlabel("time [s]")
ax.set_ylabel("frequency [Hz]")
ax.set_title("Hydrophone 1: spectrogram [dB]", fontsize=10)
fig, ax = F.new("trace", size=(10, 4.5))
tt = np.arange(obs.traces.shape[1]) / fs
ax.plot(tt, obs.traces[h], lw=0.4, color=c.SENSOR)
ax.set_xlabel("time [s]")
ax.set_ylabel("pressure [Pa]")
ax.set_title("Hydrophone 1: trace", fontsize=10)
F.save_all()
