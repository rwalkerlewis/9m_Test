#!/usr/bin/env python3
"""Step 5, removing the ships' own sound: making the echoes visible.

A silent body only reflects ship noise, and its echo is 10 to 20 dB weaker than the ship's direct sound on
the same channels.  For every strong track the chain forms a reference (the beam toward the track) and
removes from every channel the part that is a delayed copy of it (ECA, the delays following the moving
ship).  Left: one DAS channel before and after; middle: its spectrum; right: how much was removed on every
channel.

    python examples/harbor/06_walkthrough/05_direct_path_removal.py [--dataset bistatic] [--cuda]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

import _common as c

a = c.args(__doc__, default="bistatic")
obs = c.observation(a.dataset)
run = c.products(a)
fs, band = obs.fs, run["pc"]["band"]
clean, s0 = run["echo"]["clean"], run["echo"]["clean_start"]
raw = obs.traces[:, s0:s0 + clean.shape[1]].astype(float)
t_first = min(f[0] for f in run["echo"]["frames"])
lo = int((t_first + 1.0) * fs) - s0
raw, clean = raw[:, lo:], clean[:, lo:]
sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
rb, cb = sosfiltfilt(sos, raw, axis=1), sosfiltfilt(sos, clean, axis=1)
das = obs.geometry.slices[obs.geometry.cables[0].name]
e_raw, e_cl = np.sum(rb[das] ** 2, axis=1), np.sum(cb[das] ** 2, axis=1)
db = 10 * np.log10(np.maximum(e_raw, 1e-30) / np.maximum(e_cl, 1e-30))
ch = int(np.argmax(e_raw))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
t = (s0 + lo + np.arange(rb.shape[1])) / fs
w = slice(0, int(0.25 * fs))
axes[0].plot(t[w], rb[ch, w], color="0.6", lw=1, label="recorded")
axes[0].plot(t[w], cb[ch, w], color=c.HIT, lw=1, label="direct paths removed")
axes[0].set_xlabel("time [s]")
axes[0].set_title(f"DAS channel {ch} ({band[0]:.0f}-{band[1]:.0f} Hz), 0.25 s", fontsize=10)
axes[0].legend(fontsize=8)
f, p_raw = welch(raw[das.start + ch], fs=fs, nperseg=int(0.5 * fs))
_, p_cl = welch(clean[das.start + ch], fs=fs, nperseg=int(0.5 * fs))
m = (f >= band[0]) & (f <= band[1])
axes[1].plot(f[m], 10 * np.log10(p_raw[m]), color="0.6", label="recorded")
axes[1].plot(f[m], 10 * np.log10(p_cl[m]), color=c.HIT, label="direct paths removed")
axes[1].set_xlabel("frequency [Hz]")
axes[1].set_ylabel("power [dB]")
axes[1].set_title("Spectrum of the same channel", fontsize=10)
axes[1].legend(fontsize=8)
axes[2].plot(np.arange(len(db)), db, ".", color=c.TRACK, ms=3)
axes[2].axhline(np.median(db), color=c.MISS, ls="--", label=f"median {np.median(db):.1f} dB")
axes[2].set_xlabel("DAS channel")
axes[2].set_ylabel("energy removed [dB]")
axes[2].set_title("Removed on every channel (direct paths, mostly)", fontsize=10)
axes[2].legend(fontsize=8)
fig.suptitle(f"Step 5 - the direct sound of the tracked radiators removed ({a.dataset}; tracks used: "
             f"{', '.join(run['strong'])})", fontsize=12)
fig.tight_layout()
c.save(fig, c.out_dir(a) / "05_direct_path_removal.png", "The removal uses the passive tracks only (no AIS, no truth).")
