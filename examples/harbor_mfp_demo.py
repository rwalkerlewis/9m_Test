"""
Simplified harbor matched field processing demo.
Constant sound speed, flat 2D harbor, DAS fiber across a channel, AIS ship as illuminator.
Produces figures in output/harbor_demo/figs

Usage::

    python examples/harbor_mfp_demo.py
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "harbor_demo", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------- geometry ----------------
C = 1500.0                       # m/s
LX, LY = 1500.0, 300.0           # channel length, width (m)
DEPTH = 16.0
GRID = 10.0
gx = np.arange(GRID/2, LX, GRID); gy = np.arange(GRID/2, LY, GRID)
GX, GY = np.meshgrid(gx, gy)     # (ny, nx)
cells = np.c_[GX.ravel(), GY.ravel()]           # (ncell, 2)
NC = len(cells)

# fiber 1: diagonal channel crossing; fiber 2: short run along the north seawall. 2 m channel spacing
f0, f1 = np.array([500.0, 0.0]), np.array([1000.0, 300.0])
s0, s1 = np.array([1050.0, 300.0]), np.array([1350.0, 300.0])
def line(a, b):
    n = int(np.linalg.norm(b - a) / 2.0)
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    return pts, np.tile((b - a) / np.linalg.norm(b - a), (n, 1))
c1, t1 = line(f0, f1); c2, t2 = line(s0, s1)
chan = np.vstack([c1, c2]); tang = np.vstack([t1, t2]); nch = len(chan); n1 = len(c1)

# clutter: fixed scatterers on pier faces (x, y, TS dB)
clutter = np.array([[300.0, 0.0, 0.0], [900.0, 300.0, 0.0], [1200.0, 300.0, 0.0]])

# ---------------- replica model ----------------
def spread(r):
    r = np.maximum(r, 1.0)
    return np.where(r < DEPTH, 1.0 / r, 1.0 / (np.sqrt(DEPTH) * np.sqrt(r)))

def replica_tables(pts):
    """pts (n,2) -> delay (n,nch), gain (n,nch) including DAS cos^2 directivity and spreading"""
    d = pts[:, None, :] - chan[None, :, :]
    r = np.linalg.norm(d, axis=2)
    cosang = np.einsum('ijk,jk->ij', d, tang) / np.maximum(r, 1e-6)
    gain = (cosang**2) * spread(r)
    return r / C, gain

D_cell, G_cell = replica_tables(cells)

# ---------------- band and snapshots ----------------
FS = 2000.0
TSNAP = 0.5                       # snapshot length (s)
NSNAP = 11                        # 3 s window, 50 % overlap
freqs = np.arange(40.0, 800.0, 20.0)   # 38 bins
NF = len(freqs)

def db2amp(db): return 10 ** (db / 20)

def ferry_spec(f):   # source level per Hz, dB re 1 uPa at 1 m, broadband with blade rate lines
    base = 160.0 - 6 * np.log2(np.maximum(f, 100) / 100)
    lines = np.zeros_like(f)
    for h in range(1, 20):
        lines += 12 * np.exp(-0.5 * ((f - 12 * h) / 2.0) ** 2)
    return base + lines

def uuv_spec(f):     # tonal thruster, 400 Hz and harmonic
    s = np.full_like(f, -np.inf)
    for fh, lv in ((400.0, 135.0), (800.0, 115.0)):
        s = np.logaddexp(s * np.log(10) / 20, (lv - 60 * np.abs(f - fh) / 20) * np.log(10) / 20) * 20 / np.log(10)
    return s

NOISE_DB = 65.0  # harbor ambient per Hz, quiet night
UUV_TS = -20.0

def synth(ferry_xy, uuv_xy, uuv_motor, snap_seed):
    """Return data d[NSNAP, NF, nch] (complex pressure per Hz) for one 3 s window."""
    r = np.random.default_rng(snap_seed)
    d = np.zeros((NSNAP, NF, nch), complex)
    # ferry direct
    Df, Gf = replica_tables(ferry_xy[None]); Df, Gf = Df[0], Gf[0]
    Sf = db2amp(ferry_spec(freqs))
    for s in range(NSNAP):
        ph = np.exp(1j * r.uniform(0, 2 * np.pi, NF))
        src = Sf * ph                                                   # (NF,)
        d[s] += src[:, None] * Gf[None, :] * np.exp(-2j * np.pi * freqs[:, None] * Df[None, :])
        # bistatic echoes off UUV and clutter
        for (sx, sy, ts) in np.vstack([clutter, [[*uuv_xy, UUV_TS]]]):
            rs = np.linalg.norm(ferry_xy - [sx, sy]); gs = spread(rs); ds = rs / C
            Dc, Gc = replica_tables(np.array([[sx, sy]])); Dc, Gc = Dc[0], Gc[0]
            d[s] += (src * gs * db2amp(ts) * np.exp(-2j * np.pi * freqs * ds))[:, None] * Gc[None, :] \
                    * np.exp(-2j * np.pi * freqs[:, None] * Dc[None, :])
        # UUV own radiated noise
        if uuv_motor:
            Du, Gu = replica_tables(uuv_xy[None]); Du, Gu = Du[0], Gu[0]
            Su = db2amp(uuv_spec(freqs)) * np.exp(1j * r.uniform(0, 2 * np.pi, NF))
            d[s] += Su[:, None] * Gu[None, :] * np.exp(-2j * np.pi * freqs[:, None] * Du[None, :])
        # noise
        d[s] += db2amp(NOISE_DB) * (r.standard_normal((NF, nch)) + 1j * r.standard_normal((NF, nch))) / np.sqrt(2)
    return d

# ---------------- processors ----------------
def bartlett(d, D, G, fidx=None):
    """Bartlett MFP: sum_f sum_snap |w^H d|^2 / |w|^2.  d (NSNAP,NF,nch). returns (ncell,)"""
    fidx = range(NF) if fidx is None else fidx
    P = np.zeros(len(D))
    for fi in fidx:
        W = G * np.exp(2j * np.pi * freqs[fi] * D)          # conj replica (ncell,nch)
        wn = np.sum(G**2, axis=1)
        y = W @ d[:, fi, :].T                                # (ncell, NSNAP)
        P += np.sum(np.abs(y) ** 2, axis=1) / wn
    return P / (len(fidx) * d.shape[0])

def mvdr(d, D, G, fidx=None, load=0.05):
    fidx = range(NF) if fidx is None else fidx
    P = np.zeros(len(D))
    for fi in fidx:
        X = d[:, fi, :]                                      # (NSNAP,nch)
        K = X.T @ X.conj() / X.shape[0]
        K += load * np.trace(K).real / nch * np.eye(nch)
        Ki = np.linalg.inv(K)
        W = G * np.exp(-2j * np.pi * freqs[fi] * D)          # replica (ncell,nch)
        q = np.einsum("ij,jk,ik->i", W.conj(), Ki, W).real
        wn = np.sum(G**2, axis=1)
        P += wn / q
    return P / len(fidx)

def bistatic(d_raw, d_res, D, G, s_pt, fidx=None):
    """Bistatic MFP, coherent over frequency.
    Replica for cell x = (illuminator spectrum, measured by beamforming at s) x path s->x x path x->channel.
    d_raw: data with illuminator present (for the reference), d_res: data with direct path and clutter removed."""
    fidx = range(NF) if fidx is None else fidx
    Ds, Gs = replica_tables(s_pt[None]); Ds, Gs = Ds[0], Gs[0]
    rsx = np.linalg.norm(cells - s_pt, axis=1)                                    # |s - x| per cell
    tsx = rsx / C; gsx = spread(rsx)
    Z = np.zeros((len(D), d_raw.shape[0]), complex); norm = np.zeros(len(D))
    for fi in fidx:
        ws = Gs * np.exp(-2j * np.pi * freqs[fi] * Ds)
        r = (d_raw[:, fi, :] @ ws.conj()) / np.sum(Gs**2)                      # (NSNAP,) source spectrum estimate
        W = G * np.exp(2j * np.pi * freqs[fi] * D)                              # conj replica x->channel (ncell,nch)
        y = W @ d_res[:, fi, :].T                                               # (ncell,NSNAP)
        path = gsx * np.exp(-2j * np.pi * freqs[fi] * tsx)                      # s->x (ncell,)
        Z += y * np.conj(path)[:, None] * np.conj(r)[None, :]
        norm += np.sum(G**2, axis=1) * (gsx**2) * np.mean(np.abs(r) ** 2)
    return np.sum(np.abs(Z) ** 2, axis=1) / (norm * d_raw.shape[0]) ** 1 / len(fidx)

def refine(d, guess, half=15.0, step=0.25):
    """Fine Bartlett search around an AIS guess."""
    xs = np.arange(guess[0] - half, guess[0] + half + 1e-6, step)
    ys = np.arange(guess[1] - half, guess[1] + half + 1e-6, step)
    XX, YY = np.meshgrid(xs, ys); pts = np.c_[XX.ravel(), YY.ravel()]
    D, G = replica_tables(pts)
    P = bartlett(d, D, G, fidx=range(0, NF, 3))
    return pts[np.argmax(P)]

def project_out(d, pts_list, half=0.75, n=3):
    """Jointly remove everything coherent with replicas in small patches around each point in pts_list
    (one subspace projection per frequency; joint so the removals do not interfere)."""
    pts = []
    for pt in np.atleast_2d(pts_list):
        xs = np.linspace(pt[0] - half, pt[0] + half, n); ys = np.linspace(pt[1] - half, pt[1] + half, n)
        XX, YY = np.meshgrid(xs, ys); pts.append(np.c_[XX.ravel(), YY.ravel()])
    pts = np.vstack(pts)
    D, G = replica_tables(pts)
    out = d.copy()
    for fi in range(NF):
        W = (G * np.exp(-2j * np.pi * freqs[fi] * D)).T          # (nch, npts)
        Q, _ = np.linalg.qr(W)
        out[:, fi, :] -= (d[:, fi, :] @ Q.conj()) @ Q.T
    return out

def todb(P): return 10 * np.log10(np.maximum(P, 1e-30))

# ---------------- plotting helpers ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
INK, MUTED = "#1f2937", "#6b7280"

def harbor_axes(ax, title):
    ax.set_facecolor("#dbe4ee")
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, lw=1.2, ec=INK))
    ax.plot(c1[:, 0], c1[:, 1], color="#111111", lw=2.2, label="DAS fiber"); ax.plot(c2[:, 0], c2[:, 1], color="#111111", lw=2.2)
    for cx, cy, _ in clutter:
        ax.plot(cx, cy, "s", ms=7, mfc="#9ca3af", mec=INK, label="fixed scatterer (pier)")
    ax.set_xlim(-20, LX + 20); ax.set_ylim(-30, LY + 30); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=22)
    ax.tick_params(colors=MUTED)

def show_map(ax, P, title, ferry, uuv, refined=None, ais=None, vmin=None, cmap="viridis"):
    Z = todb(P).reshape(GY.shape); Z -= Z.max()
    vmin = -30 if vmin is None else vmin
    im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap=cmap, vmin=vmin, vmax=0, aspect="equal")
    ax.plot(c1[:, 0], c1[:, 1], color="white", lw=1.5); ax.plot(c2[:, 0], c2[:, 1], color="white", lw=1.5)
    for cx, cy, _ in clutter: ax.plot(cx, cy, "s", ms=6, mfc="none", mec="white")
    ax.plot(*ferry, "^", ms=10, mfc="none", mec="white", mew=1.5, label="ferry (true)")
    ax.plot(*uuv, "o", ms=10, mfc="none", mec="#ff7f0e", mew=1.8, label="UUV (true)")
    if ais is not None: ax.plot(*ais, "x", ms=8, color="white", mew=1.5, label="ferry AIS")
    if refined is not None: ax.plot(*refined, "+", ms=12, color="#ffd60a", mew=2, label="ferry refined")
    ax.set_xlim(0, LX); ax.set_ylim(0, LY)
    ax.set_title(title, loc="left", color=INK, fontweight="bold"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.tick_params(colors=MUTED)
    return im

def dedup_legend(ax, **kw):
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l): seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, **kw)

# ---------------- scene ----------------
ferry0 = np.array([600.0, 150.0]); ferry_v = np.array([4.0, 0.0])      # 8 kn down channel
uuv0 = np.array([850.0, 40.0]);   uuv_v = np.array([0.0, 1.0])         # 2 kn across

# ===== Fig 1: geometry =====
fig, ax = plt.subplots(figsize=(10, 3.4))
harbor_axes(ax, "1. Scene: 1500 m x 300 m channel, constant c = 1500 m/s, 16 m deep, two fiber lines")
t = np.arange(0, 11, 0.5)
ax.plot(ferry0[0] + ferry_v[0] * t, ferry0[1] + ferry_v[1] * t, "-", color="#1d4ed8", lw=2, label="ferry track, 8 kn (AIS)")
ax.plot(*ferry0, "^", ms=9, color="#1d4ed8")
ax.plot(uuv0[0] + uuv_v[0] * t, uuv0[1] + uuv_v[1] * t, "-", color="#ff7f0e", lw=2, label="UUV track, 2 kn")
ax.plot(*uuv0, "o", ms=8, color="#ff7f0e")
ax.text(chan[0, 0] - 10, -18, f"fiber 1: {n1} channels, {np.linalg.norm(f1-f0):.0f} m crossing; fiber 2: {nch-n1} channels, {np.linalg.norm(s1-s0):.0f} m along the seawall", fontsize=8, color=INK)
dedup_legend(ax, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_scene.png"); plt.close(fig)

# ===== window 0 data, two scenarios =====
ferry = ferry0.copy(); uuv = uuv0.copy()
ais = ferry + np.array([8.0, -6.0])          # AIS error
d_on = synth(ferry, uuv, True, 1)
d_off = synth(ferry, uuv, False, 1)

# ===== Fig 2: raw data, channel vs time =====
fig, ax = plt.subplots(figsize=(10, 3.4))
# time series of one snapshot: place band into a full spectrum and irfft
nfft = int(FS * TSNAP); spec = np.zeros((nfft // 2 + 1, nch), complex)
for fi, f in enumerate(freqs): spec[int(round(f * TSNAP)), :] = d_on[0, fi, :]
ts = np.fft.irfft(spec, n=nfft, axis=0)
ts /= np.abs(ts).max()
ax.imshow(ts.T, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4, extent=[0, TSNAP, 0, nch], origin="lower")
ax.set_xlabel("time (s)"); ax.set_ylabel("DAS channel"); ax.tick_params(colors=MUTED)
ax.set_title("2. What the fiber sees in one 0.5 s snapshot: ferry noise dominates, no target visible by eye", loc="left", color=INK, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_raw.png"); plt.close(fig)

# ===== Fig 3: passive broadband Bartlett, ferry located and refined =====
P_pass = bartlett(d_on, D_cell, G_cell)
ref = refine(d_on, ais)
fig, ax = plt.subplots(figsize=(10, 3.4))
im = show_map(ax, P_pass, "3. Passive MFP, Bartlett, 40 to 800 Hz: ferry found, AIS position refined",
              ferry, uuv, refined=ref, ais=ais)
dedup_legend(ax, loc="upper right", labelcolor="white")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="dB rel. peak")
ax.text(20, 270, f"AIS error {np.linalg.norm(ais-ferry):.1f} m, refined error {np.linalg.norm(ref-ferry):.1f} m",
        color="white", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_passive.png"); plt.close(fig)

# ===== Fig 4: passive narrowband at 400 Hz, UUV motor on =====
fi400 = [int(np.argmin(np.abs(freqs - 400)))]
d_on_res = project_out(d_on, np.vstack([ref, clutter[:, :2]]))
P_400 = bartlett(d_on_res, D_cell, G_cell, fidx=fi400)
mask400 = np.linalg.norm(cells - ref, axis=1) < 60; P_400[mask400] = P_400[~mask400].min()
fig, ax = plt.subplots(figsize=(10, 3.4))
im = show_map(ax, P_400, "4. Passive MFP, 400 Hz bin, ferry and piers projected out: UUV tonal (motor ON)", ferry, uuv)
dedup_legend(ax, loc="upper right", labelcolor="white")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="dB rel. peak")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_passive_400.png"); plt.close(fig)

# ===== Fig 5: bistatic, motor off =====
d_res = project_out(d_off, ref)                      # remove ferry direct path
P_b_raw = bistatic(d_off, d_res, D_cell, G_cell, ref)
d_res2 = project_out(d_off, np.vstack([ref, clutter[:, :2]]))   # remove ferry and stored clutter jointly
P_b = bistatic(d_off, d_res2, D_cell, G_cell, ref)
mask = np.linalg.norm(cells - ref, axis=1) < 60     # ignore cells at the illuminator
P_b_raw[mask] = P_b_raw[~mask].min(); P_b[mask] = P_b[~mask].min()

fig, axs = plt.subplots(2, 1, figsize=(10, 6.6))
show_map(axs[0], P_b_raw, "5a. Bistatic MFP (coherent over 40 to 800 Hz), motor OFF, ferry direct path removed", ferry, uuv, refined=ref)
im = show_map(axs[1], P_b, "5b. Same, with the stored pier clutter removed: only the UUV echo remains", ferry, uuv, refined=ref)
dedup_legend(axs[0], loc="upper right", labelcolor="white")
fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="dB rel. peak")
fig.savefig(f"{OUT}/fig5_bistatic.png", bbox_inches="tight"); plt.close(fig)

# ===== Fig 6: Bartlett vs MVDR, zoom on ferry =====
fsub = range(0, NF, 4)
P_bz = bartlett(d_on, D_cell, G_cell, fidx=fsub)
P_mv = mvdr(d_on, D_cell, G_cell, fidx=fsub)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, P, name in zip(axs, (P_bz, P_mv), ("Bartlett", "Minimum variance")):
    Z = todb(P).reshape(GY.shape); Z -= Z.max()
    im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap="viridis", vmin=-20, vmax=0)
    ax.plot(*ferry, "^", ms=10, mfc="none", mec="white", mew=1.5)
    ax.set_xlim(450, 800); ax.set_ylim(50, 250); ax.set_aspect("equal")
    ax.set_title(f"6. {name} processor, same data", loc="left", color=INK, fontweight="bold", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.tick_params(colors=MUTED)
fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02, label="dB rel. peak")
fig.savefig(f"{OUT}/fig6_bartlett_vs_mv.png", bbox_inches="tight"); plt.close(fig)

# ===== Fig 7: occupancy grid over 20 windows, motor off =====
from scipy.ndimage import gaussian_filter
O = np.full(NC, 1.0 / NC)
snaps = {}
NW = 20; DT = 0.5
truth = []
for k in range(NW):
    ferry = ferry0 + ferry_v * k * DT; uuv = uuv0 + uuv_v * k * DT; truth.append(uuv.copy())
    ais = ferry + rng.normal(0, 6, 2)
    d = synth(ferry, uuv, False, 100 + k)
    r_k = refine(d, ais)
    dr = project_out(d, np.vstack([r_k, clutter[:, :2]]))
    Pb = bistatic(d, dr, D_cell, G_cell, r_k)
    m = np.linalg.norm(cells - r_k, axis=1) < 60; Pb[m] = np.median(Pb[~m])
    # likelihood ratio: map power relative to the noise distribution of the map
    z = (todb(Pb) - np.median(todb(Pb[~m]))) / 1.4826 / np.median(np.abs(todb(Pb[~m]) - np.median(todb(Pb[~m]))))
    L = np.exp(np.clip(0.15 * z, -3, 4))
    # predict: diffuse with 1 m/s max speed
    O = gaussian_filter(O.reshape(GY.shape), sigma=1.0 * DT / GRID + 0.3).ravel()
    O = O * L; O /= O.sum()
    if k in (0, 4, 9, 19): snaps[k] = (O.copy(), ferry.copy(), uuv.copy())

fig, axs = plt.subplots(2, 2, figsize=(11, 3.6)); fig.subplots_adjust(hspace=0.55, wspace=0.12)
for ax, (k, (Ok, fk, uk)) in zip(axs.ravel(), snaps.items()):
    Z = Ok.reshape(GY.shape); Z = Z / Z.max()
    im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap="magma", vmin=0, vmax=1)
    ax.plot(c1[:, 0], c1[:, 1], color="white", lw=1); ax.plot(c2[:, 0], c2[:, 1], color="white", lw=1)
    tr = np.array(truth[:k + 1]); ax.plot(tr[:, 0], tr[:, 1], "-", color="#ff7f0e", lw=1.2)
    ax.plot(*uk, "o", ms=9, mfc="none", mec="#ff7f0e", mew=1.8)
    ax.plot(*fk, "^", ms=8, mfc="none", mec="white", mew=1.2)
    ax.set_title(f"7. Occupancy after window {k+1} (t = {(k+1)*DT:.1f} s), motor OFF", loc="left", color=INK, fontweight="bold", fontsize=9)
    ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.tick_params(colors=MUTED, labelsize=8)
fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="occupancy, rel. peak")
fig.savefig(f"{OUT}/fig7_occupancy.png", bbox_inches="tight"); plt.close(fig)

print("done", OUT)
