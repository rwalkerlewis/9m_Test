"""The figure rule of the repository (``AGENTS.md``, Figures), enforced at save time.

Every figure in ``src``, ``examples`` and ``scripts`` is written by :func:`save_figure`, which refuses a figure that
breaks the rule instead of writing it:

* one axes (two when ``allow_two_panels`` says the panels share an axis and are read together); colour bar axes do not
  count as panels;
* at least 1600 pixels wide at the saved resolution (150 dpi: ``figsize`` at least 10.67 inches wide);
* no text below 11 pt (titles, labels, tick labels, legends, annotations, colour bar labels);
* both axis labels present on every panel, and a label on every colour bar;
* a title that names the recording (``recording=`` given, or the recording read from the output path).

It applies one style (one font family, tight layout, 150 dpi) and writes the PNG.  Importing this module sets the
matplotlib defaults so that new figures start at the sizes the rule asks for; :func:`new_figure` gives a figure and its
one axes at the default size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt

DPI = 150
MIN_WIDTH_PX = 1600
MIN_FONT_PT = 11.0
FONT_FAMILY = "DejaVu Sans"
DEFAULT_SIZE = (12.0, 9.0)

STYLE = {
    "font.family": FONT_FAMILY,
    "font.size": 12.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 11.0,
    "legend.title_fontsize": 11.0,
    "figure.titlesize": 13.0,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
}

# Folders under output/ whose next path component is the recording (or air scenario) a figure belongs to.
_RECORDING_PARENTS = ("recordings", "walkthrough", "runs", "air")


def apply_style() -> None:
    """Set the matplotlib defaults of the figure rule (called on import)."""
    matplotlib.rcParams.update(STYLE)


apply_style()


class FigureRuleError(ValueError):
    """A figure that breaks the figure rule; nothing was written."""


def new_figure(size: tuple[float, float] = DEFAULT_SIZE):
    """A new figure with one axes at a size the rule accepts.  Returns ``(fig, ax)``."""
    apply_style()
    return plt.subplots(figsize=size)


def recording_from_path(path: str | Path) -> str | None:
    """The recording a figure belongs to, read from its output path: the folder after ``recordings``, ``walkthrough``,
    ``runs`` or ``air`` (``output/recordings/simple/map.png`` -> ``simple``); None when the path names none."""
    parts = Path(path).parts
    for i, p in enumerate(parts[:-2]):
        if p in _RECORDING_PARENTS:
            return parts[i + 1]
    return None


def _colorbar_axes(fig) -> set:
    out = set()
    for ax in fig.axes:
        cb = getattr(ax, "_colorbar", None)
        if cb is not None or ax.get_label() == "<colorbar>":
            out.add(ax)
    return out


def _texts(fig):
    """Every visible text artist of the figure with a non empty string."""
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            yield t


def check_figure(fig, path: str | Path | None = None, *, allow_two_panels: bool = False,
                 recording: str | None = None) -> list[str]:
    """Every way *fig* breaks the figure rule (empty when it follows it)."""
    problems: list[str] = []
    fig.canvas.draw()                                   # tick labels exist only once the figure is drawn
    cbars = _colorbar_axes(fig)
    panels = [ax for ax in fig.axes if ax not in cbars and ax.get_visible()]
    limit = 2 if allow_two_panels else 1
    if len(panels) > limit:
        problems.append(f"{len(panels)} panels (at most {limit})")
    if not panels:
        problems.append("no axes")
    width_px = fig.get_figwidth() * DPI
    if width_px < MIN_WIDTH_PX - 0.5:
        problems.append(f"{width_px:.0f} pixels wide at {DPI} dpi (at least {MIN_WIDTH_PX})")
    for t in _texts(fig):
        if t.get_fontsize() < MIN_FONT_PT - 1e-6:
            problems.append(f"text {t.get_text()[:40]!r} at {t.get_fontsize():.1f} pt (at least {MIN_FONT_PT:g})")
    for i, ax in enumerate(panels):
        if not ax.get_xlabel().strip():
            problems.append(f"panel {i}: no x axis label")
        if not ax.get_ylabel().strip():
            problems.append(f"panel {i}: no y axis label")
    for ax in cbars:
        cb = getattr(ax, "_colorbar", None)
        label = ""
        if cb is not None:
            label = (ax.get_ylabel() if cb.orientation == "vertical" else ax.get_xlabel()) or ""
        else:
            label = ax.get_ylabel() or ax.get_xlabel()
        if not label.strip():
            problems.append("colour bar without a label")
    titles = [ax.get_title(loc) for ax in panels for loc in ("center", "left", "right")]
    if fig._suptitle is not None:
        titles.append(fig._suptitle.get_text())
    title = " ".join(t for t in titles if t.strip())
    if not title:
        problems.append("no title")
    rec = recording if recording is not None else (recording_from_path(path) if path is not None else None)
    if rec is not None and title and rec not in title:
        problems.append(f"title {title[:60]!r} does not name the recording {rec!r}")
    return problems


def save_figure(fig, path: str | Path, *, allow_two_panels: bool = False, recording: str | None = None,
                close: bool = True) -> Path:
    """Write *fig* to *path* (PNG, 150 dpi, tight layout, one font family) if it follows the figure rule, else raise
    :class:`FigureRuleError` listing every problem.  *allow_two_panels*: two panels that share an axis and are read
    together (a map and its profile).  *recording*: the name the title must contain (default: read from *path*)."""
    path = Path(path)
    for t in _texts(fig):
        t.set_fontfamily(FONT_FAMILY)
    try:
        fig.tight_layout()
    except Exception:                                   # a layout engine that refuses: keep the figure's own layout
        pass
    problems = check_figure(fig, path, allow_two_panels=allow_two_panels, recording=recording)
    if problems:
        if close:
            plt.close(fig)
        raise FigureRuleError(f"{path}: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    if close:
        plt.close(fig)
    return path
