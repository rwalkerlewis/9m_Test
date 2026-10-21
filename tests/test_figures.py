"""The figure rule (``AGENTS.md``, Figures) and the output layout.

A static scan of ``src``, ``examples`` and ``scripts``: no grid of more than two panels, no ``savefig`` or ``imsave``
outside ``acoustic_sim/plotting.py`` (so every figure is written by ``save_figure``); unit tests of every refusal of
``save_figure``; and ``scripts/clean_outputs.py --check`` (every file under ``output/`` inside the layout).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from acoustic_sim.plotting import FigureRuleError, check_figure, new_figure, recording_from_path, save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TREES = ("src", "examples", "scripts")
RULE_MODULE = ROOT / "src" / "acoustic_sim" / "plotting.py"
GRID_CALLS = {"subplot", "add_subplot", "subplot_mosaic", "add_gridspec", "GridSpec", "subplot2grid", "subfigures"}


def _sources():
    for tree in TREES:
        for p in sorted((ROOT / tree).rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p


def _const(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) else None


def _panels(call: ast.Call):
    """Panels of a ``subplots(...)`` call (nrows x ncols), None when they are not literal integers."""
    nrows, ncols = 1, 1
    if len(call.args) >= 1:
        nrows = _const(call.args[0])
    if len(call.args) >= 2:
        ncols = _const(call.args[1])
    for kw in call.keywords:
        if kw.arg == "nrows":
            nrows = _const(kw.value)
        elif kw.arg == "ncols":
            ncols = _const(kw.value)
    if nrows is None or ncols is None:
        return None
    return nrows * ncols


def _calls(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name:
                yield name, node


def test_no_grid_of_more_than_two_panels():
    bad = []
    for p in _sources():
        for name, call in _calls(p):
            where = f"{p.relative_to(ROOT)}:{call.lineno}"
            if name == "subplots":
                n = _panels(call)
                if n is None:
                    bad.append(f"{where}: subplots with a panel count that is not a literal")
                elif n > 2:
                    bad.append(f"{where}: subplots with {n} panels")
            elif name in GRID_CALLS:
                bad.append(f"{where}: {name}")
    assert not bad, "figures with grids of panels:\n" + "\n".join(bad)


def test_every_figure_is_saved_by_save_figure():
    bad = []
    for p in _sources():
        if p == RULE_MODULE:
            continue
        for name, call in _calls(p):
            if name in ("savefig", "imsave"):
                bad.append(f"{p.relative_to(ROOT)}:{call.lineno}: {name}")
    assert not bad, "figures saved without save_figure:\n" + "\n".join(bad)


def test_output_layout():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "clean_outputs.py"), "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------
# save_figure: every refusal
# ----------------------------------------------------------------------

def _good(title="simple: passive map at 3 s"):
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1], label="a")
    ax.plot([0, 1], [1, 0], label="b")
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    return fig, ax


def test_good_figure_is_written(tmp_path):
    fig, _ = _good()
    p = save_figure(fig, tmp_path / "recordings" / "simple" / "map.png")
    assert p.exists()
    img = plt.imread(p)
    assert img.shape[1] >= 1600


def test_refuses_more_than_one_panel(tmp_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    for ax in axs:
        ax.set_xlabel("t [s]")
        ax.set_ylabel("p [Pa]")
    axs[0].set_title("simple: trace")
    with pytest.raises(FigureRuleError, match="2 panels"):
        save_figure(fig, tmp_path / "a.png")
    assert not (tmp_path / "a.png").exists()


def test_two_panels_when_they_share_an_axis(tmp_path):
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax in axs:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axs[0].set_title("simple: map and profile")
    save_figure(fig, tmp_path / "a.png", allow_two_panels=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    for ax in axs:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axs[0].set_title("simple")
    with pytest.raises(FigureRuleError, match="3 panels"):
        save_figure(fig, tmp_path / "b.png", allow_two_panels=True)


def test_refuses_narrow_figure(tmp_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("simple")
    with pytest.raises(FigureRuleError, match="pixels wide"):
        save_figure(fig, tmp_path / "a.png")


def test_refuses_small_text(tmp_path):
    fig, ax = _good()
    ax.annotate("r3", (0.5, 0.5), fontsize=8)
    with pytest.raises(FigureRuleError, match="8.0 pt"):
        save_figure(fig, tmp_path / "a.png")
    fig, ax = _good()
    ax.legend(fontsize=7)
    with pytest.raises(FigureRuleError, match="7.0 pt"):
        save_figure(fig, tmp_path / "b.png")
    fig, ax = _good()
    ax.tick_params(labelsize=9)
    with pytest.raises(FigureRuleError, match="9.0 pt"):
        save_figure(fig, tmp_path / "c.png")


def test_refuses_missing_axis_labels(tmp_path):
    fig, ax = _good()
    ax.set_xlabel("")
    with pytest.raises(FigureRuleError, match="no x axis label"):
        save_figure(fig, tmp_path / "a.png")
    fig, ax = _good()
    ax.set_ylabel("")
    with pytest.raises(FigureRuleError, match="no y axis label"):
        save_figure(fig, tmp_path / "b.png")


def test_refuses_colour_bar_without_label(tmp_path):
    fig, ax = _good()
    im = ax.imshow([[0, 1], [2, 3]])
    cb = fig.colorbar(im, ax=ax)
    assert any("colour bar" in p for p in check_figure(fig, None))
    cb.set_label("level [dB]")
    assert check_figure(fig, None) == []
    with pytest.raises(FigureRuleError, match="colour bar"):
        fig2, ax2 = _good()
        fig2.colorbar(ax2.imshow([[0, 1], [2, 3]]), ax=ax2)
        save_figure(fig2, tmp_path / "a.png")


def test_refuses_title_without_recording(tmp_path):
    fig, _ = _good("passive map at 3 s")
    with pytest.raises(FigureRuleError, match="does not name the recording 'simple'"):
        save_figure(fig, tmp_path / "walkthrough" / "simple" / "a.png")
    fig, _ = _good("passive map at 3 s")
    with pytest.raises(FigureRuleError, match="'bistatic'"):
        save_figure(fig, tmp_path / "a.png", recording="bistatic")
    fig, ax = _good("")
    with pytest.raises(FigureRuleError, match="no title"):
        save_figure(fig, tmp_path / "b.png")


def test_recording_from_path():
    assert recording_from_path("output/recordings/simple/map.png") == "simple"
    assert recording_from_path("output/reproduce/runs/multistatic_b/picture_frame.png") == "multistatic_b"
    assert recording_from_path("output/walkthrough/harbor/01_recording_chart.png") == "harbor"
    assert recording_from_path("output/air/valley_test/summary.png") == "valley_test"
    assert recording_from_path("output/studies/x/a.png") is None
