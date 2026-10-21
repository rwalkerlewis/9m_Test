#!/usr/bin/env python3
"""Check the layout of ``output/``, and list (optionally delete) its regenerable files.

``--check`` fails (exit 1) on any file under ``output/`` outside the layout of ``AGENTS.md`` (The output folder):

* ``output/recordings/<recording>/``: the files of one recording (scenario used, levels, map, components, propagation
  error, build information, logs, the observation and truth files), ``seed<n>/`` (test recordings mixed from the
  cached gathers) and ``cache/`` (FDTD gathers);
* ``output/reproduce/``: ``summary.json``, ``latest.json``, ``tables.md``, ``run.log`` and ``runs/<recording>/``;
* ``output/walkthrough/<recording>/``;
* ``output/studies/<study>/`` (anything below; ``history/<script>/`` for the history scripts);
* ``output/air/<scenario>/`` and ``output/air/comparison_results.json``;
* ``output/reports/*.md``;

and on any committed file that ``AGENTS.md`` says is never committed (``*.npz``, ``*.npy``, ``*.mp4``, caches).

Without ``--check`` it lists the files that git reports as *ignored and untracked* below ``output/`` (datasets, FDTD
caches, ``*.npy``/``*.npz``, videos) with the space per directory; nothing is removed without ``--delete``, tracked
deliverables are never candidates, nor is anything outside ``output/``.

    python scripts/clean_outputs.py --check                  # every file inside the layout?
    python scripts/clean_outputs.py                          # dry run over all of output/
    python scripts/clean_outputs.py recordings/bistatic/cache --delete

See docs/outputs.md for what each directory holds, what writes it and what reads it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def _git_files(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", *args],
        check=True, capture_output=True, text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def candidates(subdirs: list[str]) -> list[Path]:
    """Ignored, untracked regular files under ``output/<subdir>`` (default: all of output/)."""
    scopes = []
    for sub in subdirs or [""]:
        scope = (OUTPUT / sub).resolve()
        if scope != OUTPUT and OUTPUT not in scope.parents:
            sys.exit(f"refusing {sub!r}: not inside {OUTPUT}")
        scopes.append(scope.relative_to(ROOT).as_posix())
    tracked = set(_git_files("--", *scopes))
    files = []
    for rel in _git_files("--others", "--ignored", "--exclude-standard", "--", *scopes):
        path = ROOT / rel
        if rel in tracked:
            sys.exit(f"refusing {rel}: tracked by git")
        if path.is_symlink() or not path.is_file():
            print(f"skipping {rel}: not a regular file")
            continue
        if OUTPUT not in path.resolve().parents:
            sys.exit(f"refusing {rel}: resolves outside {OUTPUT}")
        files.append(path)
    return files


RECORDINGS = ("simple", "bistatic", "multistatic", "multistatic_b", "harbor")
NAME = r"[A-Za-z0-9_.+-]+"
LAYOUT = [re.compile(p) for p in (
    rf"recordings/({'|'.join(RECORDINGS)})/{NAME}",
    rf"recordings/({'|'.join(RECORDINGS)})/seed\d+/dataset\.(observation|truth)\.npz",
    rf"recordings/({'|'.join(RECORDINGS)})/cache/.+",
    r"reproduce/(summary\.json|latest\.json|tables\.md|run\.log)",
    rf"reproduce/runs/{NAME}/.+",
    rf"walkthrough/({'|'.join(RECORDINGS)})/.+",
    rf"studies/{NAME}/.+",
    rf"air/{NAME}/.+",
    r"air/comparison_results\.json",
    rf"reports/{NAME}\.md",
)]
NEVER_COMMITTED = re.compile(r"(\.npz|\.npy|\.mp4)$|(^|/)cache/")


def layout_problems() -> list[str]:
    """Every file under ``output/`` (on disk or tracked) outside the layout, and every tracked file that is never
    committed."""
    problems = []
    on_disk = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob("*") if p.is_file() or p.is_symlink()} \
        if OUTPUT.exists() else set()
    tracked = {f[len("output/"):] for f in _git_files("--", "output")}
    for rel in sorted(on_disk | tracked):
        if rel == ".gitkeep":
            continue
        if not any(p.fullmatch(rel) for p in LAYOUT):
            problems.append(f"outside the layout: output/{rel}")
        elif rel in tracked and NEVER_COMMITTED.search(rel):
            problems.append(f"committed but never to be committed: output/{rel}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subdirs", nargs="*", help="paths relative to output/ (default: all of output/)")
    ap.add_argument("--delete", action="store_true", help="actually delete the listed files")
    ap.add_argument("--check", action="store_true", help="fail on any file outside the layout of output/")
    args = ap.parse_args()
    if args.check:
        problems = layout_problems()
        for p in problems:
            print(p)
        print(f"{len(problems)} files outside the layout of output/ (AGENTS.md, The output folder; docs/outputs.md)")
        sys.exit(1 if problems else 0)

    files = candidates(args.subdirs)
    per_dir: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in files:
        key = path.parent.relative_to(ROOT).as_posix()
        per_dir[key][0] += 1
        per_dir[key][1] += path.stat().st_size
    for key in sorted(per_dir):
        n, size = per_dir[key]
        print(f"{_human(size):>10}  {n:6d} files  {key}/")
    total = sum(size for _, size in per_dir.values())
    print(f"{_human(total):>10}  {len(files):6d} files  total")

    if not args.delete:
        print("dry run: nothing deleted (pass --delete to remove these files)")
        return
    for path in files:
        path.unlink()
    for key in sorted(per_dir, key=len, reverse=True):
        directory = ROOT / key
        while directory != OUTPUT and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent
    print(f"deleted {len(files)} files")


if __name__ == "__main__":
    main()
