#!/usr/bin/env python3
"""List, and optionally delete, regenerable files under ``output/``.

Only files that git reports as *ignored and untracked* below ``output/`` are candidates (datasets,
FDTD caches, ``*.npy``/``*.npz``, untracked figures and checkpoints).  Tracked deliverables are never
touched, nor is anything outside ``output/``.  The default is a dry run that prints the space per
directory; nothing is removed without ``--delete``.

    python scripts/clean_outputs.py                          # dry run over all of output/
    python scripts/clean_outputs.py harbor_bistatic/cache    # dry run, one subtree
    python scripts/clean_outputs.py harbor_bistatic/cache --delete

See docs/outputs.md for what each directory holds and how to regenerate it.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subdirs", nargs="*", help="paths relative to output/ (default: all of output/)")
    ap.add_argument("--delete", action="store_true", help="actually delete the listed files")
    args = ap.parse_args()

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
