#!/usr/bin/env python3
"""Run all 18 FDTD example combinations.

Launches ``mpirun -np <nprocs> python examples/run_fdtd.py ...`` for every
combination of {static, moving} × {isotropic, wind, hills_vegetation} ×
{concentric, circular, linear}.

Usage::

    python examples/run_all_examples.py            # default 4 MPI ranks
    python examples/run_all_examples.py --np 2     # use 2 MPI ranks
    python examples/run_all_examples.py --np 1     # single-process mode
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DOMAINS = ["isotropic", "wind", "hills_vegetation"]
SOURCE_TYPES = ["static", "moving"]
ARRAYS = ["concentric", "circular", "linear"]

# Static sources use the WAV file; moving sources use the propeller model.
SOURCE_SIGNAL = {
    "static": ["--source-signal", "file", "--source-wav", "audio/input.wav", "--max-seconds", "0.3"],
    "moving": ["--source-signal", "propeller"],
}

# Domain-specific extra args.
DOMAIN_ARGS: dict[str, list[str]] = {
    "isotropic": [],
    "wind": ["--wind-speed", "15", "--wind-dir", "45"],
    "hills_vegetation": [],
}

# Common simulation parameters (short runs for demonstration).
COMMON = [
    "--total-time", "0.3",
    "--dx", "0.5",
    "--snapshot-interval", "50",
    "--source-freq", "25",
    "--receiver-count", "16",
]


def build_run_name(domain: str, source_type: str, array: str) -> str:
    return f"{domain}_{source_type}_{array}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=4, help="Number of MPI ranks")
    parser.add_argument("--output-root", type=str, default="output", help="Top-level output dir")
    args = parser.parse_args()

    script = str(Path(__file__).resolve().parent / "run_fdtd.py")
    results: list[tuple[str, int]] = []

    for domain in DOMAINS:
        for source_type in SOURCE_TYPES:
            for array in ARRAYS:
                name = build_run_name(domain, source_type, array)
                out_dir = str(Path(args.output_root) / name)

                cmd = [
                    "mpirun", "--allow-run-as-root",
                    "-np", str(args.np),
                    sys.executable, script,
                    "--domain", domain,
                    "--source-type", source_type,
                    "--array", array,
                    "--output-dir", out_dir,
                    *COMMON,
                    *DOMAIN_ARGS[domain],
                    *SOURCE_SIGNAL[source_type],
                ]

                # Moving source needs trajectory endpoints.
                if source_type == "moving":
                    cmd += [
                        "--source-x", "-30",
                        "--source-y", "0",
                        "--source-x1", "30",
                        "--source-y1", "0",
                        "--source-speed", "50",
                    ]

                print(f"\n{'='*70}")
                print(f"  [{len(results)+1}/18]  {name}")
                print(f"  CMD: {' '.join(cmd)}")
                print(f"{'='*70}\n")

                rc = subprocess.call(cmd)
                results.append((name, rc))

    # Summary.
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, rc in results:
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  {name:45s} {status}")
    n_ok = sum(1 for _, rc in results if rc == 0)
    print(f"\n  {n_ok}/{len(results)} completed successfully.")


if __name__ == "__main__":
    main()
