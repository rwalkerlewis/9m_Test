# Codebase cleanup and consolidation

Review and improve the repository's maintainability without changing its intended
simulation, detection, or public API behavior. Work in the current checkout and
preserve any pre-existing user changes. Do not commit or push unless explicitly
asked. Start by checking `git status` and reading the relevant code, tests,
documentation, and CI configuration; do not assume paths or branch state.

## Goals

Address the following review findings in priority order. Keep changes focused,
tested, and independently reviewable. If an item is too risky or broad to finish
safely, document the evidence and recommend a bounded follow-up rather than
performing a speculative rewrite.

### 1. Repair the known anomaly test failure

Investigate `tests/test_anomaly.py::test_reconstruction_high_for_ood` and
`test_reconstruction_low_for_in_distribution`. The OOD test currently expects
five values from the in-distribution helper, which returns four.

- Establish the intended helper/test contract before changing either function.
- Fix the root cause with the smallest clear change.
- Add or adjust assertions so the in-distribution and out-of-distribution
  behaviors are still checked, not merely the tuple shape.
- Run the focused anomaly tests when the optional PyTorch dependency is available.
- Do not change test thresholds simply to make the test pass.

### 2. Strengthen basic GitHub Actions coverage

Inspect `.github/workflows/` and `pyproject.toml`. Keep the basic check fast,
CPU-only, deterministic, and suitable for standard GitHub-hosted runners.

- Add a small, representative generic FDTD solver test (for example, a focused
  test from `tests/test_fdtd_3d.py`) to the existing basic CI command. Avoid
  benchmark-style timing assertions or GPU-only tests.
- Confirm optional GPU and ML dependencies are not required by this job.
- Run the exact CI test command locally and report the test count and duration.
- Decide whether the full suite is suitable as a separate optional or scheduled
  job only after identifying slow, optional, and currently failing tests. Do not
  silently make a known-failing suite a required check.

### 3. Bring package-layout documentation and prompts up to date

The canonical application packages are `src/acoustic_sim/air/` and
`src/acoustic_sim/water/`; shared FDTD code is under
`src/acoustic_sim/solver/`. Search README files, `docs/`, `prompts/`, scripts,
and examples for references to the former `src/acoustic_sim/harbor/` or
root-level FDTD paths.

- Update active documentation and task prompts to the canonical paths.
- Clearly label genuinely historical plans as historical rather than making
  them look like current instructions.
- Keep the explanation of supported legacy imports accurate, and do not remove
  compatibility code as part of a documentation-only cleanup.
- Search again after edits to verify stale active references are gone.

### 4. Reduce root-package facade complexity safely

Review `src/acoustic_sim/__init__.py` and the compatibility packages/modules.
The root facade exports many domain-specific symbols and registers legacy import
aliases.

- Prefer `acoustic_sim.air`, `acoustic_sim.water`, and `acoustic_sim.solver` in
  new code and current documentation.
- Preserve established root-level imports and legacy module paths unless a
  separately approved deprecation policy says otherwise.
- Do not delete aliases or change import behavior in this cleanup. If simplifying
  the facade, use tests for public exports, old module aliases, import identity,
  and `python -m acoustic_sim.air.detection_main --help` to catch cycles and
  `runpy` warnings.
- Make incremental changes only where the result is clearly simpler without
  expanding or duplicating compatibility logic.

### 5. Decompose oversized modules incrementally

Inspect the responsibilities and import relationships in the largest modules,
including `air/plotting.py`, `solver/fdtd.py`, `water/bistatic.py`,
`air/detection_main.py`, `air/processor.py`, `air/fire_control.py`,
`sources.py`, and `air/tracker.py`.

- Identify cohesive responsibilities and propose a small extraction order.
- Extract only a well-bounded unit whose public API and dependencies can be
  preserved; do not split files merely to meet a line-count target.
- Keep existing import paths working and add focused regression tests for any
  extracted public or compatibility surface.
- Avoid mixing broad refactors with bug fixes or CI changes.

### 6. Establish one source of truth for dependencies

Compare `requirements.txt` and `pyproject.toml`.

- Make `pyproject.toml` the authoritative declaration for runtime and optional
  dependencies.
- Check repository and Docker/dev-container installation flows before changing
  `requirements.txt`; preserve workflows that depend on it, or reduce it to a
  clearly documented compatibility/dev requirements file.
- Verify documented installation commands remain valid. Do not add optional
  heavy dependencies to the base install.

### 7. Document generated-output cleanup and regeneration

Inspect `.gitignore`, tracked files under `output/`, scripts, and existing
documentation. The local output tree may contain large generated datasets and
caches, while a smaller selection of reports and figures is intentionally
tracked.

- Document which outputs are generated, which are tracked deliverables, and how
  users can regenerate key artifacts.
- If adding a cleanup command or script, make it explicit, dry-run or narrowly
  scoped by default, and refuse to remove tracked files or paths outside known
  generated-output directories.
- Never delete the user's existing output or cache data during this task.
- Do not untrack committed research results or figures without explicit
  approval.

## General constraints

- Inspect before editing and follow existing naming, formatting, and packaging
  conventions.
- Preserve numerical behavior and public signatures unless a demonstrated bug
  requires a change; explain any unavoidable behavior change.
- Avoid broad exception handling, silent fallbacks, and speculative defaults.
- Keep optional CUDA, MPI, and PyTorch support optional.
- Run `git diff --check`, targeted tests for each code change, and the exact
  basic CI test command. Run wider tests when practical and report unrelated
  baseline failures separately.
- Do not edit generated benchmark results to make tests or documentation appear
  better.

## Final report

Summarize changes by priority, list validation commands and results, note any
deferred items with reasons, and identify any remaining stale paths or known
failures. Do not claim the full suite passes unless it was actually run and
passed.
