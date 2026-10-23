# Working in this repository

These rules apply to every agent session (Claude Code reads this file through `CLAUDE.md`; GitHub Copilot reads it directly and through `.github/copilot-instructions.md`). Task prompts in `prompts/` and `.github/prompts/` do not repeat them.

## What the code is

A forward side that generates synthetic acoustic data by solving the wave equation (air and water), and a processing side that detects, locates, tracks and characterises objects from recorded traces, real or synthetic, and cannot tell the difference. `docs/architecture.md` (Sensing contract) defines the split; `tests/test_contract.py` enforces it. The water system is `acoustic_sim.water.stream` over `water/streaming.py`; `docs/harbor.md` is its description and its current numbers.

## Branches and pull requests

- `main` is the only line of development. Every task is a new branch from `origin/main` named in its prompt; ignore every other branch in the repository.
- Several sessions may run at once. Each prompt names the files it owns and the files other sessions own; do not edit files you do not own. If `main` moves while you work, rebase before opening the pull request.
- Commit in small steps. Open one pull request against `main` with the tables the prompt asks for in its description. Stop where the prompt says to stop and report what it asks for.
- A pull request is merged only when `pytest tests`, `scripts/reproduce.py --check` (or `--update`, recorded) and `scripts/clean_outputs.py --check` pass **and** every criterion the task fixed before running passes, or the task changes no number (a structural task). A task whose criteria fail leaves its pull request open with its results in the description, and `main` keeps the settings it had; the next task starts from `main` as it is.

## Measurement rules

- Every experiment has its pass or fail criterion written into the script docstring and `results.json` before it runs. Criteria are never changed after a result is seen.
- Parameters are chosen on the tuning recordings (noise seed 0 of each dataset, and noise seeds 4 to 9 of `simple`, `bistatic`, `multistatic` and `multistatic_b`, mixed from the cached gathers with both noise models: `<family>_tune<n>` in `scripts/reproduce.py`) and never changed after a test recording (seeds 1 to 3, `multistatic_b` with its noise seed 0) has been seen. Set `held_out` correctly in every provenance block.
- Every result says what it was given beyond the observation (`Provenance`: `truth_inputs`, `simulator_knowledge`). A table with a row given true information carries the row with everything estimated from the traces beside it.
- `scripts/reproduce.py` regenerates every number quoted in `docs/harbor.md`, in tables and in prose: a number in prose is written through a `<!-- reproduce:<key> -->` marker that the script fills, never typed by hand. A change that moves a number runs `--update` and says so in the pull request. Prose that interprets numbers is reread after every `--update` and corrected where the numbers no longer support it.
- `pytest tests` passes before a pull request is opened. New functions get tests on inputs with known answers.

## Figures

- One figure, one panel. Every figure is a single axes with one subject, saved on its own at 1600 pixels wide or more (`figsize` at least (10, 7) at 150 dpi), with axis labels in units, a title that names the recording and the quantity, and a legend where there is more than one series. No grids of subplots: a comparison across configurations or objects is one figure per configuration or object, or one figure with the series overlaid when they share axes. The one exception is two panels that share an axis and are read together (a map and its profile), and even then each panel keeps the sizes above.
- Text is at least 11 pt at the saved size; labels never overlap (offset or thin them); colour bars have units; maps carry a scale bar or axes in metres and north up.
- Every figure goes through `acoustic_sim.plotting.save_figure(fig, path)`, which enforces these rules (axes count, size, font size, labels present) and refuses to save otherwise. `tests/test_figures.py` greps the code for `subplots(` with more than two panels and fails on any.
- A page in `docs/` that shows figures shows the split ones, each with one sentence saying what to look at.

## The output folder

- `output/` holds only what the current docs reference or `scripts/reproduce.py` regenerates, in this layout: `output/recordings/<recording>/` (build, levels, map, propagation error, scenario used), `output/reproduce/` (summary, tables, `runs/<recording>/`), `output/walkthrough/<recording>/`, `output/studies/<study>/` (one folder per study with its `results.json`, figures and `run.log`), `output/air/<scenario>/`.
- Figures, JSON, JSONL, CSV and logs are committed; `*.npz`, `*.npy`, `*.mp4` and FDTD caches are not.
- A study keeps only the figures its docs page shows plus its `results.json` and `run.log` (and the tables and summaries the docs read); the walkthrough and `reproduce/runs/` likewise keep only the figures the docs show. Every other figure, and a study's per run `picture.jsonl` and `tracks.csv`, is regenerated by its script whenever needed; the figures that are committed are listed in `.gitignore`. What is committed under `output/` stays under 60 MB and 300 PNG files; `scripts/clean_outputs.py --check` fails on either limit and lists the largest folders.
- A script that stops being part of the system or a study moves its outputs out of `output/` in the same pull request; git history keeps them, and `docs/harbor_history.md` names the commit.

## Language

- Say what was done in plain words: "given the true waveform", "given the exact track", "everything estimated from the traces". Never invent a label for a basic concept and never use "oracle", "blind", "ground truth magic" or similar in code, docs, figures or commit messages.
- No dashes in prose (no em dashes, en dashes or hyphenated compounds); reword. Identifiers and command line flags are exempt.
- State plainly what did not work. A number that fails its criterion is reported as a failure, not softened.

## Compute

- The FDTD and the chains run on the GPU when there is one (`--cuda` or the config), with CPU fallback. One GPU is shared between sessions: a task that builds datasets says so in its prompt, and FDTD runs from two sessions are not started at the same time.
