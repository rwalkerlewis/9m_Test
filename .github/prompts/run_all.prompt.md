---
mode: agent
description: Carry out the six task prompts in prompts/ in order, one branch and PR each
---
Read `AGENTS.md`. Then carry out the six task prompts in `prompts/`, one after another, each on its own branch and its own pull request, in this order:

1. `prompts/figures_and_output.md`
2. `prompts/direct_path.md`
3. `prompts/tracker_uncertainty.md`
4. `prompts/ci_and_noise_model.md`
5. `prompts/sensor_layout_study.md`
6. `prompts/float32.md`

For each task: `git fetch origin && git checkout -b <branch named in the prompt> origin/main`, carry the prompt out to its stop point, open the pull request against `main` as the prompt says, then merge it yourself (`gh pr merge --merge`) once `pytest tests` passes and `scripts/reproduce.py --check` passes (or `--update` has been run and recorded, where the prompt allows a number to move). Start the next task from the updated `origin/main`. Because the tasks run in sequence, the file ownership and "another session" clauses in the prompts do not apply; everything each prompt says to leave to another session, do in that task or in the task that owns it later in the order.

Two things change because one session does all six:

- `direct_path` and `tracker_uncertainty` both run `scripts/reproduce.py --update`; run it at the end of each, on the recordings as they are then, and in `docs/harbor.md` keep the before and after of each task separate.
- `float32` rebuilds the recordings last; `sensor_layout_study` builds its own small datasets before that, with the scheme and precision the scenario files have at that time, and says so in its docs.

After every merge write one paragraph to `output/reports/run_all.md` (the task, the pass or fail of each of its criteria, the numbers that moved, what did not work) and commit it to `main`. If a task cannot be completed (a criterion that cannot be evaluated, a test that cannot be made to pass without changing a number the prompt forbids changing, a build that fails on the hardware), write what happened in that paragraph, leave the branch and its pull request open unmerged, and continue with the next task from `origin/main`. Do not reorder the tasks and do not skip one because an earlier one failed.

At the end, report the six paragraphs together, and list every pull request with its state.
