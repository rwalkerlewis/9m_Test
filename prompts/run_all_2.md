Read `AGENTS.md`. Then carry out these five task prompts in `prompts/`, one after another, each on its own branch and its own pull request, in this order:

1. `prompts/housekeeping.md`
2. `prompts/joint_removal.md`
3. `prompts/false_tracks.md`
4. `prompts/compute_budget.md`
5. `prompts/noise_dependence.md`

For each task: `git fetch origin && git checkout -b <branch named in the prompt> origin/main`, carry the prompt out to its stop point, open the pull request against `main` as the prompt says, then merge it yourself (`gh pr merge --merge`) once `pytest tests`, `scripts/reproduce.py --check` (or `--update`, run and recorded, where the prompt allows a number to move) and `scripts/clean_outputs.py --check` pass. Start the next task from the updated `origin/main`. Because the tasks run in sequence, the file ownership clauses in the prompts do not apply between them; do in each task what its prompt says and leave to the later task what it names.

After every merge write one paragraph to `output/reports/run_all_2.md` (the task, the pass or fail of each of its criteria, the numbers that moved, what did not work) and commit it to `main`. If a task cannot be completed (a criterion that cannot be evaluated, a diagnosis in a prompt's first section that says to stop, a test that cannot pass without changing a number the prompt forbids changing, a GPU fault), write what happened in that paragraph, leave the branch and its pull request open unmerged, and continue with the next task from `origin/main`. Do not reorder the tasks and do not skip one because an earlier one failed. If the GPU drops into an error state again, record the time and the driver message in the paragraph, continue on the CPU where the prompt allows, and say which numbers were produced on the CPU.

At the end, report the five paragraphs together and list every pull request with its state.
