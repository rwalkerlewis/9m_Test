Branch `joint_removal` from `origin/main`. You own `src/acoustic_sim/water/bistatic.py` (the ECA and reference code), `StreamingEcho` in `streaming.py`, the `echo` and `bistatic` blocks of `examples/configs/processing/*.json`, `examples/harbor/05_detection/harbor_cancellation_depth.py`, `output/studies/direct_path/`, `output/reproduce/` and the direct path sections of `docs/harbor.md`. No FDTD runs (the stage 0a gathers exist on the k space scheme; a two ship case in section 1 needs two 10 s runs, say so before starting them).

Read `docs/harbor.md` (Direct path removal, Numbers), `output/studies/direct_path/results.json` and `recording_depth.json`, `src/acoustic_sim/water/bistatic.py` (`estimate_reference`, `eca_stage`, `eca_remove`), `StreamingEcho` in `streaming.py`, and `output/recordings/*/levels.json` before changing anything.

## The finding this task acts on

The direct path study measured that a lone ship 99 m or more from the cable is removed by 22.6 to 26.5 dB with the plain beam on the corrected recordings, while on the multi ship recordings every ship is removed by only 6 to 15 dB, with the beam and with the `joint` reference alike. The reference is not the limit there. The ships are removed one at a time, each with a reference beamed toward it while the other ships' direct sound is still in the traces: the reference of one ship carries the others, and the residual of each removal is left for the next. The diver and the six pilings, whose echoes are the weakest, are tracked in 0 % of frames on every multistatic recording; that is the symptom.

## 1. Measure the mechanism before fixing it

On the stage 0a framework add a two ship case: the ferry signature at 193 m and a second ship (the tug's signature) at 99 m on a crossing track, both moving, 10 s. Measure the removal depth of each ship (a) alone in the traces, (b) with the other present and removed one at a time in the current order, (c) one at a time in the other order. Criterion for the diagnosis, fixed now: if (b) and (c) are each more than 6 dB below (a) for at least one ship, the sequential removal is the mechanism and section 2 proceeds; if not, report what else the recordings have that the lone ship case does not (quay reflections, moving bodies) and stop after writing that up.

## 2. All illuminators in one solve

Replace the sequential ECA by one least squares solve per block over every confirmed illuminator at once: the direct path of every channel is the sum, over illuminators, of delayed, scaled copies of each illuminator's reference (delays from each track, as now), and the references themselves are estimated with every other illuminator's current estimate subtracted first (alternate: beam each reference on the traces minus the others' modelled direct paths, re solve, two or three passes; or the single joint solve of all waveforms and all channel scales where the block size allows it). Keep the per sample delays, the quay image option and the prefix sum Gram where they still apply; state the cost. Criteria, fixed now, on the two ship case of section 1: each ship within 3 dB of its lone depth; on the multistatic tuning recording: the median removal depth of every ship (truth components, a diagnostic) at least 18 dB.

## 3. The chain

`scripts/reproduce.py --update` on every recording. Criteria, fixed now, on the test recordings: at least three of the six pilings and the diver tracked with Pd at least 0.3 on the multistatic seeds; the moored hull and both UUVs not down by more than 3 points; compute per second of data not above its current value on any recording (this task is not the one that brings it down, but it does not raise it). The size criterion of stage 2 (the hull below a piling on the size indicator in at least 70 % of frames where both are measured) is evaluated as soon as a piling is tracked; report it.

## Rules

The repository rules in `AGENTS.md` apply, the figure rules included. Tests: the two ship analytic case (each reference recovered within a stated coherence with both ships present; each depth within 1 dB of the lone depth), the one pass against the alternating estimate on a noise free case, and the chain end to end. Update the walkthrough's step 5 text and figures. Stop after section 3. Report the depths of section 1 and 2, the pass or fail of each criterion, and which silent bodies are tracked now.
