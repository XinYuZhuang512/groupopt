# SYM-NCO × GroupOpt factorial pilot

## Question

This pilot separates two questions that a direct “Ours vs SYM-NCO” comparison
would confound:

1. Does the native conditional tail-selection framework improve the original AM?
2. Does symmetry-aware training improve either decoder, and are the two ideas
   complementary?

The four cells are:

| Variant | Decoder construction | Training objective |
|---|---|---|
| Original | `native_conditional_fixed` | REINFORCE |
| Original + SYM-NCO | `native_conditional_fixed` | SYM-NCO AM objective |
| Ours | `native_conditional_free` | REINFORCE |
| Ours + SYM-NCO | `native_conditional_free` | SYM-NCO AM objective |

`native_conditional_fixed` and `native_conditional_free` instantiate the same AM
module and therefore contain the same parameters. The fixed control does not use
the learnable tail decision; the free condition activates it. This controls the
stored solver parameter count while testing the effect of the framework. SYM-NCO
adds its official training-only projection head to both SYM-NCO cells; it is saved
separately, discarded for evaluation, and therefore adds no inference capacity.

## What is and is not changed

SYM-NCO is applied only as a training objective. It does not replace the AM
encoder, the native AM head decoder, GroupOpt's tail selector, feasibility masks,
or state transitions.

The implementation follows the official AM variant:

- the first copy is the original Euclidean instance;
- additional copies are arbitrary rotations/reflections preserving every edge
  length and node identity;
- the policy-gradient baseline is the mean cost across augmentations of the same
  underlying instance;
- encoder node representations pass through the official two-layer projection
  head, then transformed copies are encouraged to have high cosine similarity to
  the original copy;
- `alpha=0.1`, matching the official default.

This is the AM-compatible problem-symmetry component. It should not be described
as reproducing POMO-style solution-symmetry/multi-start training.

References: [SYM-NCO paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2279f2bf652b4fc9a748e8b3e8dd6bbd-Abstract-Conference.html),
[official implementation](https://github.com/alstn12088/Sym-NCO).

## Pilot protocol

- Problem: TSP50.
- Training distribution: uniform i.i.d. coordinates only.
- Training seeds: 1234 and 4321.
- Training horizon: 2,000 optimizer steps.
- Effective sampled trajectories per step: 512 in every cell.
- Standard cells: 512 independent instances × 1 rollout.
- SYM-NCO cells: 128 independent instances × 4 symmetry copies.
- Model selection: none for the reported result; evaluation uses the final
  step-2,000 checkpoint, not `best.pt`.
- Independent tests: 5,000 fixed instances for each of `uniform`, `clustered`,
  `narrow_strip`, and `clustered_outliers`.
- Inference budget: one greedy decode with no test-time symmetry augmentation in
  every cell. This isolates training effects; a full augmented-inference benchmark
  can be added later as a separately costed experiment.
- Every cell sees exactly the same saved test instances through the shared test
  seed, enabling paired differences.

The equal-trajectory protocol controls the dominant decoder compute but does not
make the number of unique underlying training instances equal: symmetry training
intentionally spends four trajectories on each base instance. Runtime and peak GPU
memory are logged and must be reported alongside quality.

## Interpretation

Tour length is minimized. Therefore:

- `Original cost - Ours cost > 0` means Ours improves over Original;
- `REINFORCE cost - SYM-NCO cost > 0` means SYM-NCO helps that decoder;
- the factorial interaction
  `(Original_SYM - Ours_SYM) - (Original_standard - Ours_standard) > 0`
  means SYM-NCO increases the relative advantage of Ours.

The same-instance 95% intervals condition on a trained checkpoint. Two training
seeds show whether the direction repeats, but are not enough for a reliable
population-level confidence interval over training randomness.

## Run

On the GPU server:

```bash
cd /root/autodl-tmp/groupopt
screen -dmS symnco_factorial bash experiments/pipelines/current/run_symnco_factorial_pilot_server.sh
```

Progress and completion:

```bash
tail -n 40 /root/autodl-tmp/groupopt-data/symnco_factorial.log
cat /root/autodl-tmp/groupopt-data/symnco_factorial.exit
```

The final CSV and JSON files are written under
`artifacts/symnco_factorial_pilot/summary/`.
