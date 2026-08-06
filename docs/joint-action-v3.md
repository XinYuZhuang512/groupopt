# Joint-action stabilizer decoder v3

## Design correction

The first adaptive decoder factorized one construction step into a tail policy and a
conditional head policy. This makes the stabilizer-chain schedule an explicit policy
decision before the model can inspect the quality of the complete edge. Gated v2 can
fall back to a fixed schedule, but it does not remove this factorization.

V3 treats the stabilizer object and its coset representative as one atomic action.
For the directed TSP instance, an action is the legal edge

```text
a_t = (x, y),  x in open tails, y in open heads.
```

Before the closing step, `x` and `y` must belong to different path components. The
transition adds `x -> y` and merges the two components. The last action closes the
remaining path into one Hamiltonian cycle.

## Joint policy

Each model supplies contextual node embeddings. A shared joint decoder augments the
tail and head embeddings with their current path-component state, computes a score
matrix `L[x, y]`, masks illegal pairs, and applies one softmax over all legal pairs:

```text
p(x, y | state) = softmax(mask(L))[x, y].
```

There is one sampled action and one log probability per construction step. The
model compares complete candidate edges before choosing a stabilizer-chain move.

Two modes use the same scorer and differ only by a framework mask:

- `joint_fixed`: restrict `x` to the tail of the anchored component;
- `joint_free`: allow every legal `(x, y)` pair.

This gives a parameter-matched comparison between a canonical stabilizer schedule
and free stabilizer-chain construction.

## Required properties

1. Every unmasked pair is a valid transition.
2. Every nonterminal state has at least one unmasked pair.
3. Both modes terminate after exactly `n` edge actions in one Hamiltonian cycle.
4. `joint_fixed` reproduces a continuous anchored path.
5. `joint_free` remains complete for all directed Hamiltonian cycles.

## Pilot rule

After unit, exhaustive small-instance, backward, and memory tests, train
`joint_fixed` and `joint_free` for 2,000 steps with seed 1234 in AM, Pointer Network,
and GPN. Promote v3 only if free-base behavior is trainable in all three models and
its quality/runtime tradeoff is more consistent than the legacy two-stage decoder.

## Seed-1234 pilot result

| Model | Fixed validation | Free validation | Fixed test | Free test | Test improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| AM | 6.3023 | **6.2502** | 6.3069 | **6.2697** | 0.59% |
| Pointer Network | 6.4147 | **6.4007** | 6.4230 | **6.4073** | 0.25% |
| GPN | 6.4022 | **6.2829** | 6.4142 | **6.3048** | 1.71% |

Each best checkpoint was evaluated greedily on the same 10,000 independent TSP50
instances. The paired per-instance 95% confidence intervals for `free - fixed` are:

- AM: `[-0.0441, -0.0303]`;
- Pointer Network: `[-0.0225, -0.0090]`;
- GPN: `[-0.1165, -0.1025]`.

Free stabilizer-chain construction improves both validation and independent test
cost in all three architectures. This is the first directionally consistent
cross-model result in the project, but it still uses one training seed and a short
2,000-step horizon. V3 therefore passes the pilot screen and should move to a small
multi-seed confirmation before any broader model or mechanism expansion.

## Four-seed confirmation

Train both modes from scratch with seeds 2345, 3456, and 4567 under the same
2,000-step protocol, then evaluate every best checkpoint on the frozen independent
test set.

| Model | Seed | Fixed test | Free test | Free - fixed |
| --- | ---: | ---: | ---: | ---: |
| AM | 1234 | 6.3069 | 6.2697 | -0.0372 |
| AM | 2345 | 6.3181 | 6.2666 | -0.0514 |
| AM | 3456 | 6.3155 | 6.2306 | -0.0849 |
| AM | 4567 | 6.3269 | 6.2603 | -0.0666 |
| Pointer Network | 1234 | 6.4230 | 6.4073 | -0.0158 |
| Pointer Network | 2345 | 6.4149 | 6.3990 | -0.0158 |
| Pointer Network | 3456 | 6.3993 | 6.3426 | -0.0566 |
| Pointer Network | 4567 | 6.4309 | 6.4008 | -0.0300 |
| GPN | 1234 | 6.4142 | 6.3048 | -0.1095 |
| GPN | 2345 | 6.4168 | 6.2464 | -0.1704 |
| GPN | 3456 | 6.4092 | 6.2497 | -0.1595 |
| GPN | 4567 | 6.4259 | 6.3275 | -0.0984 |

| Model | Mean fixed | Mean free | Relative improvement | Seed-level 95% CI | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| AM | 6.3168 | 6.2568 | 0.95% | `[-0.0926, -0.0275]` | 4/4 |
| Pointer Network | 6.4170 | 6.3874 | 0.46% | `[-0.0602, 0.0011]` | 4/4 |
| GPN | 6.4165 | 6.2821 | 2.10% | `[-0.1914, -0.0775]` | 4/4 |

Joint free wins all 12 model-seed comparisons. AM and GPN exclude zero at the
training-seed level; Pointer Network has a narrow interval crossing zero with only
four seeds, but all observed differences have the same sign. This confirms a stable
short-horizon cross-model signal and justifies a staged long-horizon test before
claiming final-quality superiority.
