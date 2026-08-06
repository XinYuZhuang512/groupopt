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
