# Model-range expansion protocol

## Objective

Test whether the `joint_free` construction advantage transfers across genuinely
different scoring backbones while keeping the construction state, legal-action
mask, transition, training objective, and decoding strategy fixed.

## Stage A: available architecture families

| Backbone | Architectural role | Status |
| --- | --- | --- |
| Transformer-LN | global self-attention | implemented |
| GAT | additive graph attention | implemented |
| Pointerformer-style | reversible Transformer | implemented |
| GRU | recurrent sequence control | implemented |

For each backbone, compare `joint_fixed` and `joint_free` with identical model
parameters. Seeds 1234, 2345, 3456, and 4567 are the experimental units. A
2,000-step multi-seed screen checks trainability only; it is not final evidence.
Every promoted backbone must be trained to 10,000 steps and evaluated using the
exact final checkpoint on a test set generated only after the promotion rule is
locked.

## Stage B: new modern architecture families

Add two cleanly separated scorer backbones:

1. a geometry-aware equivariant/message-passing graph encoder;
2. a sparse mixture-of-experts Transformer encoder.

These test geometric inductive bias and conditional model capacity without
changing the construction or inference strategy.

## Methods kept separate

POMO changes multi-start inference, Sym-NCO changes the training objective, and
LEHD derives much of its identity from a heavy decoder. Replacing those decoders
with the shared joint scorer would not constitute a faithful comparison with the
original methods. They should therefore be reported later as compatibility or
training/solver-strategy extensions, not as pure backbone controls.

## Anti-leakage rules

- Development results may use the already exposed test seed `20260805`.
- The credibility-audit seed `20260809` is also considered exposed and must not
  serve as the final model-range test.
- A new final test seed is generated only after models, horizons, checkpoint rule,
  and promoted backbones are frozen.
- Primary inference uses exact step-10,000 checkpoints; best-validation results are
  secondary only.
