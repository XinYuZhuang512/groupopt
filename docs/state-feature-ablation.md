# State-feature ablation pilot

## Question

The pilot asks which explicit path-state signal is responsible for the behavior of
`adaptive_state`, and whether its gain comes from dynamic state information rather
than merely from the extra projection capacity.

## Modes

All feature modes use the same `project_tail_state` input width and parameter count.

| Mode | Active feature blocks |
| --- | --- |
| `adaptive_static` | candidate embedding twice + constant initial size |
| `adaptive_state_mean` | current component mean only |
| `adaptive_state_start` | current path-start embedding only |
| `adaptive_state_size` | current normalized component size only |
| `adaptive_state` | component mean + path start + component size |

The existing `adaptive` run remains the no-state/no-extra-projection reference;
`fixed` remains the conventional construction benchmark. The static control is the
parameter-matched comparison for testing whether changing path state itself matters.

## Pilot protocol

- Model families: AM and Pointer Network.
- Distribution: uniform Euclidean TSP50.
- Seed: 1234.
- Training: 2,000 steps, batch size 512, learning rate `1e-4`.
- Validation: fixed 1,024-instance set, greedy decoding every 100 steps.
- Checkpoints: every 250 steps; interrupted runs resume automatically.
- New runs: the four partial/static modes above. Existing `adaptive` and
  `adaptive_state` pilot runs are reused because their implementations are unchanged.

This single-seed pilot is a screening experiment, not confirmatory evidence. We will
compare the curves and final validation costs, choose only clearly informative modes,
then run multi-seed training and independent held-out evaluation before drawing a
claim.

## Decision rules

1. If full state beats the static control, dynamic path information contributes beyond
   parameter capacity.
2. If one isolated block approaches or exceeds full state consistently within a model,
   promote it to multi-seed evaluation.
3. If a feature helps AM but hurts Pointer Network, treat that as model-dependent use of
   the shared state interface rather than failure of cross-model applicability.
4. Do not select a mode from a single best checkpoint fluctuation alone; require a
   sustained curve advantage or confirmation on independent data.
