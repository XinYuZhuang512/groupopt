# Conservative gated adaptive construction

## Motivation

The original state-aware tail selector is operational in AM, Pointer Network, and
GPN, but its final-quality gain is model dependent. It strongly improves AM, loses
to fixed construction in Pointer Network, and is statistically tied with fixed in
GPN across four training seeds. A likely source of instability is that adaptive
construction forces every model to make an additional stochastic tail decision at
every construction step.

## Policy

`gated_adaptive_state` retains the existing state-aware adaptive distribution
`p_adaptive` but mixes it with the deterministic anchored tail `t_fixed`:

```text
p(tail | state) = (1 - g(state)) delta(t_fixed) + g(state) p_adaptive(tail | state)
```

The shared gate consumes the mean full path-state representation over currently
legal tails. Its projection is zero-initialized with a sigmoid bias corresponding to
`g = 0.1`. The initial policy therefore assigns at least 90% probability to the
fixed tail while retaining differentiable exploration of adaptive alternatives.

The mixed categorical log probability is included in the same REINFORCE objective
as the head action. Greedy inference uses the mixed distribution directly. The
model output and training log expose mean tail entropy and mean gate probability for
diagnosis.

## First screen

Reuse the existing seed-1234, 2,000-step `fixed` and `adaptive_state` pilots. Train
only `gated_adaptive_state` for AM, Pointer Network, and GPN with their existing
TSP50 pilot configurations. Promote the mechanism only if it avoids a meaningful
loss to fixed in all three models and improves at least two models.

## Seed-1234 pilot result

| Model | Fixed best | State best | Gated best | Final gate |
| --- | ---: | ---: | ---: | ---: |
| AM | 6.7826 | **6.4513** | 6.7715 | 0.0605 |
| Pointer Network | 7.3011 | **6.6581** | 6.6943 | 0.9990 |
| GPN | **6.3601** | 6.4031 | 6.3609 | 0.0003 |

All values are truncated to the common 2,000-step budget. The gate learns sharply
different behavior across architectures: Pointer Network adopts the adaptive policy,
GPN almost completely falls back to fixed, and AM remains conservative. Gated v2
avoids a meaningful loss to fixed in all three models and improves AM slightly and
Pointer Network substantially, so it passes the predefined pilot screen. Its next
test is a seed-1234 continuation to 10,000 steps, where the original Pointer Network
early advantage previously disappeared.
