# Graph Pointer Network integration

## Scope

Here GPN means the Graph Pointer Network introduced by Ma et al., not a generic
abbreviation for any graph-pointer architecture. The original model extends a Pointer
Network with graph embedding layers and uses a vector context relative to the current
city. The authors' public implementation targets PyTorch 1.1, so this repository uses
a clean current-PyTorch implementation rather than vendoring the historical code.

Primary references:

- Paper: https://arxiv.org/abs/1911.04936
- Author implementation: https://github.com/qiang-ma/graph-pointer-network

## Framework adapter

`AdaptiveGraphPointerNetwork` keeps the GPN-specific scorer separate from the shared
TSP construction state. Its model-specific components are:

1. residual complete-graph message-passing layers;
2. candidate embeddings relative to the previous selected head for base selection;
3. candidate embeddings relative to the selected tail for representative selection;
4. an LSTM decoder and additive pointer attention.

The model returns scores only. `BatchedTSPState` still owns legal masks, path merges,
termination, and tour decoding. No GPN-specific feasibility rule is introduced.

This is an architectural adaptation of the paper's GPN principles, not a numerical
reproduction of its old training code. In particular, the original one-city decoder
is replaced by the framework's tail/head decisions while preserving graph messages
and vector context.

## First pilot

Run TSP50 with seed 1234 for 2,000 steps under three construction modes:

- `fixed`;
- `adaptive`;
- `adaptive_state`.

Use batch size 512, validation size 1,024, embedding width 128, three graph embedding
layers, greedy validation every 100 steps, and the same centered REINFORCE objective
as the AM and Pointer Network experiments. This pilot answers only whether the GPN
adapter trains normally and whether adaptive state is promising enough for a
10,000-step follow-up.

### Pilot results

| Mode | Best validation cost | Peak GPU memory |
| --- | ---: | ---: |
| `fixed` | 6.3601 | 1.75 GiB |
| `adaptive` | 6.5159 | 3.03 GiB |
| `adaptive_state` | 6.4031 | 5.54 GiB |

State improves on plain adaptive by about 1.73%, but remains about 0.68% worse than
fixed at 2,000 steps. Because the state-aware curve obtains its best value at step
1,900 and is still improving, continue only `fixed` and `adaptive_state` from their
existing checkpoints to 10,000 total steps. Plain adaptive is screened out.

## Seed-1234 long-horizon result

| Mode | Best validation cost | Independent mean cost |
| --- | ---: | ---: |
| `fixed` | 6.2101 | 6.2248 |
| `adaptive_state` | **6.1966** | **6.2065** |

Both modes were continued from step 2,000 to step 10,000. On the independent
10,000-instance test set, state-aware adaptive construction improves on fixed by
0.0183 cost units (about 0.29%). The paired per-instance 95% CI for
`adaptive_state - fixed` is `[-0.0242, -0.0124]`, with a 52.07% instance win rate.
This is a positive single-training-seed result and motivates a small multi-seed
confirmation rather than an immediate full experiment matrix.

## Multi-seed confirmation

Train the same two modes from scratch with seeds 2345, 3456, and 4567, preserving
the seed-1234 protocol exactly. Evaluate each best checkpoint on the same 10,000
independent TSP50 instances.

| Training seed | Fixed test cost | State test cost | State - fixed |
| ---: | ---: | ---: | ---: |
| 1234 | 6.2248 | 6.2065 | -0.0183 |
| 2345 | 6.2149 | 6.2051 | -0.0098 |
| 3456 | 6.2256 | 6.2420 | +0.0164 |
| 4567 | 6.2194 | 6.2274 | +0.0080 |
| Mean | 6.2212 | 6.2202 | -0.0009 |

State wins two of four training seeds. The mean improvement is only 0.0009 cost
units (about 0.015%), and the training-seed-level paired 95% CI is
`[-0.0263, 0.0244]`. The single-seed signal therefore does not generalize into a
stable GPN advantage. The framework is operational across architectures, but this
state policy's performance gain remains model dependent; do not expand this GPN
branch without a revised mechanism.
