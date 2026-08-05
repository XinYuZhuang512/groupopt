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
