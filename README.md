# GroupOpt

GroupOpt is a stabilizer-guided construction interface for neural combinatorial
optimization. This branch is the minimal research release: it contains the method,
the three model integrations used to test cross-model behavior, and the comparison
experiments. Historical pilots, ablations, tests, server wrappers, and research notes
are intentionally excluded.

## Method

At each TSP construction step, the native decoder first provides
`P(head | tail, state)` for every legal tail. GroupOpt summarizes each conditional
distribution with entropy, expected edge length, and maximum probability, combines
that summary with the tail and open-path state, and selects the next tail. The head is
still selected by the model's native decoder. The problem layer masks premature
subtours and closes the tour only at the final step.

The two public experiment modes are:

- `native_conditional_fixed`: Original model;
- `native_conditional_free`: Original model + GroupOpt (Ours).

## Repository layout

```text
src/groupopt/
├── framework/    model-independent construction contracts and engine
├── problems/     TSP state, masks, transitions, objective, and data distributions
├── models/       AM, PtrNet, GPN, and GroupOpt tail-selection components
├── adapters/     one registry for the three model integrations
└── objectives/   REINFORCE and SYM-NCO training objectives

experiments/
├── train.py, train_ptrnet.py, train_gpn.py
├── evaluate.py
├── compare_models.sh
├── compare_symnco.sh
├── evaluate_distribution_shift.sh
├── validate_comparison.py
└── summarize_symnco.py
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,dev]'
```

## Minimal model use

```python
import torch

from groupopt.models.am import AttentionModel

model = AttentionModel()
coordinates = torch.rand(16, 50, 2)

original = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_conditional_fixed",
)
ours = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_conditional_free",
)
```

## Comparisons

Run commands from the repository root:

```bash
# Original vs Ours on AM, PtrNet, and GPN
bash experiments/compare_models.sh

# Original, Ours, SYM-NCO, and Ours + SYM-NCO on AM
bash experiments/compare_symnco.sh

# Evaluate trained checkpoints under distribution shift
bash experiments/evaluate_distribution_shift.sh

# Parameter matching and held-out seed checks also run automatically at the
# beginning of compare_models.sh.
```

Generated datasets, checkpoints, logs, and summaries are written under `artifacts/`
and are excluded from Git.
