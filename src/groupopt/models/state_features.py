"""Shared path-state feature ablations for adaptive base selectors."""

from __future__ import annotations

import torch
from torch import Tensor

from groupopt.problems.tsp_tensor import BatchedTSPState

ADAPTIVE_BASE_MODES = (
    "adaptive",
    "adaptive_static",
    "adaptive_state_mean",
    "adaptive_state_start",
    "adaptive_state_size",
    "adaptive_state",
    "gated_adaptive_state",
)
FEATURE_BASE_MODES = ADAPTIVE_BASE_MODES[1:]
JOINT_BASE_MODES = ("joint_fixed", "joint_free")
BASE_MODES = ("fixed", *ADAPTIVE_BASE_MODES, *JOINT_BASE_MODES)


def select_tail_state_features(
    base_mode: str,
    state: BatchedTSPState,
    node_embeddings: Tensor,
) -> Tensor:
    """Return a parameter-matched feature tensor for one ablation mode."""
    if base_mode not in FEATURE_BASE_MODES:
        raise ValueError(f"base mode does not use tail state features: {base_mode}")

    embedding_dim = node_embeddings.size(-1)
    if base_mode == "adaptive_static":
        initial_size = torch.full(
            (*node_embeddings.shape[:2], 1),
            1.0 / state.n,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return torch.cat((node_embeddings, node_embeddings, initial_size), dim=-1)

    full_features = state.path_state_features(node_embeddings)
    component_mean = full_features[..., :embedding_dim]
    path_start = full_features[..., embedding_dim : 2 * embedding_dim]
    path_size = full_features[..., 2 * embedding_dim :]
    zero_embedding = torch.zeros_like(component_mean)
    zero_size = torch.zeros_like(path_size)

    if base_mode == "adaptive_state_mean":
        return torch.cat((component_mean, zero_embedding, zero_size), dim=-1)
    if base_mode == "adaptive_state_start":
        return torch.cat((zero_embedding, path_start, zero_size), dim=-1)
    if base_mode == "adaptive_state_size":
        return torch.cat((zero_embedding, zero_embedding, path_size), dim=-1)
    return full_features
