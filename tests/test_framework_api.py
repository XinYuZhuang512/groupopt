import unittest

import torch

from groupopt.framework import (
    construct,
)
from groupopt.framework.neural import (
    BatchedConstructionProcess,
    ConstructionModel,
    ConstructionOutput,
)
from groupopt.models.am import AdaptiveAttentionModel
from groupopt.models.gpn import AdaptiveGraphPointerNetwork
from groupopt.models.ptrnet import AdaptivePointerNetwork
from groupopt.problems.tsp import DirectedTSPConstruction
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState


class _RecordingTSPProcess:
    def __init__(self) -> None:
        self.delegate = BatchedTSPConstruction()
        self.transitions = 0
        self.objective_calls = 0
        self.solution_calls = 0

    def initial_state(self, instance: torch.Tensor) -> BatchedTSPState:
        return self.delegate.initial_state(instance)

    def is_terminal(self, state: BatchedTSPState) -> bool:
        return self.delegate.is_terminal(state)

    def base_mask(self, state: BatchedTSPState) -> torch.Tensor:
        return self.delegate.base_mask(state)

    def representative_mask(
        self, state: BatchedTSPState, selected_base: torch.Tensor
    ) -> torch.Tensor:
        return self.delegate.representative_mask(state, selected_base)

    def action_mask(
        self, state: BatchedTSPState, fixed_base: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.delegate.action_mask(state, fixed_base)

    def fixed_base(self, state: BatchedTSPState, anchor: int = 0) -> torch.Tensor:
        return self.delegate.fixed_base(state, anchor)

    def transition(
        self,
        state: BatchedTSPState,
        selected_base: torch.Tensor,
        selected_representative: torch.Tensor,
    ) -> BatchedTSPState:
        self.transitions += 1
        return self.delegate.transition(state, selected_base, selected_representative)

    def objective(
        self,
        state: BatchedTSPState,
        selected_bases: torch.Tensor,
        selected_representatives: torch.Tensor,
    ) -> torch.Tensor:
        self.objective_calls += 1
        return self.delegate.objective(state, selected_bases, selected_representatives)

    def solution(self, state: BatchedTSPState) -> torch.Tensor:
        self.solution_calls += 1
        return self.delegate.solution(state)


class FrameworkAPITests(unittest.TestCase):
    def test_reference_engine_is_independent_of_any_neural_model(self) -> None:
        trace = construct(
            DirectedTSPConstruction(),
            5,
            select_base=lambda _state, candidates: candidates[0],
            select_representative=lambda _state, _base, candidates: candidates[0],
        )

        self.assertEqual(len(trace.actions), 5)
        self.assertEqual(len(set(trace.solution.order)), 5)

    def test_tensor_tsp_plugin_satisfies_public_protocol(self) -> None:
        self.assertIsInstance(BatchedTSPConstruction(), BatchedConstructionProcess)

    def test_all_reference_models_delegate_construction_semantics(self) -> None:
        builders = (
            lambda process: AdaptiveAttentionModel(
                embedding_dim=16,
                n_heads=4,
                n_encoder_layers=1,
                feed_forward_dim=32,
                normalization="layer",
                construction_process=process,
            ),
            lambda process: AdaptivePointerNetwork(
                embedding_dim=16,
                n_encoder_layers=1,
                construction_process=process,
            ),
            lambda process: AdaptiveGraphPointerNetwork(
                embedding_dim=16,
                n_encoder_layers=1,
                construction_process=process,
            ),
        )
        coordinates = torch.rand(2, 5, 2)
        for builder in builders:
            with self.subTest(builder=builder):
                process = _RecordingTSPProcess()
                model = builder(process)
                output = model(
                    coordinates,
                    decode_type="greedy",
                    base_mode="native_conditional_fixed",
                )
                self.assertIsInstance(model, ConstructionModel)
                self.assertIsInstance(output, ConstructionOutput)
                self.assertEqual(process.transitions, 5)
                self.assertEqual(process.objective_calls, 1)
                self.assertEqual(process.solution_calls, 1)


if __name__ == "__main__":
    unittest.main()
