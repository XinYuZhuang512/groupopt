import unittest

import torch

from groupopt.models.gpn import AdaptiveGraphPointerNetwork
from groupopt.problems.tsp import DirectedTSPConstruction


class AdaptiveGraphPointerNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(8642)
        self.coordinates = torch.rand(4, 6, 2, generator=generator)
        self.model = AdaptiveGraphPointerNetwork(
            embedding_dim=32,
            n_encoder_layers=2,
        )

    def _assert_valid_tours(self, base_mode: str, decode_type: str) -> None:
        output = self.model(
            self.coordinates,
            base_mode=base_mode,
            decode_type=decode_type,
        )
        process = DirectedTSPConstruction()
        self.assertEqual(output.tails.shape, (4, 6))
        self.assertTrue(torch.isfinite(output.cost).all())
        self.assertTrue(torch.isfinite(output.log_likelihood).all())

        for batch_index in range(4):
            state = process.initial_state(6)
            for tail, head in zip(
                output.tails[batch_index].tolist(),
                output.heads[batch_index].tolist(),
            ):
                state = process.transition(state, tail, head)
            self.assertEqual(
                tuple(output.successor[batch_index].tolist()),
                process.decode(state).successor,
            )

    def test_all_base_modes_produce_valid_tours(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for base_mode in (
                "fixed",
                "adaptive",
                "adaptive_static",
                "adaptive_state_mean",
                "adaptive_state_start",
                "adaptive_state_size",
                "adaptive_state",
            ):
                for decode_type in ("greedy", "sampling"):
                    with self.subTest(base_mode=base_mode, decode_type=decode_type):
                        self._assert_valid_tours(base_mode, decode_type)

    def test_fixed_mode_builds_one_continuous_path(self) -> None:
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.coordinates,
                base_mode="fixed",
                decode_type="sampling",
            )
        self.assertTrue(torch.equal(output.tails[:, 0], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(output.tails[:, 1:], output.heads[:, :-1]))

    def test_graph_messages_and_state_features_support_backpropagation(self) -> None:
        self.model.train()
        output = self.model(
            self.coordinates,
            base_mode="adaptive_state",
            decode_type="sampling",
        )
        (-output.log_likelihood.mean()).backward()

        state_gradient = self.model.project_tail_state.weight.grad
        graph_gradient = self.model.encoder.layers[0].project_neighbors.weight.grad
        self.assertIsNotNone(state_gradient)
        self.assertIsNotNone(graph_gradient)
        assert state_gradient is not None and graph_gradient is not None
        self.assertGreater(state_gradient.abs().sum().item(), 0.0)
        self.assertGreater(graph_gradient.abs().sum().item(), 0.0)

    def test_relative_vector_context_changes_with_current_node(self) -> None:
        candidates = self.model._relative_context(
            self.coordinates,
            self.coordinates[:, 0],
        )
        shifted = self.model._relative_context(
            self.coordinates,
            self.coordinates[:, 1],
        )
        self.assertFalse(torch.equal(candidates, shifted))


if __name__ == "__main__":
    unittest.main()
