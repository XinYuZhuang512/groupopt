import unittest

import torch

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.problems.tsp import DirectedTSPConstruction


class AdaptiveAttentionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        self.coordinates = torch.rand(4, 6, 2, generator=generator)
        self.model = AdaptiveAttentionModel(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=2,
            feed_forward_dim=64,
            normalization="layer",
        )

    def _assert_outputs_are_valid_tours(
        self, decode_type: str, base_mode: str = "adaptive"
    ) -> None:
        output = self.model(
            self.coordinates, decode_type=decode_type, base_mode=base_mode
        )
        process = DirectedTSPConstruction()

        self.assertEqual(output.tails.shape, (4, 6))
        self.assertEqual(output.heads.shape, (4, 6))
        self.assertEqual(output.successor.shape, (4, 6))
        self.assertTrue(torch.isfinite(output.cost).all())
        self.assertTrue(torch.isfinite(output.log_likelihood).all())

        for batch_index in range(4):
            state = process.initial_state(6)
            for tail, head in zip(
                output.tails[batch_index].tolist(), output.heads[batch_index].tolist()
            ):
                state = process.transition(state, tail, head)
            tour = process.decode(state)
            self.assertEqual(tuple(output.successor[batch_index].tolist()), tour.successor)

    def test_greedy_forward_produces_valid_tours(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self._assert_outputs_are_valid_tours("greedy")

    def test_sampling_forward_produces_valid_tours(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self._assert_outputs_are_valid_tours("sampling")

    def test_state_aware_adaptive_forward_produces_valid_tours(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self._assert_outputs_are_valid_tours(
                "greedy", base_mode="adaptive_state"
            )

    def test_state_aware_tail_embeddings_change_after_path_merge(self) -> None:
        from groupopt.problems.tsp_tensor import BatchedTSPState

        with torch.no_grad():
            node_embeddings, _ = self.model.encoder(self.coordinates)
            initial = BatchedTSPState.initialize(self.coordinates)
            updated = initial.update(
                torch.zeros(4, dtype=torch.long),
                torch.ones(4, dtype=torch.long),
            )
            initial_embeddings = self.model._state_aware_tail_embeddings(
                node_embeddings, initial
            )
            updated_embeddings = self.model._state_aware_tail_embeddings(
                node_embeddings, updated
            )

        self.assertFalse(torch.equal(initial_embeddings, updated_embeddings))
        state_delta = updated_embeddings - node_embeddings
        self.assertTrue(torch.allclose(state_delta[:, 0], state_delta[:, 1]))

    def test_state_aware_selector_supports_backpropagation(self) -> None:
        self.model.train()
        output = self.model(
            self.coordinates, decode_type="sampling", base_mode="adaptive_state"
        )
        (-output.log_likelihood.mean()).backward()

        gradient = self.model.project_tail_state.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_fixed_base_forward_is_a_continuous_anchored_path(self) -> None:
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.coordinates, decode_type="sampling", base_mode="fixed"
            )

        self.assertTrue(torch.equal(output.tails[:, 0], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(output.tails[:, 1:], output.heads[:, :-1]))
        self._assert_outputs_are_valid_tours("greedy", base_mode="fixed")

    def test_sampled_log_likelihood_supports_backpropagation(self) -> None:
        self.model.train()
        output = self.model(self.coordinates, decode_type="sampling")
        loss = -output.log_likelihood.mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()
