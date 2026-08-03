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

    def _assert_outputs_are_valid_tours(self, decode_type: str) -> None:
        output = self.model(self.coordinates, decode_type=decode_type)
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
