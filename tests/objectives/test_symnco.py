import unittest

import torch

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.objectives import (
    SymmetryProjectionHead,
    augment_euclidean_symmetries,
    symnco_am_loss,
)


class SymNCOTests(unittest.TestCase):
    def test_augmentation_preserves_original_and_pairwise_distances(self) -> None:
        coordinates = torch.rand(3, 7, 2, generator=torch.Generator().manual_seed(1))
        augmented = augment_euclidean_symmetries(
            coordinates,
            factor=4,
            generator=torch.Generator().manual_seed(2),
        ).reshape(4, 3, 7, 2)

        self.assertTrue(torch.equal(augmented[0], coordinates))
        expected_distances = torch.cdist(coordinates, coordinates)
        for transformed in augmented[1:]:
            self.assertTrue(
                torch.allclose(
                    torch.cdist(transformed, transformed),
                    expected_distances,
                    atol=2e-6,
                    rtol=2e-6,
                )
            )

    def test_augmentation_is_generator_deterministic(self) -> None:
        coordinates = torch.rand(2, 5, 2)
        first = augment_euclidean_symmetries(
            coordinates, 3, torch.Generator().manual_seed(9)
        )
        second = augment_euclidean_symmetries(
            coordinates, 3, torch.Generator().manual_seed(9)
        )
        self.assertTrue(torch.equal(first, second))

    def test_loss_uses_within_instance_augmentation_baseline(self) -> None:
        cost = torch.tensor([1.0, 10.0, 3.0, 14.0])
        log_likelihood = torch.tensor([-1.0, -2.0, -3.0, -5.0], requires_grad=True)
        base_embeddings = torch.randn(2, 5, 8, requires_grad=True)
        embeddings = base_embeddings.repeat(2, 1, 1)

        terms = symnco_am_loss(cost, log_likelihood, embeddings, factor=2, alpha=0.1)
        terms.total.backward()

        self.assertAlmostEqual(terms.policy.item(), -2.0)
        self.assertAlmostEqual(terms.similarity.item(), 1.0, places=6)
        self.assertAlmostEqual(terms.invariance_penalty.item(), 0.0, places=6)
        self.assertTrue(torch.isfinite(log_likelihood.grad).all())
        self.assertTrue(torch.isfinite(base_embeddings.grad).all())

    def test_am_can_expose_symmetry_embeddings_without_changing_default_api(self) -> None:
        model = AdaptiveAttentionModel(
            embedding_dim=16,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=32,
            normalization="layer",
        )
        coordinates = torch.rand(2, 6, 2)
        ordinary = model(
            coordinates,
            decode_type="greedy",
            base_mode="native_conditional_fixed",
        )
        exposed = model(
            coordinates,
            decode_type="greedy",
            base_mode="native_conditional_fixed",
            return_symmetry_embeddings=True,
        )

        self.assertIsNone(ordinary.symmetry_node_embeddings)
        self.assertEqual(exposed.symmetry_node_embeddings.shape, (2, 6, 16))
        self.assertTrue(torch.allclose(ordinary.cost, exposed.cost))

    def test_projection_head_is_training_only_and_differentiable(self) -> None:
        head = SymmetryProjectionHead(8)
        embeddings = torch.randn(2, 5, 8, requires_grad=True)
        projected = head(embeddings)
        projected.square().mean().backward()

        self.assertEqual(projected.shape, embeddings.shape)
        self.assertTrue(torch.isfinite(embeddings.grad).all())


if __name__ == "__main__":
    unittest.main()
