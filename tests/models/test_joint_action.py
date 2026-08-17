import unittest

import torch

from groupopt.models.joint_action import PairActionScorer, decode_joint_actions
from groupopt.problems.tsp_tensor import BatchedTSPState


class JointActionDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(31415)
        self.coordinates = torch.rand(3, 6, 2, generator=generator)
        self.embeddings = torch.rand(3, 6, 16, generator=generator)
        self.scorer = PairActionScorer(embedding_dim=16)

    def test_scorer_normalizes_only_over_legal_pairs(self) -> None:
        state = BatchedTSPState.initialize(self.coordinates)
        mask = state.edge_action_mask()
        log_p = self.scorer(
            self.coordinates,
            self.embeddings,
            state,
            mask,
            temperature=1.0,
        )

        self.assertEqual(log_p.shape, (3, 6, 6))
        self.assertTrue(torch.isneginf(log_p[mask]).all())
        self.assertTrue(
            torch.allclose(log_p.exp().sum(dim=(1, 2)), torch.ones(3))
        )

    def test_joint_modes_produce_finite_differentiable_outputs(self) -> None:
        for base_mode in ("joint_fixed", "joint_free"):
            with self.subTest(base_mode=base_mode):
                embeddings = self.embeddings.detach().clone().requires_grad_(True)
                output = decode_joint_actions(
                    self.coordinates,
                    embeddings,
                    self.scorer,
                    base_mode,
                    "sampling",
                    anchor=0,
                    temperature=1.0,
                    generator=torch.Generator().manual_seed(2718),
                )
                self.assertTrue(torch.isfinite(output.cost).all())
                self.assertTrue(torch.isfinite(output.log_likelihood).all())
                self.assertTrue(torch.isfinite(output.action_entropy).all())
                (-output.log_likelihood.mean()).backward()
                self.assertIsNotNone(embeddings.grad)
                assert embeddings.grad is not None
                self.assertGreater(embeddings.grad.abs().sum().item(), 0.0)
                self.scorer.zero_grad(set_to_none=True)

    def test_joint_fixed_tracks_one_anchored_path(self) -> None:
        output = decode_joint_actions(
            self.coordinates,
            self.embeddings,
            self.scorer,
            "joint_fixed",
            "sampling",
            anchor=0,
            temperature=1.0,
            generator=torch.Generator().manual_seed(1618),
        )
        self.assertTrue(torch.equal(output.tails[:, 0], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(output.tails[:, 1:], output.heads[:, :-1]))


if __name__ == "__main__":
    unittest.main()
