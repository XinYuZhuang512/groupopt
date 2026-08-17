import unittest

import torch

from groupopt.models.encoder_controls import JointEncoderModel, ModernNativeAttentionModel
from groupopt.problems.tsp import DirectedTSPConstruction


class JointEncoderModelTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(9753)
        self.coordinates = torch.rand(3, 6, 2, generator=generator)

    def test_controls_produce_valid_tours_and_gradients(self) -> None:
        process = DirectedTSPConstruction()
        for encoder_type, layers in (
            ("gat", 2),
            ("gru", 1),
            ("pointerformer", 2),
            ("geometric", 2),
            ("moe_transformer", 2),
        ):
            for base_mode in ("joint_fixed", "joint_free"):
                with self.subTest(encoder_type=encoder_type, base_mode=base_mode):
                    model = JointEncoderModel(
                        encoder_type=encoder_type,
                        embedding_dim=32,
                        n_encoder_layers=layers,
                        n_heads=4,
                    )
                    output = model(self.coordinates, base_mode=base_mode)
                    self.assertEqual(output.tails.shape, (3, 6))
                    self.assertTrue(torch.isfinite(output.cost).all())
                    self.assertTrue(torch.isfinite(output.log_likelihood).all())
                    (-output.log_likelihood.mean()).backward()
                    gradient = model.joint_action_scorer.project_tails.weight.grad
                    self.assertIsNotNone(gradient)
                    assert gradient is not None
                    self.assertGreater(gradient.abs().sum().item(), 0.0)
                    for batch_index in range(3):
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

    def test_geometric_encoder_is_rigid_motion_invariant(self) -> None:
        model = JointEncoderModel(
            "geometric", embedding_dim=32, n_encoder_layers=2, n_heads=4
        )
        angle = torch.tensor(0.73)
        rotation = torch.stack(
            (
                torch.stack((torch.cos(angle), -torch.sin(angle))),
                torch.stack((torch.sin(angle), torch.cos(angle))),
            )
        )
        transformed = self.coordinates @ rotation.T + torch.tensor([2.0, -3.0])
        original_nodes = model.encoder(self.coordinates)
        transformed_nodes = model.encoder(transformed)
        self.assertTrue(
            torch.allclose(original_nodes, transformed_nodes, atol=2e-5, rtol=2e-5)
        )

    def test_non_joint_modes_are_rejected(self) -> None:
        model = JointEncoderModel("gru", embedding_dim=16)
        with self.assertRaisesRegex(ValueError, "only joint_fixed and joint_free"):
            model(self.coordinates, base_mode="fixed")

    def test_modern_models_preserve_native_fixed_decoder(self) -> None:
        for model_type in (
            "reversible_transformer",
            "geometric_transformer",
            "sparse_moe_transformer",
        ):
            model = ModernNativeAttentionModel(
                model_type, embedding_dim=16, n_encoder_layers=1, n_heads=4
            )
            model.eval()
            with self.subTest(model=model_type), torch.no_grad():
                original = model(
                    self.coordinates, decode_type="greedy", base_mode="fixed"
                )
                fixed = model(
                    self.coordinates,
                    decode_type="greedy",
                    base_mode="native_conditional_fixed",
                )
                free = model(
                    self.coordinates,
                    decode_type="greedy",
                    base_mode="native_conditional_free",
                )
            self.assertTrue(torch.equal(original.tails, fixed.tails))
            self.assertTrue(torch.equal(original.heads, fixed.heads))
            self.assertTrue(torch.equal(original.cost, fixed.cost))
            self.assertTrue(torch.isfinite(free.cost).all())


if __name__ == "__main__":
    unittest.main()
