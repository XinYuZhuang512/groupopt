import unittest

import torch

from groupopt.models.ptrnet import AdaptivePointerNetwork
from groupopt.problems.tsp import DirectedTSPConstruction


class AdaptivePointerNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(4321)
        self.coordinates = torch.rand(4, 6, 2, generator=generator)
        self.model = AdaptivePointerNetwork(embedding_dim=32)

    def _assert_valid_tours(self, base_mode: str, decode_type: str) -> None:
        output = self.model(self.coordinates, base_mode=base_mode, decode_type=decode_type)
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
                "gated_adaptive_state",
                "joint_fixed",
                "joint_free",
                "native_fixed",
                "native_free",
                "native_conditional_fixed",
                "native_conditional_free",
            ):
                for decode_type in ("greedy", "sampling"):
                    with self.subTest(base_mode=base_mode, decode_type=decode_type):
                        self._assert_valid_tours(base_mode, decode_type)

    def test_fixed_mode_builds_one_continuous_path(self) -> None:
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.coordinates, base_mode="fixed", decode_type="sampling")
        self.assertTrue(torch.equal(output.tails[:, 0], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(output.tails[:, 1:], output.heads[:, :-1]))

    def test_state_aware_mode_backpropagates_through_path_features(self) -> None:
        self.model.train()
        output = self.model(
            self.coordinates,
            base_mode="adaptive_state",
            decode_type="sampling",
        )
        (-output.log_likelihood.mean()).backward()

        gradient = self.model.project_tail_state.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_gated_mode_backpropagates_through_gate(self) -> None:
        self.model.train()
        output = self.model(
            self.coordinates,
            base_mode="gated_adaptive_state",
            decode_type="sampling",
        )
        (-output.log_likelihood.mean()).backward()
        gradient = self.model.tail_gate.projection.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(output.tail_entropy).all())

    def test_joint_free_backpropagates_through_pair_scorer(self) -> None:
        self.model.train()
        output = self.model(
            self.coordinates,
            base_mode="joint_free",
            decode_type="sampling",
        )
        (-output.log_likelihood.mean()).backward()
        gradient = self.model.joint_action_scorer.project_tails.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(output.action_entropy).all())

    def test_native_fixed_exactly_reproduces_fixed_decoder(self) -> None:
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.coordinates, base_mode="fixed", decode_type="greedy")
            lifted = self.model(self.coordinates, base_mode="native_fixed", decode_type="greedy")
        self.assertTrue(torch.equal(original.tails, lifted.tails))
        self.assertTrue(torch.equal(original.heads, lifted.heads))
        self.assertTrue(torch.equal(original.successor, lifted.successor))
        self.assertTrue(torch.allclose(original.log_likelihood, lifted.log_likelihood))

    def test_native_free_uses_tail_state_and_native_head_pointer(self) -> None:
        self.model.train()
        output = self.model(self.coordinates, base_mode="native_free", decode_type="sampling")
        (-output.log_likelihood.mean()).backward()
        for parameter in (
            self.model.project_tail_state.weight,
            self.model.project_head_context.weight,
            self.model.head_pointer.project_nodes.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(output.action_entropy).all())

    def test_native_conditional_modes_preserve_fixed_and_train_tail_summary(self) -> None:
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.coordinates, base_mode="fixed", decode_type="greedy")
            fixed = self.model(
                self.coordinates,
                base_mode="native_conditional_fixed",
                decode_type="greedy",
            )
        self.assertTrue(torch.equal(original.tails, fixed.tails))
        self.assertTrue(torch.equal(original.heads, fixed.heads))
        self.assertTrue(torch.allclose(original.log_likelihood, fixed.log_likelihood))

        self.model.train()
        free = self.model(
            self.coordinates,
            base_mode="native_conditional_free",
            decode_type="sampling",
        )
        (-free.log_likelihood.mean()).backward()
        gradient = self.model.project_native_head_summary.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(free.action_entropy).all())


if __name__ == "__main__":
    unittest.main()
