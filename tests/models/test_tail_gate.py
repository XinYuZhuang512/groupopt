import unittest

import torch

from groupopt.models.tail_gate import (
    StateAwareTailGate,
    categorical_entropy,
    mix_with_fixed_tail,
)
from groupopt.problems.tsp_tensor import BatchedTSPState


class StateAwareTailGateTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        self.coordinates = torch.rand(3, 5, 2, generator=generator)
        self.embeddings = torch.rand(3, 5, 8, generator=generator)
        self.state = BatchedTSPState.initialize(self.coordinates)

    def test_initial_gate_is_conservative(self) -> None:
        gate = StateAwareTailGate(embedding_dim=8, initial_probability=0.1)
        probability = gate(self.state, self.embeddings)
        self.assertTrue(torch.allclose(probability, torch.full((3,), 0.1)))

    def test_fixed_mixture_preserves_probability_and_masks(self) -> None:
        adaptive_log_p = torch.log_softmax(torch.randn(3, 5), dim=1)
        fixed_tail = torch.tensor([0, 2, 4])
        gate_probability = torch.tensor([0.1, 0.25, 0.8])
        mixed_log_p = mix_with_fixed_tail(
            adaptive_log_p,
            fixed_tail,
            gate_probability,
        )
        mixed = mixed_log_p.exp()

        self.assertTrue(torch.allclose(mixed.sum(dim=1), torch.ones(3)))
        fixed_mass = mixed.gather(1, fixed_tail[:, None]).squeeze(1)
        self.assertTrue(torch.all(fixed_mass >= 1.0 - gate_probability))
        self.assertTrue(torch.isfinite(categorical_entropy(mixed_log_p)).all())

    def test_gate_and_adaptive_policy_receive_gradients(self) -> None:
        gate = StateAwareTailGate(embedding_dim=8)
        adaptive_logits = torch.randn(3, 5, requires_grad=True)
        adaptive_log_p = torch.log_softmax(adaptive_logits, dim=1)
        probability = gate(self.state, self.embeddings)
        mixed_log_p = mix_with_fixed_tail(
            adaptive_log_p,
            torch.zeros(3, dtype=torch.long),
            probability,
        )
        loss = -mixed_log_p[:, 1].mean()
        loss.backward()

        self.assertIsNotNone(adaptive_logits.grad)
        self.assertIsNotNone(gate.projection.weight.grad)
        assert adaptive_logits.grad is not None
        assert gate.projection.weight.grad is not None
        self.assertGreater(adaptive_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(gate.projection.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
