import unittest

import torch

from groupopt.models.native_conditional import (
    joint_action_entropy,
    masked_conditional_log_probabilities,
    native_head_summary,
)


class NativeConditionalUtilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(731)
        self.logits = torch.randn(2, 4, 4, generator=generator)
        self.mask = torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1).clone()
        self.mask[:, 3] = True

    def test_conditional_rows_are_normalized_and_fully_masked_rows_stay_masked(self) -> None:
        log_p = masked_conditional_log_probabilities(self.logits, self.mask, 1.0)
        probabilities = log_p.exp()
        self.assertTrue(torch.allclose(probabilities[:, :3].sum(dim=-1), torch.ones(2, 3)))
        self.assertTrue(torch.isneginf(log_p[:, 3]).all())

    def test_conditional_policy_is_invariant_to_tail_specific_logit_offsets(self) -> None:
        offsets = torch.tensor([[-9.0, 2.0, 11.0, 4.0], [7.0, -3.0, 5.0, 1.0]])
        original = masked_conditional_log_probabilities(self.logits, self.mask, 1.0)
        shifted = masked_conditional_log_probabilities(
            self.logits + offsets.unsqueeze(-1), self.mask, 1.0
        )
        finite = torch.isfinite(original)
        self.assertTrue(torch.allclose(original[finite], shifted[finite], atol=1e-6))

    def test_detached_summary_does_not_backpropagate_into_native_logits(self) -> None:
        logits = self.logits.detach().clone().requires_grad_(True)
        log_p = masked_conditional_log_probabilities(logits, self.mask, 1.0)
        distances = torch.rand_like(logits)
        projection = torch.nn.Linear(3, 5, bias=False)
        projection(native_head_summary(log_p, distances).detach()).sum().backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(projection.weight.grad)

    def test_factorized_action_entropy_is_finite(self) -> None:
        head_log_p = masked_conditional_log_probabilities(self.logits, self.mask, 1.0)
        tail_logits = torch.randn(2, 4)
        tail_logits[:, 3] = -torch.inf
        tail_log_p = torch.log_softmax(tail_logits, dim=-1)
        entropy = joint_action_entropy(tail_log_p, head_log_p)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy > 0.0).all())


if __name__ == "__main__":
    unittest.main()
