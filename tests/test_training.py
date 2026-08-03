import unittest

import torch

from groupopt.training import reinforce_loss


class TrainingUtilityTests(unittest.TestCase):
    def test_reinforce_loss_has_expected_centered_gradient(self) -> None:
        cost = torch.tensor([1.0, 3.0])
        log_likelihood = torch.tensor([-2.0, -4.0], requires_grad=True)

        loss = reinforce_loss(cost, log_likelihood)
        loss.backward()

        self.assertAlmostEqual(loss.item(), -1.0)
        self.assertTrue(torch.equal(log_likelihood.grad, torch.tensor([-0.5, 0.5])))

    def test_reinforce_loss_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal shape"):
            reinforce_loss(torch.ones(2), torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
