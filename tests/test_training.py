import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.train_am_experiment import _write_or_validate_config
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

    def test_resume_may_extend_but_not_shorten_training_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            initial = {"base_mode": "adaptive_state_mean", "steps": 2000}
            path.write_text(json.dumps(initial), encoding="utf-8")

            extended = {**initial, "steps": 10000}
            _write_or_validate_config(path, extended, resuming=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), extended)

            with self.assertRaisesRegex(ValueError, "cannot be shorter"):
                _write_or_validate_config(path, initial, resuming=True)

    def test_resume_still_rejects_other_configuration_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            initial = {"base_mode": "adaptive_state_mean", "steps": 2000}
            path.write_text(json.dumps(initial), encoding="utf-8")

            changed = {"base_mode": "adaptive_state_start", "steps": 10000}
            with self.assertRaisesRegex(ValueError, "config differs"):
                _write_or_validate_config(path, changed, resuming=True)


if __name__ == "__main__":
    unittest.main()
