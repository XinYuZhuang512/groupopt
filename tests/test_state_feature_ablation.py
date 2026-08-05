import unittest

import torch

from groupopt.models.state_features import select_tail_state_features
from groupopt.problems.tsp_tensor import BatchedTSPState


class StateFeatureAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = torch.tensor([[[1.0], [3.0], [8.0], [12.0]]])
        initial = BatchedTSPState.initialize(torch.rand(1, 4, 2))
        self.state = initial.update(torch.tensor([0]), torch.tensor([1]))

    def test_feature_blocks_are_isolated(self) -> None:
        mean = select_tail_state_features(
            "adaptive_state_mean", self.state, self.embeddings
        )
        start = select_tail_state_features(
            "adaptive_state_start", self.state, self.embeddings
        )
        size = select_tail_state_features(
            "adaptive_state_size", self.state, self.embeddings
        )
        full = select_tail_state_features(
            "adaptive_state", self.state, self.embeddings
        )

        self.assertTrue(torch.equal(mean[0, 0], torch.tensor([2.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(start[0, 0], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(size[0, 0], torch.tensor([0.0, 0.0, 0.5])))
        self.assertTrue(torch.equal(full[0, 0], torch.tensor([2.0, 1.0, 0.5])))

    def test_static_control_does_not_change_after_path_merge(self) -> None:
        initial = BatchedTSPState.initialize(torch.rand(1, 4, 2))
        initial_features = select_tail_state_features(
            "adaptive_static", initial, self.embeddings
        )
        updated_features = select_tail_state_features(
            "adaptive_static", self.state, self.embeddings
        )
        self.assertTrue(torch.equal(initial_features, updated_features))
        self.assertTrue(
            torch.equal(initial_features[0, 0], torch.tensor([1.0, 1.0, 0.25]))
        )

    def test_non_feature_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not use"):
            select_tail_state_features("adaptive", self.state, self.embeddings)


if __name__ == "__main__":
    unittest.main()
