import unittest

import torch

from groupopt.problems.tsp import DirectedTSPConstruction
from groupopt.problems.tsp_tensor import BatchedTSPState


class BatchedTSPStateTests(unittest.TestCase):
    def test_tensor_state_matches_reference_process(self) -> None:
        coordinates = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            ]
        )
        tail_steps = ((0, 0), (1, 2), (2, 1), (3, 3))
        head_steps = ((1, 2), (2, 1), (3, 3), (0, 0))
        tensor_state = BatchedTSPState.initialize(coordinates)
        process = DirectedTSPConstruction()
        reference_states = [process.initial_state(4), process.initial_state(4)]

        for tails, heads in zip(tail_steps, head_steps):
            selected_tails = torch.tensor(tails, dtype=torch.long)
            tensor_head_mask = tensor_state.head_mask(selected_tails)
            for batch_index, selected_tail in enumerate(tails):
                legal = process.representative_candidates(
                    reference_states[batch_index], selected_tail
                )
                expected_mask = tuple(vertex not in legal for vertex in range(4))
                self.assertEqual(
                    tuple(tensor_head_mask[batch_index].tolist()), expected_mask
                )

            tensor_state = tensor_state.update(
                selected_tails, torch.tensor(heads, dtype=torch.long)
            )
            reference_states = [
                process.transition(state, tail, head)
                for state, tail, head in zip(reference_states, tails, heads)
            ]
            for batch_index, reference in enumerate(reference_states):
                self.assertEqual(
                    tuple(tensor_state.successor[batch_index].tolist()),
                    tuple(-1 if value is None else value for value in reference.successor),
                )
                self.assertEqual(
                    tuple(tensor_state.component[batch_index].tolist()),
                    reference.component,
                )

        self.assertTrue(tensor_state.terminal)

    def test_tensor_state_rejects_premature_cycle(self) -> None:
        coordinates = torch.rand(1, 4, 2)
        state = BatchedTSPState.initialize(coordinates)
        state = state.update(torch.tensor([0]), torch.tensor([1]))

        with self.assertRaisesRegex(ValueError, "violates"):
            state.update(torch.tensor([1]), torch.tensor([0]))

    def test_sequential_base_tracks_the_anchored_path_tail(self) -> None:
        state = BatchedTSPState.initialize(torch.rand(2, 4, 2))
        self.assertTrue(torch.equal(state.sequential_base(), torch.tensor([0, 0])))

        state = state.update(torch.tensor([0, 0]), torch.tensor([2, 1]))
        self.assertTrue(torch.equal(state.sequential_base(), torch.tensor([2, 1])))

    def test_path_state_features_follow_component_merges(self) -> None:
        embeddings = torch.tensor([[[1.0], [3.0], [8.0], [12.0]]])
        state = BatchedTSPState.initialize(torch.rand(1, 4, 2))
        state = state.update(torch.tensor([0]), torch.tensor([1]))

        features = state.path_state_features(embeddings)
        self.assertEqual(features.shape, (1, 4, 3))
        self.assertTrue(torch.equal(features[0, 0], features[0, 1]))
        self.assertTrue(torch.allclose(features[0, 0], torch.tensor([2.0, 1.0, 0.5])))

        with self.assertRaisesRegex(ValueError, "node_embeddings"):
            state.path_state_features(torch.rand(2, 4, 3))

    def test_joint_edge_mask_matches_reference_candidates(self) -> None:
        coordinates = torch.rand(2, 5, 2)
        state = BatchedTSPState.initialize(coordinates)
        state = state.update(torch.tensor([0, 3]), torch.tensor([1, 4]))
        process = DirectedTSPConstruction()
        references = [
            process.transition(process.initial_state(5), 0, 1),
            process.transition(process.initial_state(5), 3, 4),
        ]

        mask = state.edge_action_mask()
        for batch_index, reference in enumerate(references):
            expected = {
                (tail, head)
                for tail in process.base_candidates(reference)
                for head in process.representative_candidates(reference, tail)
            }
            actual = {
                (tail, head)
                for tail in range(5)
                for head in range(5)
                if not mask[batch_index, tail, head]
            }
            self.assertEqual(actual, expected)

    def test_joint_fixed_mask_only_exposes_anchored_tail(self) -> None:
        state = BatchedTSPState.initialize(torch.rand(2, 4, 2))
        state = state.update(torch.tensor([0, 0]), torch.tensor([2, 1]))
        fixed_tail = state.sequential_base()
        mask = state.edge_action_mask(fixed_tail)

        legal_tails = (~mask).any(dim=2).to(torch.long).argmax(dim=1)
        self.assertTrue(torch.equal(legal_tails, fixed_tail))
        self.assertTrue((~mask).flatten(1).any(dim=1).all())


if __name__ == "__main__":
    unittest.main()
