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


if __name__ == "__main__":
    unittest.main()
