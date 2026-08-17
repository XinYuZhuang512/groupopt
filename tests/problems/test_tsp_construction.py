import unittest
from itertools import pairwise, permutations

from groupopt.problems.tsp import (
    DirectedTSPConstruction,
    InvalidAction,
    TSPState,
    tour_length,
)


def _construct_cycle(order: tuple[int, ...]):
    process = DirectedTSPConstruction()
    state = process.initial_state(len(order))
    for source, target in pairwise(order):
        state = process.transition(state, source, target)
    state = process.transition(state, order[-1], order[0])
    return process, state


class DirectedTSPConstructionTests(unittest.TestCase):
    def test_initial_state_exposes_all_adaptive_bases(self) -> None:
        process = DirectedTSPConstruction()
        state = process.initial_state(4)

        self.assertEqual(process.base_candidates(state), (0, 1, 2, 3))
        self.assertEqual(process.representative_candidates(state, 1), (0, 2, 3))
        self.assertEqual(state.num_components, 4)

    def test_mask_rejects_premature_subtour(self) -> None:
        process = DirectedTSPConstruction()
        state = process.initial_state(4)
        state = process.transition(state, 0, 1)

        self.assertNotIn(0, process.representative_candidates(state, 1))
        with self.assertRaisesRegex(InvalidAction, "masked"):
            process.transition(state, 1, 0)

    def test_known_cycle_decodes_to_one_hamiltonian_cycle(self) -> None:
        process, state = _construct_cycle((0, 2, 1, 3))
        tour = process.decode(state)

        self.assertTrue(process.is_terminal(state))
        self.assertEqual(tour.order, (0, 2, 1, 3))
        self.assertEqual(tour.successor, (2, 3, 1, 0))
        self.assertEqual(set(tour.edges), {(0, 2), (2, 1), (1, 3), (3, 0)})

    def test_every_directed_tour_has_a_legal_construction(self) -> None:
        """Exhaust all tours with vertex 0 as the canonical first vertex."""
        for n in range(2, 7):
            for suffix in permutations(range(1, n)):
                with self.subTest(n=n, suffix=suffix):
                    order = (0, *suffix)
                    process, state = _construct_cycle(order)
                    self.assertEqual(process.decode(state).order, order)

    def test_every_legal_trajectory_for_n5_terminates_in_a_valid_tour(self) -> None:
        process = DirectedTSPConstruction()
        terminal_count = 0

        def visit(state) -> None:
            nonlocal terminal_count
            if process.is_terminal(state):
                tour = process.decode(state)
                self.assertEqual(len(set(tour.order)), 5)
                terminal_count += 1
                return

            for selected_base in process.base_candidates(state):
                representatives = process.representative_candidates(state, selected_base)
                for representative in representatives:
                    visit(process.transition(state, selected_base, representative))

        visit(process.initial_state(5))
        self.assertEqual(terminal_count, 2880)

    def test_sequential_base_embeds_one_path_decoding(self) -> None:
        process = DirectedTSPConstruction()
        state = process.initial_state(4)

        self.assertEqual(process.sequential_base(state), 0)
        state = process.transition(state, process.sequential_base(state), 2)
        self.assertEqual(process.sequential_base(state), 2)
        state = process.transition(state, process.sequential_base(state), 1)
        self.assertEqual(process.sequential_base(state), 1)
        state = process.transition(state, process.sequential_base(state), 3)
        self.assertEqual(process.sequential_base(state), 3)
        state = process.transition(state, process.sequential_base(state), 0)

        self.assertEqual(process.decode(state).order, (0, 2, 1, 3))

    def test_tour_length_supports_asymmetric_distances(self) -> None:
        process, state = _construct_cycle((0, 1, 2))
        distances = (
            (0.0, 1.0, 9.0),
            (4.0, 0.0, 2.0),
            (3.0, 8.0, 0.0),
        )

        self.assertAlmostEqual(tour_length(process.decode(state), distances), 6.0)

    def test_invalid_instance_and_nonterminal_decode_are_rejected(self) -> None:
        process = DirectedTSPConstruction()
        with self.assertRaisesRegex(ValueError, "at least two"):
            process.initial_state(1)

        state = process.initial_state(3)
        with self.assertRaisesRegex(ValueError, "non-terminal"):
            process.decode(state)

    def test_validator_rejects_component_labels_that_disagree_with_graph(self) -> None:
        process = DirectedTSPConstruction()
        state = TSPState(
            n=3,
            successor=(1, None, None),
            predecessor=(None, 0, None),
            component=(0, 1, 1),
            edges_added=1,
        )

        with self.assertRaisesRegex(ValueError, "component labels"):
            process.validate_state(state)


if __name__ == "__main__":
    unittest.main()
