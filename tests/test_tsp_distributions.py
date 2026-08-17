import unittest

import torch

from groupopt.problems.distributions import TSP_DISTRIBUTIONS, generate_tsp_coordinates


class TSPDistributionTests(unittest.TestCase):
    def test_every_distribution_is_deterministic_and_bounded(self) -> None:
        for distribution in TSP_DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                first = generate_tsp_coordinates(8, 20, distribution, seed=17)
                second = generate_tsp_coordinates(8, 20, distribution, seed=17)
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first.shape, (8, 20, 2))
                self.assertTrue(torch.isfinite(first).all())
                self.assertGreaterEqual(first.min().item(), 0.0)
                self.assertLessEqual(first.max().item(), 1.0)

    def test_shifted_seed_changes_instances(self) -> None:
        first = generate_tsp_coordinates(4, 10, "clustered", seed=17)
        second = generate_tsp_coordinates(4, 10, "clustered", seed=18)
        self.assertFalse(torch.equal(first, second))

    def test_strong_clusters_have_shorter_nearest_neighbor_distances(self) -> None:
        uniform = generate_tsp_coordinates(128, 50, "uniform", seed=23)
        clustered = generate_tsp_coordinates(128, 50, "clustered_strong", seed=23)

        def nearest_mean(values: torch.Tensor) -> torch.Tensor:
            distances = torch.cdist(values, values)
            diagonal = torch.eye(values.size(1), dtype=torch.bool).unsqueeze(0)
            return distances.masked_fill(diagonal, torch.inf).amin(dim=-1).mean()

        self.assertLess(nearest_mean(clustered), nearest_mean(uniform))


if __name__ == "__main__":
    unittest.main()
