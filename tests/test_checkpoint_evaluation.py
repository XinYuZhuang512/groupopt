import unittest

import torch

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.models.ptrnet import AdaptivePointerNetwork


class CheckpointEvaluationModelBuilderTests(unittest.TestCase):
    def test_model_outputs_remain_compatible_with_independent_evaluation(self) -> None:
        coordinates = torch.rand(3, 5, 2)
        models = (
            AdaptiveAttentionModel(
                embedding_dim=16,
                n_heads=4,
                n_encoder_layers=1,
                feed_forward_dim=32,
                normalization="layer",
            ),
            AdaptivePointerNetwork(embedding_dim=16),
        )
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(
                    coordinates,
                    decode_type="greedy",
                    base_mode="adaptive_state",
                )
            self.assertEqual(output.cost.shape, (3,))
            self.assertTrue(torch.isfinite(output.cost).all())


if __name__ == "__main__":
    unittest.main()
