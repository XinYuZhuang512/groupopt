import tempfile
import unittest
from pathlib import Path

import torch

from experiments.evaluate_checkpoint import load_model
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

    def test_state_ablation_checkpoint_loads_without_legacy_missing_keys(self) -> None:
        model = AdaptiveAttentionModel(
            embedding_dim=16,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=32,
            normalization="layer",
        )
        config = {
            "base_mode": "adaptive_state_mean",
            "embedding_dim": 16,
            "encoder_layers": 1,
            "feed_forward_dim": 32,
            "heads": 4,
            "normalization": "layer",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            loaded, _ = load_model(checkpoint, config, torch.device("cpu"))

        self.assertIsInstance(loaded, AdaptiveAttentionModel)


if __name__ == "__main__":
    unittest.main()
