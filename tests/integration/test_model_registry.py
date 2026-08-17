import unittest

from torch import nn

from groupopt.adapters import ModelRegistry, build_model, model_registry
from groupopt.models.am import AdaptiveAttentionModel


class ModelRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_all_experiment_families(self) -> None:
        self.assertEqual(
            set(model_registry.names()),
            {
                "am",
                "gat",
                "geometric",
                "geometric_transformer",
                "gpn",
                "gru",
                "moe_transformer",
                "pointerformer",
                "ptrnet",
                "reversible_transformer",
                "sparse_moe_transformer",
                "transformer_ln",
            },
        )

    def test_build_model_uses_one_stable_config_interface(self) -> None:
        model = build_model(
            {
                "model": "am",
                "embedding_dim": 16,
                "heads": 4,
                "encoder_layers": 1,
                "feed_forward_dim": 32,
                "normalization": "layer",
            }
        )
        self.assertIsInstance(model, AdaptiveAttentionModel)

    def test_external_method_can_register_without_framework_changes(self) -> None:
        registry = ModelRegistry()
        registry.register("dummy", lambda _config: nn.Linear(2, 2))
        self.assertIsInstance(registry.build({"model": "dummy"}), nn.Linear)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("dummy", lambda _config: nn.Identity())


if __name__ == "__main__":
    unittest.main()
