"""One explicit extension point for model families used with GroupOpt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.models.encoder_controls import JointEncoderModel, ModernNativeAttentionModel
from groupopt.models.gpn import AdaptiveGraphPointerNetwork
from groupopt.models.ptrnet import AdaptivePointerNetwork

ModelBuilder = Callable[[Mapping[str, Any]], nn.Module]


class ModelRegistry:
    """Register and build neural methods without changing framework code."""

    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if not name or not name.isidentifier():
            raise ValueError("model name must be a non-empty Python identifier")
        if name in self._builders:
            raise ValueError(f"model is already registered: {name}")
        self._builders[name] = builder

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        name = str(config.get("model", "am"))
        try:
            builder = self._builders[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model {name!r}; available models: {', '.join(self.names())}"
            ) from error
        return builder(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def _am(config: Mapping[str, Any]) -> nn.Module:
    return AdaptiveAttentionModel(
        embedding_dim=int(config["embedding_dim"]),
        n_heads=int(config["heads"]),
        n_encoder_layers=int(config["encoder_layers"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        normalization=str(config["normalization"]),
    )


def _transformer_ln(config: Mapping[str, Any]) -> nn.Module:
    return AdaptiveAttentionModel(
        embedding_dim=int(config["embedding_dim"]),
        n_heads=int(config["heads"]),
        n_encoder_layers=int(config["encoder_layers"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        normalization="layer",
    )


def _ptrnet(config: Mapping[str, Any]) -> nn.Module:
    return AdaptivePointerNetwork(
        embedding_dim=int(config["embedding_dim"]),
        n_encoder_layers=int(config["encoder_layers"]),
    )


def _gpn(config: Mapping[str, Any]) -> nn.Module:
    return AdaptiveGraphPointerNetwork(
        embedding_dim=int(config["embedding_dim"]),
        n_encoder_layers=int(config["encoder_layers"]),
    )


def _joint_encoder(model_name: str) -> ModelBuilder:
    def builder(config: Mapping[str, Any]) -> nn.Module:
        return JointEncoderModel(
            encoder_type=model_name,
            embedding_dim=int(config["embedding_dim"]),
            n_encoder_layers=int(config["encoder_layers"]),
            n_heads=int(config["heads"]),
        )

    return builder


def _modern_native(model_name: str) -> ModelBuilder:
    def builder(config: Mapping[str, Any]) -> nn.Module:
        return ModernNativeAttentionModel(
            model_name,
            embedding_dim=int(config["embedding_dim"]),
            n_encoder_layers=int(config["encoder_layers"]),
            n_heads=int(config["heads"]),
            feed_forward_dim=int(config["feed_forward_dim"]),
        )

    return builder


model_registry = ModelRegistry()
model_registry.register("am", _am)
model_registry.register("transformer_ln", _transformer_ln)
model_registry.register("ptrnet", _ptrnet)
model_registry.register("gpn", _gpn)
for _name in ("gat", "gru", "pointerformer", "geometric", "moe_transformer"):
    model_registry.register(_name, _joint_encoder(_name))
for _name in (
    "reversible_transformer",
    "geometric_transformer",
    "sparse_moe_transformer",
):
    model_registry.register(_name, _modern_native(_name))


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build one registered neural adapter from an experiment config."""
    return model_registry.build(config)
