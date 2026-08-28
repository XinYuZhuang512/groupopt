"""训练可断点续训的 Original 或 GroupOpt 模型。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from groupopt.models.am import AttentionModel
from groupopt.objectives import (
    SymmetryProjectionHead,
    augment_euclidean_symmetries,
    reinforce_loss,
    symnco_am_loss,
)

BASE_MODES = (
    "official_gpn_original",
    "official_gpn_fixed",
    "official_gpn_free",
    "native_original",
    "native_conditional_fixed",
    "native_conditional_free",
    "native_capacity_single_chain",
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "official_original",
    "official_conditional_fixed",
    "official_forest_fixed",
    "official_groupopt_global_anchor",
    "official_groupopt_graph_tail",
    "official_groupopt_forest_context",
    "official_groupopt_amstyle_interface",
    "official_amstyle_fixed",
    "official_amstyle_free",
    "official_amstyle_forest_fixed",
    "official_amstyle_free_no_head_summary",
    "official_amstyle_free_no_path_state",
    "official_amstyle_free_no_last_head",
    "official_groupopt_edge_native",
    "official_groupopt_edge_native_detached",
    "official_edge_native_fixed",
    "official_edge_native_free",
    "official_capacity_single_chain",
    "official_nested_fixed",
    "official_nested_free",
    "official_hybrid_fixed",
    "official_hybrid_free",
)


def evaluate(
    model: nn.Module,
    coordinates: torch.Tensor,
    base_mode: str,
) -> float:
    model.eval()
    with torch.no_grad():
        best_rollout_cost = getattr(model, "best_rollout_cost", None)
        if best_rollout_cost is not None:
            chunk_size = int(getattr(model, "evaluation_batch_size", 1))
            costs = []
            for start in range(0, coordinates.size(0), chunk_size):
                output = model(
                    coordinates[start : start + chunk_size],
                    decode_type="greedy",
                    base_mode=base_mode,
                )
                costs.append(best_rollout_cost(output))
            return torch.cat(costs).mean().item()
        output = model(coordinates, decode_type="greedy", base_mode=base_mode)
    return output.cost.mean().item()


def build_am_model(args: argparse.Namespace) -> nn.Module:
    return AttentionModel(
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        normalization=args.normalization,
    )


def _tail_free_probability(args: argparse.Namespace, step: int) -> float:
    """返回当前训练 step 中允许自由选择 tail 的样本比例。"""
    ramp_steps = int(args.tail_curriculum_ramp_steps)
    if ramp_steps <= 0:
        return 1.0
    warmup_steps = int(args.tail_curriculum_warmup_steps)
    if step <= warmup_steps:
        return 0.0
    return min(1.0, (step - warmup_steps) / ramp_steps)


def run(
    args: argparse.Namespace,
    model_builder: Callable[[argparse.Namespace], nn.Module] = build_am_model,
) -> None:
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    config = _experiment_config(args, device)
    _write_or_validate_config(output_dir / "config.json", config, args.resume is not None)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = model_builder(args).to(device)
    if args.initialize_model_from is not None:
        source_checkpoint = torch.load(
            args.initialize_model_from, map_location=device, weights_only=False
        )
        model.load_state_dict(source_checkpoint["model"], strict=True)
    if args.initialize_native_from is not None:
        source_checkpoint = torch.load(
            args.initialize_native_from, map_location=device, weights_only=False
        )
        native_state = {
            name.removeprefix("native."): value
            for name, value in source_checkpoint["model"].items()
            if name.startswith("native.")
        }
        if not native_state or not hasattr(model, "native"):
            raise RuntimeError(
                "--initialize-native-from requires a checkpoint and model with native.* parameters"
            )
        model.native.load_state_dict(native_state, strict=True)
        refresh_interface = getattr(model, "_initialize_edge_native_from_official", None)
        if refresh_interface is not None:
            refresh_interface()
        if args.edge_native_single_chain_init:
            exact_initializer = getattr(model, "initialize_edge_native_single_chain", None)
            if exact_initializer is None:
                raise RuntimeError(
                    "--edge-native-single-chain-init requires the official AM adapter"
                )
            exact_initializer()
    if args.freeze_native:
        if not hasattr(model, "native"):
            raise RuntimeError("--freeze-native requires a model with a native host")
        model.native.requires_grad_(False)
    native_project_out_parameters: list[nn.Parameter] = []
    if args.unfreeze_native_project_out:
        if not hasattr(model, "native") or not hasattr(model.native, "project_out"):
            raise RuntimeError("--unfreeze-native-project-out requires the official AM adapter")
        model.native.project_out.requires_grad_(True)
        native_project_out_parameters = list(model.native.project_out.parameters())
    if args.train_groupopt_tail_only:
        # 嵌套对照必须冻结共同 encoder/head/Adapter，只允许 Tail Selector 更新。
        model.requires_grad_(False)
        tail_parameter_prefixes = (
            "project_tail_state.",
            "project_head_summary.",
            "project_tail_nodes.",
            "project_tail_graph.",
            "project_tail_step.",
            "project_tail_out.",
        )
        for name, parameter in model.named_parameters():
            if name.startswith(tail_parameter_prefixes):
                parameter.requires_grad_(True)
    symmetry_projection_head: nn.Module | None = None
    native_project_out_ids = {id(parameter) for parameter in native_project_out_parameters}
    optimizer_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in native_project_out_ids
    ]
    if args.training_scheme == "symnco_am":
        symmetry_projection_head = SymmetryProjectionHead(args.embedding_dim).to(device)
        optimizer_parameters.extend(symmetry_projection_head.parameters())
    optimizer_groups: list[dict[str, Any]] = [
        {"params": optimizer_parameters, "lr": args.learning_rate}
    ]
    if native_project_out_parameters:
        optimizer_groups.append(
            {
                "params": native_project_out_parameters,
                "lr": args.native_project_out_learning_rate,
            }
        )
    optimizer = torch.optim.Adam(optimizer_groups)

    data_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    action_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    symmetry_generator = torch.Generator(device=device).manual_seed(args.seed + 4)
    validation = torch.rand(
        args.validation_size,
        args.graph_size,
        2,
        device=device,
        generator=validation_generator,
    )

    start_step = 0
    best_cost = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if symmetry_projection_head is not None:
            if "symmetry_projection_head" not in checkpoint:
                raise RuntimeError("SYM-NCO checkpoint is missing its projection head")
            symmetry_projection_head.load_state_dict(checkpoint["symmetry_projection_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_cost = float(checkpoint["best_cost"])
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
        action_generator.set_state(checkpoint["action_generator_state"].cpu())
        if "symmetry_generator_state" in checkpoint:
            symmetry_generator.set_state(checkpoint["symmetry_generator_state"].cpu())

    metrics_path = output_dir / "metrics.jsonl"
    started_at = time.monotonic()
    if start_step == 0:
        initial_cost = evaluate(model, validation, args.base_mode)
        _append_metric(
            metrics_path,
            {
                "step": 0,
                "validation_greedy_cost": initial_cost,
                "elapsed_seconds": 0.0,
            },
        )
        best_cost = initial_cost
        initial_payload = _checkpoint_payload(
            model,
            optimizer,
            0,
            best_cost,
            config,
            data_generator,
            action_generator,
            symmetry_generator,
            symmetry_projection_head,
        )
        _atomic_torch_save(initial_payload, checkpoint_dir / "best.pt")
        print(f"step=000000 val_greedy_cost={initial_cost:.6f}", flush=True)

    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            if args.freeze_native or args.train_groupopt_tail_only:
                # 冻结不仅包括权重，也包括官方 BatchNorm 的运行统计量。
                model.native.eval()
            coordinates = torch.rand(
                args.batch_size,
                args.graph_size,
                2,
                device=device,
                generator=data_generator,
            )
            if args.training_scheme == "symnco_am":
                model_coordinates = augment_euclidean_symmetries(
                    coordinates,
                    args.symmetry_factor,
                    symmetry_generator,
                )
            else:
                model_coordinates = coordinates
            forward_options: dict[str, Any] = {}
            if args.training_scheme == "symnco_am":
                forward_options["return_symmetry_embeddings"] = True
            tail_free_probability = _tail_free_probability(args, step)
            if args.tail_curriculum_ramp_steps > 0:
                forward_options["tail_free_probability"] = tail_free_probability
            output = model(
                model_coordinates,
                decode_type="sampling",
                base_mode=args.base_mode,
                generator=action_generator,
                **forward_options,
            )
            symmetry_terms = None
            if args.training_scheme == "symnco_am":
                if output.symmetry_node_embeddings is None:
                    raise RuntimeError("AM did not return projected embeddings for SYM-NCO")
                assert symmetry_projection_head is not None
                projected_embeddings = symmetry_projection_head(output.symmetry_node_embeddings)
                symmetry_terms = symnco_am_loss(
                    output.cost,
                    output.log_likelihood,
                    projected_embeddings,
                    args.symmetry_factor,
                    args.symmetry_alpha,
                )
                loss = symmetry_terms.total
                policy_loss = symmetry_terms.policy
            elif args.training_scheme == "gpn_self_critic":
                with torch.no_grad():
                    baseline_output = model(
                        coordinates,
                        decode_type="greedy",
                        base_mode=args.base_mode,
                        temperature=1.0,
                    )
                advantage = output.cost - baseline_output.cost
                advantage = (advantage - advantage.mean()).detach()
                loss = (advantage * output.log_likelihood).mean()
                policy_loss = loss
            elif args.training_scheme == "pomo":
                pomo_policy_loss = getattr(model, "pomo_policy_loss", None)
                if pomo_policy_loss is None:
                    raise RuntimeError("pomo training requires a POMO-compatible model")
                loss = pomo_policy_loss(output)
                policy_loss = loss
            else:
                loss = reinforce_loss(output.cost, output.log_likelihood)
                policy_loss = loss
            tail_entropy_bonus = args.tail_entropy_coefficient * output.tail_entropy.mean()
            loss = loss - tail_entropy_bonus
            interface_regularization = (
                output.interface_regularization.mean()
                if output.interface_regularization is not None
                else output.cost.new_zeros(())
            )
            loss = loss + args.interface_kl_coefficient * interface_regularization

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            native_project_out_gradient_norm = 0.0
            if native_project_out_parameters:
                squared_norm = output.cost.new_zeros(())
                for parameter in native_project_out_parameters:
                    if parameter.grad is not None:
                        squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
                native_project_out_gradient_norm = float(squared_norm.sqrt())
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            should_evaluate = step % args.eval_every == 0 or step == args.steps
            validation_cost: float | None = None
            is_new_best = False
            if should_evaluate:
                validation_cost = evaluate(model, validation, args.base_mode)
                is_new_best = validation_cost < best_cost
                if is_new_best:
                    best_cost = validation_cost
                metric = {
                    "step": step,
                    "train_cost": output.cost.mean().item(),
                    "loss": loss.item(),
                    "reinforce_loss": policy_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "gradient_norm": float(gradient_norm),
                    "validation_greedy_cost": validation_cost,
                    "best_validation_greedy_cost": best_cost,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "peak_gpu_memory_gb": _peak_memory_gb(device),
                    "tail_entropy": output.tail_entropy.mean().item(),
                    "gate_probability": output.gate_probability.mean().item(),
                    "action_entropy": output.action_entropy.mean().item(),
                    "tail_entropy_bonus": tail_entropy_bonus.item(),
                    "interface_kl": interface_regularization.item(),
                    "interface_kl_penalty": (
                        args.interface_kl_coefficient * interface_regularization
                    ).item(),
                    "effective_batch_size": int(output.cost.shape[0]),
                    "tail_free_probability": tail_free_probability,
                }
                if symmetry_terms is not None:
                    metric.update(
                        {
                            "symmetry_similarity": symmetry_terms.similarity.item(),
                            "invariance_penalty": symmetry_terms.invariance_penalty.item(),
                        }
                    )
                if native_project_out_parameters:
                    metric["native_project_out_gradient_norm"] = native_project_out_gradient_norm
                _append_metric(metrics_path, metric)
                symmetry_log = (
                    f"symmetry_similarity={metric['symmetry_similarity']:.4f} "
                    if symmetry_terms is not None
                    else ""
                )
                print(
                    f"step={step:06d} "
                    f"train_cost={metric['train_cost']:.6f} "
                    f"loss={metric['loss']:.6f} "
                    f"val_greedy_cost={validation_cost:.6f} "
                    f"best={best_cost:.6f} "
                    f"tail_entropy={metric['tail_entropy']:.4f} "
                    f"gate={metric['gate_probability']:.4f} "
                    f"tail_free_p={metric['tail_free_probability']:.3f} "
                    f"action_entropy={metric['action_entropy']:.4f} "
                    f"interface_kl={metric['interface_kl']:.5f} "
                    f"{symmetry_log}"
                    f"peak_gb={metric['peak_gpu_memory_gb']:.3f}",
                    flush=True,
                )

                if is_new_best:
                    best_payload = _checkpoint_payload(
                        model,
                        optimizer,
                        step,
                        best_cost,
                        config,
                        data_generator,
                        action_generator,
                        symmetry_generator,
                        symmetry_projection_head,
                    )
                    _atomic_torch_save(best_payload, checkpoint_dir / "best.pt")

            should_checkpoint = step % args.checkpoint_every == 0 or step == args.steps
            if should_checkpoint:
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    best_cost,
                    config,
                    data_generator,
                    action_generator,
                    symmetry_generator,
                    symmetry_projection_head,
                )
                _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
                _atomic_torch_save(payload, checkpoint_dir / f"step-{step:06d}.pt")
    except KeyboardInterrupt:
        payload = _checkpoint_payload(
            model,
            optimizer,
            step,
            best_cost,
            config,
            data_generator,
            action_generator,
            symmetry_generator,
            symmetry_projection_head,
        )
        _atomic_torch_save(payload, checkpoint_dir / "interrupted.pt")
        _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        raise


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_cost: float,
    config: dict[str, Any],
    data_generator: torch.Generator,
    action_generator: torch.Generator,
    symmetry_generator: torch.Generator,
    symmetry_projection_head: nn.Module | None,
) -> dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_cost": best_cost,
        "config": config,
        "data_generator_state": data_generator.get_state(),
        "action_generator_state": action_generator.get_state(),
        "symmetry_generator_state": symmetry_generator.get_state(),
    }
    if symmetry_projection_head is not None:
        payload["symmetry_projection_head"] = symmetry_projection_head.state_dict()
    return payload


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def _write_or_validate_config(path: Path, config: dict[str, Any], resuming: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "steps"}
        comparable_config = {key: value for key, value in config.items() if key != "steps"}
        if comparable_existing != comparable_config:
            raise ValueError("experiment config differs from the existing output directory")
        if not resuming:
            raise FileExistsError("output directory already contains an experiment; use --resume")
        if int(config["steps"]) < int(existing["steps"]):
            raise ValueError("resume training horizon cannot be shorter than the existing one")
        if config != existing:
            path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return
    if resuming:
        raise ValueError("resume output directory does not contain config.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _experiment_config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    excluded = {"resume", "output_dir"}
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    if config.get("model") != "pomo":
        config.pop("qkv_dim", None)
        config.pop("pomo_size", None)
    if config.get("initialize_native_from") is None:
        config.pop("initialize_native_from", None)
    if config.get("initialize_model_from") is None:
        config.pop("initialize_model_from", None)
    if float(config.get("tail_entropy_coefficient", 0.0)) == 0.0:
        config.pop("tail_entropy_coefficient", None)
    if float(config.get("interface_kl_coefficient", 0.0)) == 0.0:
        config.pop("interface_kl_coefficient", None)
    if not bool(config.get("freeze_native", False)):
        config.pop("freeze_native", None)
    if not bool(config.get("train_groupopt_tail_only", False)):
        config.pop("train_groupopt_tail_only", None)
    if not bool(config.get("edge_native_single_chain_init", False)):
        config.pop("edge_native_single_chain_init", None)
    if not bool(config.get("unfreeze_native_project_out", False)):
        config.pop("unfreeze_native_project_out", None)
        config.pop("native_project_out_learning_rate", None)
    if int(config.get("tail_curriculum_warmup_steps", 0)) == 0:
        config.pop("tail_curriculum_warmup_steps", None)
    if int(config.get("tail_curriculum_ramp_steps", 0)) == 0:
        config.pop("tail_curriculum_ramp_steps", None)
    config["device"] = str(device)
    if args.training_scheme == "symnco_am":
        config["effective_batch_size"] = int(args.batch_size) * int(args.symmetry_factor)
    elif args.training_scheme == "reinforce":
        # 保留既有配置格式，使加入对称训练支持前中断的标准实验仍可续训。
        config.pop("training_scheme", None)
        config.pop("symmetry_factor", None)
        config.pop("symmetry_alpha", None)
    elif args.training_scheme == "pomo":
        config["effective_batch_size"] = int(args.batch_size) * int(args.pomo_size)
        config.pop("symmetry_factor", None)
        config.pop("symmetry_alpha", None)
    else:
        config.pop("symmetry_factor", None)
        config.pop("symmetry_alpha", None)
    return config


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.graph_size,
        args.steps,
        args.batch_size,
        args.validation_size,
        args.eval_every,
        args.checkpoint_every,
    )
    if min(positive) < 1:
        raise ValueError("sizes and step intervals must be positive")
    if args.base_mode not in BASE_MODES:
        raise ValueError(f"base_mode must be one of {BASE_MODES}")
    if args.training_scheme == "symnco_am" and args.symmetry_factor < 2:
        raise ValueError("symnco_am requires --symmetry-factor of at least two")
    if args.training_scheme == "symnco_am" and getattr(args, "model", "am") != "am":
        raise ValueError("symnco_am is currently implemented only for the AM comparison")
    if args.training_scheme == "gpn_self_critic" and getattr(args, "model", None) != "official_gpn":
        raise ValueError("gpn_self_critic is defined only for official GPN")
    if args.training_scheme == "pomo" and getattr(args, "model", None) != "pomo":
        raise ValueError("pomo training is defined only for the POMO model")
    if args.pomo_size < 1 or args.pomo_size > args.graph_size:
        raise ValueError("pomo_size must be in [1, graph_size]")
    if args.symmetry_alpha < 0:
        raise ValueError("symmetry alpha must be non-negative")
    if args.tail_entropy_coefficient < 0:
        raise ValueError("tail entropy coefficient must be non-negative")
    if args.interface_kl_coefficient < 0:
        raise ValueError("interface KL coefficient must be non-negative")
    if args.tail_curriculum_warmup_steps < 0 or args.tail_curriculum_ramp_steps < 0:
        raise ValueError("tail curriculum step counts must be non-negative")
    if args.native_project_out_learning_rate <= 0:
        raise ValueError("native project-out learning rate must be positive")
    curriculum_requested = args.tail_curriculum_ramp_steps > 0
    edge_free_modes = {
        "official_groupopt_edge_native_detached",
        "official_edge_native_free",
    }
    if curriculum_requested and args.base_mode not in edge_free_modes:
        raise ValueError("tail curriculum is defined only for edge-native Free")
    if args.edge_native_single_chain_init:
        if args.base_mode not in edge_free_modes:
            raise ValueError(
                "single-chain edge initialization is defined only for edge-native Free"
            )
        if args.initialize_native_from is None:
            raise ValueError("--edge-native-single-chain-init requires --initialize-native-from")
    initialization_count = sum(
        option is not None for option in (args.initialize_native_from, args.initialize_model_from)
    )
    if initialization_count > 1:
        raise ValueError("choose only one model initialization checkpoint")
    if args.resume is not None and args.initialize_model_from is not None:
        raise ValueError("--resume cannot be combined with --initialize-model-from")
    if args.freeze_native and initialization_count == 0:
        raise ValueError("--freeze-native requires a model initialization checkpoint")
    if args.unfreeze_native_project_out:
        if not args.freeze_native:
            raise ValueError("--unfreeze-native-project-out requires --freeze-native")
        if getattr(args, "model", None) != "official_am":
            raise ValueError("selective native project-out tuning is defined only for official AM")
    if args.train_groupopt_tail_only:
        if args.initialize_model_from is None:
            raise ValueError("--train-groupopt-tail-only requires --initialize-model-from")
        if args.base_mode != "official_nested_free":
            raise ValueError("tail-only training is defined only for official_nested_free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-mode",
        choices=BASE_MODES,
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-native-from",
        help="从官方宿主 checkpoint 载入 native.* 参数，但重新开始优化",
    )
    parser.add_argument(
        "--initialize-model-from",
        help="载入完整模型 checkpoint，但以新的优化器从 step 0 开始训练",
    )
    parser.add_argument(
        "--freeze-native",
        action="store_true",
        help="冻结已载入的官方宿主，只训练 GroupOpt 接口参数",
    )
    parser.add_argument(
        "--train-groupopt-tail-only",
        action="store_true",
        help="冻结共享 encoder/head/Adapter，只训练 GroupOpt Tail Selector",
    )
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--training-scheme",
        choices=("reinforce", "symnco_am", "gpn_self_critic", "pomo"),
        default="reinforce",
    )
    parser.add_argument("--symmetry-factor", type=int, default=4)
    parser.add_argument("--symmetry-alpha", type=float, default=0.1)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--feed-forward-dim", type=int, default=512)
    parser.add_argument("--qkv-dim", type=int, default=16)
    parser.add_argument("--pomo-size", type=int, default=8)
    parser.add_argument("--normalization", choices=("batch", "layer"), default="batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tail-entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--interface-kl-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--edge-native-single-chain-init",
        action="store_true",
        help="将 edge-native head 接口初始化为官方锚定单链 decoder",
    )
    parser.add_argument(
        "--unfreeze-native-project-out",
        action="store_true",
        help="在其余宿主冻结时，只解冻官方 decoder 的 project_out",
    )
    parser.add_argument(
        "--native-project-out-learning-rate",
        type=float,
        default=1e-5,
        help="官方 decoder project_out 的独立小学习率",
    )
    parser.add_argument(
        "--tail-curriculum-warmup-steps",
        type=int,
        default=0,
        help="课程训练中保持锚定单链的 step 数",
    )
    parser.add_argument(
        "--tail-curriculum-ramp-steps",
        type=int,
        default=0,
        help="从锚定单链线性过渡到完全 Free 的 step 数；0 表示禁用",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
