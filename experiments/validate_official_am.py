"""验证 Original 分支与官方 AM 前向结果完全一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.official_am import OfficialAttentionModelGroupOpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-am-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260823)
    model = OfficialAttentionModelGroupOpt(
        official_root=args.official_am_root,
        embedding_dim=32,
        n_heads=4,
        n_encoder_layers=2,
    ).to(device)
    model.eval()
    coordinates = torch.rand(8, 20, 2, device=device)

    model.native.set_decode_type("greedy")
    with torch.no_grad():
        native_cost, native_ll, native_tour = model.native(coordinates, return_pi=True)
        wrapped = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_original",
        )

    exact_tour = torch.equal(native_tour, wrapped.tails)
    exact_cost = torch.equal(native_cost, wrapped.cost)
    exact_log_likelihood = torch.equal(native_ll, wrapped.log_likelihood)

    # Conditional Fixed 应等价于官方 decoder 在外部强制首节点后的 rollout。
    anchor = 3
    model.native.set_decode_type("greedy", temp=1.0)
    with torch.no_grad():
        anchored = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_conditional_fixed",
            anchor=anchor,
            temperature=1.0,
        )
        embeddings, _ = model.native.embedder(model.native._init_embed(coordinates))
        native_fixed = model.native._precompute(embeddings)
        native_state = model.native.problem.make_state(coordinates)
        forced_first = torch.full(
            (coordinates.size(0),), anchor, dtype=torch.long, device=device
        )
        native_state = native_state.update(forced_first)
        forced_tour = [forced_first]
        forced_log_probabilities = []
        while not native_state.all_finished():
            native_log_p, native_mask = model.native._get_log_p(native_fixed, native_state)
            selected = native_log_p[:, 0, :].argmax(dim=-1)
            if native_mask[:, 0, :].gather(1, selected[:, None]).any():
                raise RuntimeError("forced native rollout selected a masked node")
            forced_log_probabilities.append(
                native_log_p[:, 0, :].gather(1, selected[:, None]).squeeze(1)
            )
            forced_tour.append(selected)
            native_state = native_state.update(selected)
        forced_tour_tensor = torch.stack(forced_tour, dim=1)
        forced_log_likelihood = torch.stack(forced_log_probabilities, dim=1).sum(dim=1)
        forced_cost = native_state.get_final_cost().squeeze(-1)

    anchored_exact_tour = torch.equal(anchored.tails, forced_tour_tensor)
    anchored_exact_log_likelihood = torch.equal(
        anchored.log_likelihood, forced_log_likelihood
    )
    anchored_cost_close = torch.allclose(
        anchored.cost, forced_cost, atol=1e-6, rtol=1e-6
    )

    model.zero_grad(set_to_none=True)
    training_output = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_original",
    )
    (-training_output.log_likelihood.mean()).backward()
    groupopt_inactive = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    model.zero_grad(set_to_none=True)
    fixed = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_conditional_fixed",
        anchor=0,
        generator=torch.Generator(device=device).manual_seed(20260824),
    )
    (-fixed.log_likelihood.mean()).backward()
    fixed_tail_chain = bool(
        torch.all(fixed.tails[:, 0] == 0) and torch.equal(fixed.tails[:, 1:], fixed.heads[:, :-1])
    )
    fixed_successor_is_permutation = bool(
        torch.equal(
            fixed.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(fixed.successor),
        )
    )
    groupopt_inactive_in_fixed = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    model.zero_grad(set_to_none=True)
    forest_fixed = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_forest_fixed",
        generator=torch.Generator(device=device).manual_seed(20260826),
    )
    (-forest_fixed.log_likelihood.mean()).backward()
    forest_state = model.process.initial_state(coordinates)
    forest_fixed_schedule_exact = True
    for step in range(coordinates.size(1)):
        valid_tail = ~model.process.action_mask(forest_state).all(dim=-1)
        expected_tail = valid_tail.to(torch.long).argmax(dim=-1)
        forest_fixed_schedule_exact = forest_fixed_schedule_exact and torch.equal(
            forest_fixed.tails[:, step], expected_tail
        )
        forest_state = model.process.transition(
            forest_state,
            forest_fixed.tails[:, step],
            forest_fixed.heads[:, step],
        )
    forest_successor_is_permutation = bool(
        torch.equal(
            forest_fixed.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(forest_fixed.successor),
        )
    )
    groupopt_inactive_in_forest_fixed = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    model.zero_grad(set_to_none=True)
    free = model(
        coordinates,
        decode_type="sampling",
        base_mode="native_conditional_free",
        generator=torch.Generator(device=device).manual_seed(20260825),
    )
    (-free.log_likelihood.mean()).backward()
    groupopt_active_in_free = any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    context_variants_valid = True
    for context_mode in (
        "official_groupopt_global_anchor",
        "official_groupopt_graph_tail",
        "official_groupopt_forest_context",
        "official_groupopt_amstyle_interface",
        "official_amstyle_fixed",
        "official_amstyle_free",
        "official_hybrid_fixed",
        "official_hybrid_free",
        "official_groupopt_edge_native",
        "official_groupopt_edge_native_detached",
        "official_edge_native_fixed",
        "official_edge_native_free",
        "official_capacity_single_chain",
    ):
        with torch.no_grad():
            variant = model(
                coordinates,
                decode_type="greedy",
                base_mode=context_mode,
                anchor=0,
            )
        context_variants_valid = context_variants_valid and bool(
            torch.isfinite(variant.cost).all()
            and torch.equal(
                variant.successor.sort(dim=1).values,
                torch.arange(coordinates.size(1), device=device)
                .unsqueeze(0)
                .expand_as(variant.successor),
            )
        )

    # Forest State Adapter 的双端路径特征必须准确描述当前分量。
    feature_state = model.process.initial_state(coordinates)
    first_tail = torch.zeros(coordinates.size(0), dtype=torch.long, device=device)
    first_head = torch.ones(coordinates.size(0), dtype=torch.long, device=device)
    feature_state = model.process.transition(feature_state, first_tail, first_head)
    with torch.no_grad():
        feature_embeddings, _ = model.native.embedder(model.native._init_embed(coordinates))
        forest_features = feature_state.forest_decoder_features(feature_embeddings)
    feature_dim = feature_embeddings.size(-1)
    expected_mean = (feature_embeddings[:, 0] + feature_embeddings[:, 1]) / 2
    forest_features_exact = bool(
        torch.allclose(forest_features[:, 0, :feature_dim], expected_mean)
        and torch.allclose(
            forest_features[:, 0, feature_dim : 2 * feature_dim],
            feature_embeddings[:, 0],
        )
        and torch.allclose(
            forest_features[:, 0, 2 * feature_dim : 3 * feature_dim],
            feature_embeddings[:, 1],
        )
        and torch.allclose(
            forest_features[:, 0, -1],
            coordinates.new_full((coordinates.size(0),), 2 / coordinates.size(1)),
        )
    )

    with torch.no_grad():
        graph_tail_zero = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_groupopt_graph_tail",
        )
        forest_context_zero = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_groupopt_forest_context",
        )
    forest_zero_init_matches_graph_tail = bool(
        torch.equal(graph_tail_zero.tails, forest_context_zero.tails)
        and torch.equal(graph_tail_zero.heads, forest_context_zero.heads)
        and torch.allclose(
            graph_tail_zero.log_likelihood,
            forest_context_zero.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
        and torch.allclose(
            graph_tail_zero.cost, forest_context_zero.cost, atol=1e-6, rtol=1e-6
        )
    )

    model.zero_grad(set_to_none=True)
    forest_context = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_groupopt_forest_context",
        generator=torch.Generator(device=device).manual_seed(20260827),
    )
    (-forest_context.log_likelihood.mean()).backward()
    forest_adapter_active = all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name.startswith("project_decoder_")
    )

    model.zero_grad(set_to_none=True)
    amstyle_interface = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_groupopt_amstyle_interface",
        generator=torch.Generator(device=device).manual_seed(20260828),
    )
    (-amstyle_interface.log_likelihood.mean()).backward()
    amstyle_interface_active = all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name.startswith("project_amstyle_")
    )

    model.zero_grad(set_to_none=True)
    amstyle_fixed = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_amstyle_fixed",
        generator=torch.Generator(device=device).manual_seed(20260831),
    )
    (-amstyle_fixed.log_likelihood.mean()).backward()
    amstyle_fixed_head_active = all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name.startswith("project_amstyle_")
    )
    amstyle_fixed_tail_inactive = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "project_tail_state.",
                "project_head_summary.",
                "project_tail_nodes.",
                "project_tail_graph.",
                "project_tail_step.",
                "project_tail_out.",
            )
        )
    )
    amstyle_fixed_tail_chain = bool(
        torch.equal(amstyle_fixed.tails[:, 1:], amstyle_fixed.heads[:, :-1])
    )

    with torch.no_grad():
        edge_native_zero = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_groupopt_edge_native",
        )
    edge_zero_init_matches_graph_tail = bool(
        torch.equal(graph_tail_zero.tails, edge_native_zero.tails)
        and torch.equal(graph_tail_zero.heads, edge_native_zero.heads)
        and torch.allclose(
            graph_tail_zero.log_likelihood,
            edge_native_zero.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
        and torch.allclose(
            graph_tail_zero.cost, edge_native_zero.cost, atol=1e-6, rtol=1e-6
        )
    )

    model.zero_grad(set_to_none=True)
    edge_native = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_groupopt_edge_native",
        generator=torch.Generator(device=device).manual_seed(20260829),
    )
    (-edge_native.log_likelihood.mean()).backward()
    edge_native_active = all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name in {
            "project_edge_query.weight",
            "project_edge_head.weight",
            "project_edge_tail_state.weight",
            "project_edge_pair.2.weight",
        }
    )

    # Edge-native 的嵌套 Fixed/Free 必须共享同一 head 接口；Fixed 中只有
    # Tail Selector 不参与梯度，且锚定调度始终保持一条连续链。
    model.zero_grad(set_to_none=True)
    edge_nested_fixed = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_edge_native_fixed",
        generator=torch.Generator(device=device).manual_seed(20260831),
    )
    (-edge_nested_fixed.log_likelihood.mean()).backward()
    tail_parameter_prefixes = (
        "project_tail_state.",
        "project_head_summary.",
        "project_tail_nodes.",
        "project_tail_graph.",
        "project_tail_step.",
        "project_tail_out.",
        "project_edge_tail_state.",
    )
    edge_fixed_tail_inactive = all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name.startswith(tail_parameter_prefixes)
    )
    edge_fixed_tail_chain = bool(
        torch.equal(
            edge_nested_fixed.tails[:, 1:], edge_nested_fixed.heads[:, :-1]
        )
    )
    with torch.no_grad():
        edge_nested_free = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_edge_native_free",
        )
    edge_free_successor_is_permutation = bool(
        torch.equal(
            edge_nested_free.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(edge_nested_free.successor),
        )
    )

    edge_parameter_prefixes = (
        "project_head_summary",
        "project_tail_nodes",
        "project_tail_graph",
        "project_tail_step",
        "project_tail_out",
        "project_edge_",
    )
    edge_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name == "first_tail_context" or name.startswith(edge_parameter_prefixes)
    )
    capacity_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name == "capacity_extra" or name.startswith("project_capacity_")
    )
    capacity_parameter_count_matches_edge = (
        capacity_parameter_count == edge_parameter_count
    )

    with torch.no_grad():
        capacity_zero = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_capacity_single_chain",
            anchor=0,
        )
    capacity_zero_init_matches_original = bool(
        torch.equal(capacity_zero.tails, wrapped.tails)
        and torch.equal(capacity_zero.heads, wrapped.heads)
        and torch.allclose(
            capacity_zero.log_likelihood,
            wrapped.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
        and torch.allclose(
            capacity_zero.cost, wrapped.cost, atol=1e-6, rtol=1e-6
        )
    )

    model.zero_grad(set_to_none=True)
    capacity_control = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_capacity_single_chain",
        anchor=0,
        generator=torch.Generator(device=device).manual_seed(20260830),
    )
    (-capacity_control.log_likelihood.mean()).backward()
    capacity_adapter_active = all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name in {"capacity_extra", "project_capacity_out.weight"}
    )
    capacity_tail_chain = bool(
        torch.equal(capacity_control.tails[:, 1:], capacity_control.heads[:, :-1])
    )

    # 嵌套 Fixed 必须与容量匹配单链逐动作一致；否则 Fixed/Free 仍混入了
    # head decoder 或状态接口差异，不能用于归因 GroupOpt 的调度收益。
    with torch.no_grad():
        nested_fixed = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_nested_fixed",
        )
        nested_free = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_nested_free",
        )
    nested_fixed_matches_capacity = bool(
        torch.equal(nested_fixed.tails, capacity_zero.tails)
        and torch.equal(nested_fixed.heads, capacity_zero.heads)
        and torch.allclose(
            nested_fixed.log_likelihood,
            capacity_zero.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
        and torch.allclose(
            nested_fixed.cost, capacity_zero.cost, atol=1e-6, rtol=1e-6
        )
    )
    nested_free_successor_is_permutation = bool(
        torch.equal(
            nested_free.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(nested_free.successor),
        )
    )

    # Hybrid residual 从零开始时，Fixed 必须逐动作退化为未经修改的官方 AM。
    with torch.no_grad():
        hybrid_fixed = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_hybrid_fixed",
        )
        hybrid_free = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_hybrid_free",
        )
    hybrid_fixed_matches_original = bool(
        torch.equal(hybrid_fixed.tails, wrapped.tails)
        and torch.equal(hybrid_fixed.heads, wrapped.heads)
        and torch.allclose(
            hybrid_fixed.log_likelihood,
            wrapped.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
        and torch.allclose(hybrid_fixed.cost, wrapped.cost, atol=1e-6, rtol=1e-6)
    )
    hybrid_zero_kl = bool(
        hybrid_fixed.interface_regularization is not None
        and torch.allclose(
            hybrid_fixed.interface_regularization,
            torch.zeros_like(hybrid_fixed.interface_regularization),
            atol=1e-7,
            rtol=0.0,
        )
    )
    hybrid_free_successor_is_permutation = bool(
        torch.equal(
            hybrid_free.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(hybrid_free.successor),
        )
    )
    model.zero_grad(set_to_none=True)
    hybrid_sample = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_hybrid_free",
        generator=torch.Generator(device=device).manual_seed(20260901),
    )
    hybrid_loss = -hybrid_sample.log_likelihood.mean()
    if hybrid_sample.interface_regularization is not None:
        hybrid_loss = hybrid_loss + 0.1 * hybrid_sample.interface_regularization.mean()
    hybrid_loss.backward()
    hybrid_residual_active = bool(
        model.project_hybrid_out.weight.grad is not None
        and torch.count_nonzero(model.project_hybrid_out.weight.grad) > 0
    )
    with torch.no_grad():
        efficient_embeddings, _ = model.native.embedder(
            model.native._init_embed(coordinates)
        )
        efficient_fixed = model.native._precompute(
            efficient_embeddings, num_steps=1
        )
        efficient_query = (
            efficient_fixed.context_node_projected
            + model.native.project_step_context(
                torch.cat((efficient_embeddings, efficient_embeddings), dim=-1)
            )
        )
        efficient_mask = torch.eye(
            coordinates.size(1), dtype=torch.bool, device=device
        ).unsqueeze(0).expand(coordinates.size(0), -1, -1)
        native_logits, _ = model.native._one_to_many_logits(
            efficient_query,
            efficient_fixed.glimpse_key,
            efficient_fixed.glimpse_val,
            efficient_fixed.logit_key,
            efficient_mask,
        )
        efficient_logits = model._native_head_logits_efficient(
            efficient_query,
            efficient_fixed.glimpse_key,
            efficient_fixed.glimpse_val,
            efficient_fixed.logit_key,
            efficient_mask,
        )
    native_efficient_logits_equal = bool(
        torch.allclose(native_logits, efficient_logits, atol=1e-6, rtol=1e-6)
    )

    # 课程初始化在自由比例为 0 时必须退化为官方锚定单链；比例为 1 时
    # 必须恢复完整 Free，且 warmup 不应更新 Tail Selector。
    model.initialize_edge_native_single_chain()
    with torch.no_grad():
        curriculum_zero = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_edge_native_free",
            anchor=anchor,
            tail_free_probability=0.0,
        )
        curriculum_one = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_edge_native_free",
            anchor=anchor,
            tail_free_probability=1.0,
        )
    curriculum_zero_matches_forced_native = bool(
        torch.equal(curriculum_zero.tails, forced_tour_tensor)
        and torch.equal(curriculum_zero.heads, forced_tour_tensor.roll(-1, dims=1))
        and torch.allclose(
            curriculum_zero.cost, forced_cost, atol=1e-6, rtol=1e-6
        )
        and torch.count_nonzero(curriculum_zero.log_likelihood) == 0
        and torch.count_nonzero(curriculum_zero.gate_probability) == 0
    )
    curriculum_one_is_full_free = bool(
        torch.all(curriculum_one.gate_probability == 1)
        and torch.equal(
            curriculum_one.successor.sort(dim=1).values,
            torch.arange(coordinates.size(1), device=device)
            .unsqueeze(0)
            .expand_as(curriculum_one.successor),
        )
    )
    model.zero_grad(set_to_none=True)
    curriculum_warmup = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_edge_native_free",
        anchor=anchor,
        tail_free_probability=0.0,
        generator=torch.Generator(device=device).manual_seed(20260902),
    )
    (-curriculum_warmup.log_likelihood.mean()).backward()
    curriculum_zero_tail_inactive = all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameter in model.named_parameters()
        if name.startswith(tail_parameter_prefixes)
    )
    curriculum_zero_adapter_inactive = all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    result = {
        "exact_cost": exact_cost,
        "exact_log_likelihood": exact_log_likelihood,
        "exact_tour": exact_tour,
        "anchored_cost_close": anchored_cost_close,
        "anchored_exact_log_likelihood": anchored_exact_log_likelihood,
        "anchored_exact_tour": anchored_exact_tour,
        "context_variants_valid": context_variants_valid,
        "fixed_successor_is_permutation": fixed_successor_is_permutation,
        "fixed_tail_chain": fixed_tail_chain,
        "forest_fixed_schedule_exact": forest_fixed_schedule_exact,
        "forest_successor_is_permutation": forest_successor_is_permutation,
        "forest_adapter_active": forest_adapter_active,
        "forest_features_exact": forest_features_exact,
        "forest_zero_init_matches_graph_tail": forest_zero_init_matches_graph_tail,
        "amstyle_interface_active": amstyle_interface_active,
        "amstyle_fixed_head_active": amstyle_fixed_head_active,
        "amstyle_fixed_tail_inactive": amstyle_fixed_tail_inactive,
        "amstyle_fixed_tail_chain": amstyle_fixed_tail_chain,
        "edge_native_active": edge_native_active,
        "edge_fixed_tail_inactive": edge_fixed_tail_inactive,
        "edge_fixed_tail_chain": edge_fixed_tail_chain,
        "edge_free_successor_is_permutation": edge_free_successor_is_permutation,
        "edge_zero_init_matches_graph_tail": edge_zero_init_matches_graph_tail,
        "capacity_adapter_active": capacity_adapter_active,
        "capacity_parameter_count_matches_edge": capacity_parameter_count_matches_edge,
        "capacity_zero_init_matches_original": capacity_zero_init_matches_original,
        "capacity_tail_chain": capacity_tail_chain,
        "nested_fixed_matches_capacity": nested_fixed_matches_capacity,
        "nested_free_successor_is_permutation": nested_free_successor_is_permutation,
        "hybrid_fixed_matches_original": hybrid_fixed_matches_original,
        "hybrid_free_successor_is_permutation": hybrid_free_successor_is_permutation,
        "hybrid_residual_active": hybrid_residual_active,
        "hybrid_zero_kl": hybrid_zero_kl,
        "native_efficient_logits_equal": native_efficient_logits_equal,
        "curriculum_zero_matches_forced_native": (
            curriculum_zero_matches_forced_native
        ),
        "curriculum_one_is_full_free": curriculum_one_is_full_free,
        "curriculum_zero_tail_inactive": curriculum_zero_tail_inactive,
        "curriculum_zero_adapter_inactive": curriculum_zero_adapter_inactive,
        "groupopt_active_in_free": groupopt_active_in_free,
        "groupopt_parameters_inactive_in_forest_fixed": (groupopt_inactive_in_forest_fixed),
        "groupopt_parameters_inactive_in_fixed": groupopt_inactive_in_fixed,
        "groupopt_parameters_inactive_in_original": groupopt_inactive,
        "official_am_root": str(Path(args.official_am_root).resolve()),
    }
    if not all(
        (
            exact_tour,
            exact_cost,
            exact_log_likelihood,
            groupopt_inactive,
            anchored_exact_tour,
            anchored_exact_log_likelihood,
            anchored_cost_close,
            context_variants_valid,
            fixed_tail_chain,
            fixed_successor_is_permutation,
            groupopt_inactive_in_fixed,
            forest_fixed_schedule_exact,
            forest_successor_is_permutation,
            groupopt_inactive_in_forest_fixed,
            groupopt_active_in_free,
            forest_adapter_active,
            forest_features_exact,
            forest_zero_init_matches_graph_tail,
            amstyle_interface_active,
            amstyle_fixed_head_active,
            amstyle_fixed_tail_inactive,
            amstyle_fixed_tail_chain,
            edge_native_active,
            edge_fixed_tail_inactive,
            edge_fixed_tail_chain,
            edge_free_successor_is_permutation,
            edge_zero_init_matches_graph_tail,
            capacity_adapter_active,
            capacity_parameter_count_matches_edge,
            capacity_zero_init_matches_original,
            capacity_tail_chain,
            nested_fixed_matches_capacity,
            nested_free_successor_is_permutation,
            hybrid_fixed_matches_original,
            hybrid_free_successor_is_permutation,
            hybrid_residual_active,
            hybrid_zero_kl,
            native_efficient_logits_equal,
            curriculum_zero_matches_forced_native,
            curriculum_one_is_full_free,
            curriculum_zero_tail_inactive,
            curriculum_zero_adapter_inactive,
        )
    ):
        raise RuntimeError(f"official AM equivalence failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
