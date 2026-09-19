"""验证 head summary 梯度实验只改变反向传播路径，不改变前向计算。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from groupopt.models.am import AttentionModel


def main(output: Path, device_name: str) -> None:
    device = torch.device(device_name)
    torch.manual_seed(1901)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1901)
    detached = AttentionModel(
        embedding_dim=128,
        n_heads=8,
        n_encoder_layers=3,
        feed_forward_dim=512,
        normalization="batch",
    ).to(device)
    end_to_end = copy.deepcopy(detached)
    coordinates = torch.rand(
        8,
        20,
        2,
        device=device,
        generator=torch.Generator(device=device).manual_seed(1902),
    )
    detached_output = detached(
        coordinates,
        decode_type="greedy",
        base_mode="native_conditional_free",
        generator=torch.Generator(device=device).manual_seed(1903),
    )
    end_to_end_output = end_to_end(
        coordinates,
        decode_type="greedy",
        base_mode="native_free_end_to_end_summary",
        generator=torch.Generator(device=device).manual_seed(1903),
    )
    same_actions = bool(
        torch.equal(detached_output.tails, end_to_end_output.tails)
        and torch.equal(detached_output.heads, end_to_end_output.heads)
        and torch.equal(detached_output.successor, end_to_end_output.successor)
    )
    maximum_cost_difference = float(
        (detached_output.cost - end_to_end_output.cost).detach().abs().max()
    )
    maximum_log_likelihood_difference = float(
        (
            detached_output.log_likelihood - end_to_end_output.log_likelihood
        ).detach().abs().max()
    )
    same_forward = bool(
        same_actions
        and torch.allclose(
            detached_output.cost, end_to_end_output.cost, atol=1e-6, rtol=1e-6
        )
        and torch.allclose(
            detached_output.log_likelihood,
            end_to_end_output.log_likelihood,
            atol=1e-5,
            rtol=1e-6,
        )
    )
    detached_loss = (
        detached_output.cost.detach() * detached_output.log_likelihood
    ).mean()
    end_to_end_loss = (
        end_to_end_output.cost.detach() * end_to_end_output.log_likelihood
    ).mean()
    detached_loss.backward()
    end_to_end_loss.backward()

    parameter_name = "project_head_context.weight"
    detached_gradient = dict(detached.named_parameters())[parameter_name].grad
    end_to_end_gradient = dict(end_to_end.named_parameters())[parameter_name].grad
    if detached_gradient is None or end_to_end_gradient is None:
        raise RuntimeError("head scorer did not receive a gradient")
    gradient_difference = torch.linalg.vector_norm(
        end_to_end_gradient - detached_gradient
    ).item()
    finite = bool(
        torch.isfinite(detached_gradient).all()
        and torch.isfinite(end_to_end_gradient).all()
    )
    assert same_forward
    assert finite
    assert gradient_difference > 0.0

    payload = {
        "status": "passed",
        "same_forward_actions_cost_and_log_likelihood": same_forward,
        "same_discrete_actions": same_actions,
        "maximum_cost_difference": maximum_cost_difference,
        "maximum_log_likelihood_difference": maximum_log_likelihood_difference,
        "changed_backward_parameter": parameter_name,
        "gradient_difference_norm": gradient_difference,
        "finite_gradients": finite,
        "interpretation": (
            "The intervention preserves the exact forward policy and changes only "
            "whether tail-selection credit flows through the summary into the head scorer"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    main(arguments.output.resolve(), arguments.device)
