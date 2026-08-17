# GroupOpt

GroupOpt is a stabilizer-guided neural construction framework for combinatorial
optimization. It separates the semantics of constructing a feasible solution from
the neural architecture used to score the next action.

当前仓库实现最初群作用与稳定化子链设想在 TSP 上的第一个可训练版本。当前研究主张是
**跨模型**，而不是跨搜索策略或跨所有组合优化问题。

## 核心方法

TSP 的部分解由若干条互不相交的有向路径组成。一次 GroupOpt 动作分成两步：

1. 从所有开放路径的 `tail` 中选择下一条要扩展的路径；
2. 保留接入模型的原生 decoder，从合法 `head` 中选择连接目标。

对于每个候选 tail，原生 decoder 先给出条件分布
`P(head | tail, state)`。轻量 tail scorer 将该分布的熵、预期边长和最大概率，与节点表示、
所属路径的平均表示、路径起点、路径规模及上一步 head 融合后选择 tail。问题插件统一维护
合法 mask：在最后一步之前禁止连接同一分量，从而避免提前形成子环。

在论文和实验表格中建议使用以下名称：

- `Original`：模型的原生固定顺序构造；
- `Ours`：原生模型加 GroupOpt tail selection；
- `SYM-NCO`：原生模型加 SYM-NCO 训练目标。

## 代码结构

```text
src/groupopt/
├── framework/   # 稳定公共接口和模型无关的构造执行器
├── problems/    # 问题状态、合法动作、状态转移和目标函数
├── models/      # AM、PtrNet、GPN 及其他神经方法
├── adapters/    # 模型注册与统一构建入口
└── objectives/  # REINFORCE、SYM-NCO 等训练目标

experiments/
├── train/       # 单模型训练入口
├── evaluation/  # 独立 checkpoint 评估
├── analysis/    # 统计汇总与可视化
└── pipelines/   # 当前与历史批量实验流程

tests/
├── framework/   # 接口和依赖边界
├── problems/    # 状态、约束和分布
├── models/      # 各模型和 Tail Selector
├── objectives/  # 训练目标
└── integration/ # 跨层兼容性

docs/            # 数学定义、架构决策与实验记录
```

依赖边界和接入新模型的方法见
[代码架构](docs/code-architecture.md)，数学定义和当前主张见
[框架说明](docs/framework-v0.md)。

## 安装

需要 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml,dev]'
```

## 最小示例

模型无关的参考执行器不依赖 PyTorch：

```python
from groupopt import construct
from groupopt.problems.tsp import DirectedTSPConstruction

trace = construct(
    DirectedTSPConstruction(),
    5,
    select_base=lambda _state, candidates: candidates[0],
    select_representative=lambda _state, _base, candidates: candidates[0],
)
print(trace.solution.order)
```

神经模型统一使用 `base_mode` 切换原生构造与 GroupOpt：

```python
import torch

from groupopt.models.am import AdaptiveAttentionModel

model = AdaptiveAttentionModel(
    embedding_dim=128,
    n_heads=8,
    n_encoder_layers=3,
    feed_forward_dim=512,
    normalization="batch",
)
coordinates = torch.rand(16, 50, 2)

original = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_conditional_fixed",
)
ours = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_conditional_free",
)
```

## 验证

```bash
python -m pytest -q
python -m ruff check src tests experiments
```

当前测试覆盖公共接口、TSP 构造可行性、防止提前成环、AM/PtrNet/GPN 接入、条件原生
decoder、模型注册、训练目标和独立 checkpoint 评估。

## 研究状态

- 已实现 TSP 的固定基与自适应基构造；
- 已实现保留原生 decoder 的条件式 GroupOpt 接口；
- 已接入 AM、PtrNet、GPN 和多种实验性编码器；
- 已实现 REINFORCE 与 SYM-NCO 训练目标；
- 当前实验属于研究 pilot，不应解读为最终论文结论；
- CSC、VRP、一步多边群作用和共轭群作用仍属于后续工作。

生成数据、checkpoint、服务器备份和实验产物不纳入版本管理。
