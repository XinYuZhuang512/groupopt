# GroupOpt

GroupOpt 将神经组合优化模型的“依次选下一个点”提升为“在若干条开放局部路径之间选择下一条边”。它不是把训练好的 decoder 原封不动搬过来，也不是替换整个宿主模型；它保留宿主的编码器、Attention/Pointer 评分风格和训练目标，重新定义 decoder 的构造接口，并与宿主模型一起从头训练。

本分支只保存已经在 AM-style、PtrNet、GPN、POMO 上产生一致正向结果的 **Forest-native canonical GroupOpt**。早期 official-AM Adapter、单链 bridge、门控补丁、SYM-NCO 和分布漂移 pilot 不属于当前方法定义，已从本分支移除，仍可在历史分支中恢复。

## 新范式的逻辑

原生顺序 decoder 在第 `t` 步只维护一条链：

```text
当前节点 u_t -> P(v | u_t, 已访问节点) -> 下一节点 v
```

GroupOpt 维护由若干条不相交局部路径组成的 Forest。一步动作分解为：

```text
P(tail, head | Forest)
= P(tail | Forest, 所有 head 提案)
  × P(head | tail, Forest)
```

具体流程如下：

1. 问题层给出所有合法开放端点，以及不会重复连边、不会让顶点度数超过 2、不会提前形成子环的 `tail -> head` mask。
2. 宿主模型按自身风格，为每个合法 tail 计算 `P(head | tail, Forest)`。
3. 将每个 tail 的条件分布概括为归一化熵、预期边长和最大概率，并与 tail 节点表示、所属局部路径的两个端点/平均表示、路径规模、上一步 head 拼接融合。
4. Tail Selector 在所有合法 tail 之间计算 `P(tail | Forest, proposals)`。
5. 先选 tail，再从该行原生风格的条件分布中选 head，加入边并更新 Forest。
6. 只有最后一步允许闭合覆盖全部顶点的 Hamiltonian 环。

这里的关键不是“给原 decoder 前面加一个小模块”，而是把顺序 decoder 的评分原语提升为完整的 Forest-aware decoder。宿主模型的风格被保留，但输入语义从单链上下文变为多路径状态，因此所有参数需要联合重新训练。

## 稳定接口

核心接口在 `src/groupopt/framework/forest_decoder.py`，新模型只需要提供三个钩子：

| 接口 | 输入 | 输出 | 归属 |
|---|---|---|---|
| `score_heads` | Forest 状态、合法边 mask | 每个 tail 的条件 head 分布与候选摘要 | 宿主 decoder 风格 |
| `score_tails` | Forest 状态、全部 head 提案、tail mask | tail 对数概率 | GroupOpt |
| `update_context` | 本步选中的 tail/head | 下一步所需的递归上下文 | 宿主模型 |

`decode_forest_edges` 统一负责采样/贪心选择、状态转移、可行性检查、似然和熵统计。模型适配器只负责评分，不再各自复制整段 Forest rollout。

## 比较组的严格含义

- `native_original`：宿主模型原生的单链解码，是论文主基线。
- `native_conditional_free`：应用完整 GroupOpt 后的 Forest-aware 解码，是论文中的 Ours。
- `native_conditional_fixed`：历史兼容名称；在当前 AM/PtrNet/GPN 实现中与固定起点的原生单链逐动作等价，只用于接口桥接检查。
- `native_capacity_single_chain`：AM 的等活跃参数单链对照。它激活与 Ours 相同数量的参数，但禁止多路径构造，用来排除“仅因参数增加而提升”。

论文的效果结论始终比较 `native_original` 与 `native_conditional_free`。Fixed 和容量对照只解释机制，不能替代 Original。

## 代码结构

```text
src/groupopt/
├── framework/
│   ├── forest_decoder.py  新范式的稳定神经接口与统一 rollout
│   └── neural.py          宿主模型、问题过程和输出协议
├── problems/
│   └── tsp_tensor.py      Forest 状态、合法性 mask、转移、闭环与目标函数
├── models/
│   ├── am.py              AM-style 接入
│   ├── ptrnet.py          Pointer Network 接入
│   ├── gpn.py             Graph Pointer Network 接入
│   ├── pomo.py            官方兼容的 POMO 原生分支与 GroupOpt 接入
│   └── native_conditional.py  条件分布摘要与概率工具
├── adapters/
│   └── registry.py        模型注册和配置构建入口
└── objectives/
    └── reinforce.py       主实验使用的策略梯度目标

experiments/
├── train.py               统一训练、断点和配置记录
├── train_ptrnet.py        PtrNet 宿主入口
├── train_gpn.py           GPN 宿主入口
├── train_pomo.py          POMO 宿主入口
├── evaluate.py            固定独立测试集评估并保存逐实例 cost
├── validate_comparison.py 数据隔离与非活跃参数检查
├── validate_native_original.py  原生单链基线等价性检查
└── paper/
    ├── protocol_tsp50.json       已完成主表的冻结协议
    ├── run_main_tsp50.sh         四宿主×两方法×三种子复现入口
    ├── summarize_main_tsp50.py   配对统计与论文主表汇总
    ├── protocol_convergence_tsp50.json  长程训练选择协议
    ├── run_convergence_tsp50.sh         从已有 checkpoint 独立续训
    └── summarize_convergence_tsp50.py   各训练步配对曲线汇总
```

更详细的逐文件说明见 `docs/CODE_LAYOUT.md`，实验边界和结果出处见
`docs/EXPERIMENT_PROTOCOL.md`。

## 已确认的短程主实验

在同一批 10,000 个 uniform TSP50 实例、三个训练种子上，当前记录为：

| 宿主 | Native Original | Full GroupOpt | 相对改善 |
|---|---:|---:|---:|
| AM-style | 6.7494 | 6.3231 | 6.32% |
| PtrNet | 7.4644 | 6.5935 | 11.67% |
| GPN | 6.4618 | 6.3320 | 2.01% |
| POMO | 6.2761 | 6.1344 | 2.26% |

这些数值用于确认跨宿主方向，不等同于最终充分收敛的论文主表。下一阶段将延长训练预算，检查优势是否在收敛后保持。可审计的逐种子摘要保存在 `paper_records/main_tsp50_v1/`。

## 新模型接入原则

接入新模型时只实现上述三个钩子，不修改问题状态机。实验必须同时满足：

1. `native_original` 先与宿主官方/原生 rollout 做逐动作等价验证；
2. Original 与 Ours 使用相同训练数据、seed、预算、优化器和独立测试实例；
3. 两组都从头训练，不把预训练 decoder 直接迁移到不同状态语义下；
4. 保存逐实例 `costs.pt`，报告配对均值差、95% CI 和胜率；
5. 至少加入等参数或参数量匹配对照，排除新增容量解释。

## 最小调用示例

```python
import torch

from groupopt.models.am import AttentionModel

model = AttentionModel()
coordinates = torch.rand(16, 50, 2)

original = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_original",
)
ours = model(
    coordinates,
    decode_type="greedy",
    base_mode="native_conditional_free",
)
```

训练和评估结果写入 `artifacts/`，该目录默认不进入 Git。
