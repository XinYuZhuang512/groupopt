# 代码分层与公共接口

## 目标

代码结构需要直接表达论文中的边界：GroupOpt 范式只定义“怎样构造”，问题插件定义
“什么动作合法”，神经方法只定义“怎样评分”，训练目标和实验编排都不是范式本身。

依赖方向固定为：

```text
experiments
  ├── objectives
  └── adapters / models
          ↓
      framework  ←  problems
```

`framework` 不允许反向导入 `models`、`adapters`、`objectives` 或 `experiments`。

## 目录职责

### `groupopt/framework/`

范式的稳定公共 API：

- `ConstructionProcess`：标量参考过程；
- `ConstructionAction`、`ConstructionTrace`：通用动作和轨迹类型；
- `construct(...)`：完全不依赖神经网络的参考执行器。

`framework/neural.py` 是可选的 PyTorch 桥接层，提供
`BatchedConstructionProcess`、`ConstructionModel` 和 `ConstructionOutput`。
因此只使用数学构造核心时不需要安装 PyTorch。

这里不包含 TSP、AM、PtrNet、GPN、REINFORCE 或 SYM-NCO 的实现。

### `groupopt/problems/`

问题层。当前包含：

- `DirectedTSPConstruction`：便于证明和穷举测试的标量实现；
- `BatchedTSPConstruction`：训练使用的张量插件；
- `BatchedTSPState`：TSP 部分路径状态。

合法 mask、防止提前成环、固定基、状态转移、最终解和目标函数全部由这里提供。

### `groupopt/models/`

具体神经方法的实现，包括 AM、PtrNet、GPN 和现代 encoder controls。模型可以读取状态
特征并给候选动作评分，但通过 `BatchedConstructionProcess` 调用合法性、转移和目标，
不再直接拥有这些语义。

### `groupopt/adapters/`

方法接入层。`ModelRegistry` 是增加新模型的唯一注册入口。评估代码只接收统一 config，
不再包含针对每一种模型的条件分支。

增加新模型时：

1. 在 `models/` 中实现满足 `ConstructionModel` 输出契约的方法；
2. 在 `adapters/registry.py` 注册 builder；
3. 添加接口测试，不修改 `framework/`。

### `groupopt/objectives/`

训练方法层。标准 REINFORCE 与 SYM-NCO 位于这里。训练目标可以改变梯度和数据增强，
但不能改变构造合法性或模型原生 decoder。

### `experiments/`

只负责配置、运行、评估、汇总和服务器生命周期。实验脚本不是库的公共 API。

## 两个核心扩展接口

新问题实现 `BatchedConstructionProcess`：

```python
class MyProblem:
    def initial_state(self, instance): ...
    def is_terminal(self, state): ...
    def base_mask(self, state): ...
    def representative_mask(self, state, selected_base): ...
    def action_mask(self, state, fixed_base=None): ...
    def fixed_base(self, state, anchor=0): ...
    def transition(self, state, selected_base, selected_representative): ...
    def objective(self, state, selected_bases, selected_representatives): ...
    def solution(self, state): ...
```

新模型通过统一入口输出 `ConstructionOutput`：

```python
output = model(
    instance,
    decode_type="greedy",
    base_mode="native_conditional_free",
)
```

只要替换模型时无需修改问题插件，跨模型主张成立；只要替换问题时无需修改框架契约，
问题层抽象成立。当前论文仍只主张前者已经被实验验证。

## 兼容策略

旧入口 `groupopt.construction`、`groupopt.training` 和 `groupopt.symnco` 暂时保留为轻量
转发模块，已有 checkpoint 和实验脚本不会因此失效。新代码分别使用：

- `groupopt.framework`
- `groupopt.objectives`
- `groupopt.adapters`

发布前可以给兼容入口增加弃用周期，但本轮不做破坏性删除。
