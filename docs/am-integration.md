# AM 首个模型适配

## 目标

保留 Attention Model 的图注意力编码器和注意力式动作评分，同时把解码语义替换为框架定义的两阶段动作：

1. 从可用 `tail` 中选择本步基元素；
2. 从合法 `head` 中选择代表元对应的像；
3. 由模型外部的 TSP 状态机执行遮罩、合并和终止。

## 与原始 AM 的关系

- 编码器仍然对全部二维节点进行多层自注意力编码；
- decoder 仍然使用 query-key attention、glimpse 和 tanh clipping；
- 原始 AM 每步输出一个“下一节点”，本实现每步输出 `tail` 和 `head` 两个条件动作；
- 原始预训练 decoder 的参数形状和动作含义均不匹配，因此第一轮从头训练；
- 后续固定基对照应使用相同 encoder 规模、训练数据和 REINFORCE 配置。

## 当前验证范围

当前仅验证：

- 张量状态与纯 Python 参考状态的转移和遮罩一致；
- greedy 和 sampling 前向均生成合法 Hamilton 环；
- 采样动作的 log likelihood 可以反向传播。

这些测试只说明 AM 已经能够接入框架，还不能说明模型已经学会生成短 tour。下一阶段需要实现训练循环和固定基对照实验。

最小训练链路可以用下面的命令检查：

```bash
PYTHONPATH=src python3 experiments/train/train_am_smoke.py --steps 100
```

脚本默认自动使用 CUDA，也可以显式传入 `--device cpu` 或 `--device cuda`。最终记录同时包含峰值 GPU 显存。

该脚本使用 batch mean baseline 的 REINFORCE，仅用于确认数据生成、采样、代价、梯度、参数更新和 greedy 验证可以形成闭环，不应把一次短训练的数值当作正式实验结论。

## Smoke 记录 001

- 日期：2026-08-03
- 问题：随机均匀欧氏 TSP-10
- 更新步数：100
- batch size：128
- 固定验证集：256 个实例
- 初始 greedy 平均长度：4.483403
- 训练后 greedy 平均长度：3.699204
- 相对变化：-17.49%

该记录表明训练信号和模型适配链路有效，但尚未包含固定基 AM 对照、多随机种子、置信区间和相对最优差距。

## 固定基对照

同一个 `AdaptiveAttentionModel` 可以通过 `base_mode` 使用三种构造方式：

- `fixed`：固定锚点为节点 0，每一步的 base 是当前锚定路径的唯一 tail；模型只对 head 评分。这等价于固定起点后的普通连续路径构造。
- `adaptive`：模型先对所有合法 tail 评分，再条件于 tail 对 head 评分。
- `adaptive_state`：保留上述两阶段决策，并为每个 tail 动态编码其所在路径的
  起点、成员节点平均表示和归一化路径长度，再进行 tail attention。这使基选择器
  能区分“同一个节点处于不同局部路径结构”时的状态。

两种模式共用 encoder、head decoder、TSP 状态机、优化器和数据生成逻辑。实验入口为 `experiments/train/train_am_experiment.py`，会记录 JSONL 指标并原子写入 latest、best 和周期 checkpoint。训练数据和动作采样使用不同的随机数生成器，保证相同 seed 下两种模式每一步看到相同实例。

服务器 pilot 使用 `experiments/pipelines/archive/run_am_pilot.sh`，顺序执行 TSP-50 的固定基与自适应基各 2000 步。该 pilot 用于选择正式训练长度，不作为最终论文表格。
