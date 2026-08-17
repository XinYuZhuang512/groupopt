# Pointer Network 模型适配

## 目的

使用与 AM 实验相同的 TSP 状态机、两阶段动作和 REINFORCE 训练配置，验证
adaptive-base 框架能否从 Transformer 型 AM 迁移到 LSTM Pointer Network。

## 模型结构

- 线性层将二维坐标映射到节点 embedding；
- LSTM 编码节点序列并初始化 LSTMCell decoder；
- additive pointer attention 对候选节点评分；
- 每次确定 head 后，用对应节点 embedding 更新 decoder hidden state。

三种模式共享 encoder、decoder、head pointer 和训练数据：

- `fixed`：状态机确定锚定路径的 tail，模型只选择 head；
- `adaptive`：tail pointer 从所有合法 tail 中选择基；
- `adaptive_state`：tail pointer 额外接收路径起点、分量平均 embedding 和路径长度。

首轮入口为 `experiments/pipelines/archive/run_ptrnet_pilot.sh`，依次运行三种模式的 TSP-50、
seed 1234、2,000-step pilot。
