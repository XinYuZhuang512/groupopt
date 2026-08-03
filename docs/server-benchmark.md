# 首台实验服务器基准

日期：2026-08-03

## 环境

- GPU：NVIDIA GeForce RTX 4090，24GB
- 平台分配 CPU：25 核
- 平台分配内存：90GB
- 系统：Ubuntu 22.04
- NVIDIA 驱动：570.124.04
- PyTorch：2.8.0+cu128
- 系统盘：30GB
- 高速数据盘：50GB

代码安装在 `/root/groupopt`，虚拟环境和实验产物位于 `/root/autodl-tmp`。checkpoint、日志和数据不写入系统盘。

## 正确性检查

服务器端完整运行 16 项单元测试，全部通过。CUDA、GPU 名称和 PyTorch CUDA runtime 均已核对。

## TSP-10 训练链路

- graph size：10
- 更新步数：100
- batch size：128
- validation size：256
- device：CUDA
- 初始 greedy cost：4.510839
- 最终 greedy cost：3.553000
- 相对变化：-21.23%
- 峰值 GPU 显存：0.043GB

该结果仅证明训练链路有效，不作为模型效果结论。

## TSP-100 容量测试

共同配置：

- embedding dim：128
- attention heads：8
- encoder layers：3
- feed-forward dim：512
- validation size：256
- 训练更新：1 步

| Batch size | 峰值 GPU 显存 | 完整 smoke 命令耗时 |
| ---: | ---: | ---: |
| 256 | 5.80GB | 4.53 秒 |
| 512 | 11.97GB | 5.03 秒 |

完整命令包含初始 greedy 验证、一次采样训练、训练中验证和最终验证，因此不能将表中耗时直接解释为单个训练 step 的稳定吞吐。

## 当前建议

第一轮正式实验从 TSP-100、batch size 512 开始。该设置只使用约一半显存，为更长训练、框架开销和后续功能保留了安全余量。在增加 batch 前，应先加入稳定的 checkpoint、日志和异常恢复机制。
