# AM 与 PtrNet 学习曲线分析

## 方法

- 数据源：五个正式训练 seed 的 `metrics.jsonl`；
- 纵轴：截至每个评估点的 best validation greedy cost；
- 等 step：在每 250 steps 的共同评估点上对五个 seed 求均值；
- 等时间：修复断点续训造成的 elapsed time 重置后，以最近一个已完成评估点构造
  阶梯曲线，再对五个 seed 求均值；
- 等时间比较只使用三种模式都覆盖的公共时间区间。

分析数据由 `experiments/analysis/analyze_learning_curves.py` 生成。

## Attention Model

`adaptive_state` 从 step 250 开始，在所有后续共同评估点上均优于 fixed 和
adaptive。代表性结果如下：

| 预算 | fixed | adaptive | adaptive_state |
|---:|---:|---:|---:|
| 2,000 steps | 6.7452 | 6.6595 | **6.3539** |
| 6,000 steps | 6.5008 | 6.5019 | **6.1341** |
| 10,000 steps | 6.4094 | 6.4411 | **6.0696** |
| 300 s | 6.6618 | 6.6614 | **6.3831** |
| 600 s | 6.5181 | 6.5519 | **6.2355** |
| 900 s | 6.4346 | 6.5019 | **6.1591** |

AM 的 adaptive_state 平均训练耗时为 1,710 s，fixed 为 1,065 s，即前者约慢
60.6%。但 adaptive_state 在相同 wall-clock budget 下仍持续领先，因此 AM 的质量
提升不能用额外计算量完全解释。

## Pointer Network

PtrNet 呈现两阶段行为：adaptive_state 在 step 250 至 8,250 之间优于 fixed，
随后 fixed 在 step 8,250–8,500 之间反超。

| 预算 | fixed | adaptive | adaptive_state |
|---:|---:|---:|---:|
| 2,000 steps | 7.3059 | 6.7315 | **6.6578** |
| 6,000 steps | 6.7409 | 6.5498 | **6.4492** |
| 8,000 steps | 6.4313 | 6.5051 | **6.3988** |
| 8,500 steps | **6.3801** | 6.4970 | 6.3922 |
| 10,000 steps | **6.3187** | 6.4778 | 6.3606 |

等时间结论相同：adaptive_state 在约 120–810 s 之间领先 fixed，反超发生在约
810–840 s。代表性结果：

| 时间预算 | fixed | adaptive | adaptive_state |
|---:|---:|---:|---:|
| 300 s | 7.2168 | 6.6957 | **6.6845** |
| 600 s | 6.8639 | 6.5857 | **6.5200** |
| 900 s | **6.3801** | 6.5280 | 6.4459 |

PtrNet 的 adaptive_state 平均训练耗时为 1,483 s，fixed 为 1,046 s，约慢
41.8%。它显著改善早期样本效率，但 fixed 在训练后段继续下降并取得更好的最终
结果。

## 阶段性结论

1. AM 中 adaptive_state 同时改善早期收敛、等时间质量和最终质量；
2. PtrNet 中 adaptive_state 改善早期收敛，但没有改善最终质量；
3. PtrNet 的失败不是“从一开始就学不会”，而是 fixed 后期收敛能力更强；
4. 下一轮消融应优先解释状态编码带来的早期优势，以及双动作策略为何较早进入
   平台期。
