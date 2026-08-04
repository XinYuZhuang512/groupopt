# 正式实验记录（2026-08-05）

## 公共配置

- 问题：随机均匀欧氏 TSP-50
- 训练：每组 10,000 steps，batch size 512
- 验证：每组固定 2,048 个实例，greedy decoding
- seeds：1234、2345、3456、4567、5678
- 指标：训练期间最优 validation greedy cost（越低越好）

## Attention Model：已完成

| seed | fixed | adaptive | adaptive_state |
|---:|---:|---:|---:|
| 1234 | 6.402240 | 6.448270 | **6.068567** |
| 2345 | 6.405267 | 6.377750 | **6.036407** |
| 3456 | 6.427669 | 6.442460 | **6.068284** |
| 4567 | 6.412895 | 6.548978 | **6.027446** |
| 5678 | 6.399013 | 6.387825 | **6.147495** |
| mean | 6.409417 | 6.441057 | **6.069640** |
| sample SD | 0.011426 | 0.068085 | 0.047300 |

`adaptive_state` 在 5/5 个配对 seed 上胜过 fixed 和 adaptive；平均相对 fixed
改善 5.30%，相对 adaptive 改善 5.77%。峰值模型显存约 8.22 GiB。

## Pointer Network：暂停时状态

| seed | fixed | adaptive | adaptive_state |
|---:|---:|---:|---:|
| 1234 | 6.340660 | 6.490411 | 6.373386 |
| 2345 | 6.304067 | 6.478384 | 6.361577 |
| 3456 | 6.306364 | 6.481494 | 6.357069 |
| 4567 | 6.289928 | 6.471039 | 6.482329（验证到 step 4,750，未完成） |
| 5678 | 未运行 | 未运行 | 未运行 |

完整的前四个 seed 中，fixed 与 adaptive 的均值分别为 6.310255 和 6.480332，
adaptive 比 fixed 差 2.70%。前三个完整 seed 的 `adaptive_state` 均值为
6.364010：比 adaptive 好 1.79%，但比 fixed 差 0.85%。

阶段性解释：PtrNet 上 adaptive 与 adaptive_state 在早期训练收敛更快；到
10,000 steps 后 fixed 当前更优。路径状态编码仍稳定改善朴素 adaptive，但尚未
跨模型稳定胜过 fixed。

## 暂停信息

- 记录时间：2026-08-05 00:35 CST
- PtrNet 总计划：150,000 steps（3 modes × 5 seeds）
- 已记录训练进度：114,823 / 150,000 steps（76.55%）
- 中断目标：`ptrnet_tsp50_adaptive_state_seed4567`
- 可恢复 checkpoint：`interrupted.pt`，step 4,823，best cost 6.482329
- 恢复入口：`experiments/run_ptrnet_formal_tsp50.sh`
- 脚本会优先从 `interrupted.pt`（若比 `latest.pt` 新）恢复。
