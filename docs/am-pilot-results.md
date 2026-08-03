# AM TSP-50 Pilot 结果

日期：2026-08-03
代码提交：`855d03b`

## 实验设置

- 问题：均匀随机欧氏 TSP-50
- 模型：AM-style encoder 和 attention decoder
- embedding dim：128
- attention heads：8
- encoder layers：3
- feed-forward dim：512
- batch size：512
- validation size：1024
- 更新步数：2000
- seed：1234
- optimizer：Adam，learning rate `1e-4`
- 固定基和自适应基使用相同模型初始化、相同训练实例序列和相同验证实例

## 结果

| 构造方式 | 初始验证 cost | 最终/最佳验证 cost | 耗时 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| 固定基 | 15.725454 | 6.782593 | 213.54 秒 | 1.936GB |
| 自适应基 | 18.287048 | 6.632879 | 297.57 秒 | 3.267GB |

在这个单 seed pilot 中，自适应基最终验证 cost 相对固定基降低 `2.21%`。自适应基耗时增加 `39.35%`，峰值显存增加 `68.74%`，但绝对显存占用仍远低于 RTX 4090 的 24GB 容量。

## 初步解释

该结果支持继续实验，但不能单独支持论文结论：

- 两条曲线在 2000 步时仍有下降趋势，尚未充分收敛；
- 当前只有一个随机种子；
- 训练中的 validation set 在同一 seed 的两种模式间相同，但正式报告仍需要独立的共享 test set；
- 尚未计算相对 Concorde/LKH 解的 optimality gap；
- 自适应基的优势需要在第二种模型上复现，才能支持跨模型主张。

下一阶段采用 5 个随机种子、每组 10000 步，并保留完整 checkpoint。正式训练完成后，在单独生成并冻结的公共 test set 上统一评估所有 best checkpoint。
