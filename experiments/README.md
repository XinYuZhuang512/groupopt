# 实验代码

本目录只保存可复现的实验入口和编排，不包含 GroupOpt 范式、问题约束或模型实现。

```text
experiments/
├── train/              单模型训练入口
├── evaluation/         锁定 checkpoint 的独立评估
├── analysis/           统计汇总与可视化
└── pipelines/
    ├── current/        当前论文方向仍使用的实验流程
    └── archive/        已被后续方案取代的历史实验流程
```

## 当前入口

- `train/train_*_experiment.py`：训练一个具体模型；
- `evaluation/evaluate_checkpoint.py`：在固定独立测试集上评估 checkpoint；
- `analysis/summarize_*.py`：聚合配对结果和因子实验；
- `analysis/plot_*.py`：生成图表；
- `pipelines/current/run_*_server.sh`：服务器后台运行包装器。

当前主要实验流程：

- `run_native_conditional_pilot.sh`：AM、PtrNet、GPN 的原生条件 decoder 接入；
- `run_native_pointer_pilot.sh`：PtrNet/GPN 原生 decoder pilot；
- `run_modern_native_full.sh`：现代模型扩展；
- `run_distribution_shift_eval.sh`：分布偏移测试；
- `run_credibility_audit.sh`：参数量、数据泄露和公平性检查；
- `run_symnco_factorial_pilot.sh`：AM、Ours、SYM-NCO 因子实验。

所有命令均从仓库根目录运行，例如：

```bash
PYTHONPATH=src python3 experiments/train/train_am_experiment.py --help
bash experiments/pipelines/current/run_native_conditional_pilot.sh
```

`pipelines/archive/` 只用于复现实验演化过程，不代表当前推荐方案。生成数据、checkpoint
和结果保存到 `artifacts/`，不直接提交到 Git。
