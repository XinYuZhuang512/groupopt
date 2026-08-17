# 实验

这里只保存可复现的实验编排，不放范式、问题、模型或训练目标实现。

- `train_*_experiment.py`：单个模型训练入口；
- `evaluate_checkpoint.py`：统一独立测试入口；
- `run_*.sh`：可恢复的实验批次；
- `summarize_*.py`、`plot_*.py`：统计与可视化；
- `*_server.sh`：服务器后台运行和 exit 状态文件。

生成数据、checkpoint 和结果不直接提交到 Git。公共代码边界见
`docs/code-architecture.md`。
