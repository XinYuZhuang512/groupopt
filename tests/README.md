# 测试

测试按职责划分，而不是按实验时间排列：

```text
tests/
├── framework/    公共接口和包依赖边界
├── problems/     TSP 状态、合法性、分布和终止性质
├── models/       AM、PtrNet、GPN、Tail Selector 和状态特征
├── objectives/   REINFORCE、SYM-NCO 与训练辅助逻辑
└── integration/  模型注册和 checkpoint 兼容性
```

运行全部测试：

```bash
python -m pytest -q
```

新增代码时，测试应放到其职责对应的子目录。跨多个层次的端到端检查放入
`integration/`。
