# GCP-RL

图着色项目，支持：

- **RL + 局部搜索**（SA / Tabu）
- **无RL消融**（仅SA或仅Tabu）
- **模型消融**（GNN vs MLP）

## 训练

> `trainer.py` 的第一个位置参数是**策略输出文件路径**，训练结束后会保存到该文件。

```bash
mkdir -p checkpoints
python trainer.py checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

如果希望在已有策略上继续训练，可使用：

```bash
python trainer.py checkpoints/policy_v2.pth \
  --input checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

可选：

- `--model-type {gnn,mlp}`：GNN/无GNN消融
- `--search-algorithm {sa,tabu}`：RL后接SA或Tabu
- `--tabu-iters --tabu-tenure`：Tabu参数

## 推理/求解

### 完整方案（RL + 局部搜索）

```bash
python runner.py test_graph.txt --input policy.pth --ablation full --model-type gnn --search-algorithm sa
```

### 去掉RL（消融）

```bash
python runner.py test_graph.txt --ablation no_rl --search-algorithm tabu --tabu-iters 5000 --tabu-tenure 20
```

输出可通过 `--output result.json` 保存。


## 环境准备（最小依赖）

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

- `requirements.txt`：项目最小运行依赖清单。
- `scripts/check_env.py`：一键检查 Python/依赖版本与缺失项。
