# GCP-RL

基于强化学习（PPO）的图着色项目，支持与局部搜索结合求解：

- `RL + SA`（模拟退火）
- `RL + Tabu`（禁忌搜索）
- 消融模式：`no_rl`（只跑局部搜索）
- 模型消融：`GNN` / `MLP`

## 1. 环境准备

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

依赖见 `requirements.txt`，核心库包括 `torch`、`tianshou`、`torch-geometric`、`networkx`。

## 2. 训练策略

`trainer.py` 的第一个位置参数是输出模型路径（`.pth`）。

```bash
python trainer.py checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

在已有策略上继续训练：

```bash
python trainer.py checkpoints/policy_v2.pth \
  --input checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

常用参数：

- `--model-type {gnn,mlp}`：策略网络结构
- `--search-algorithm {sa,tabu}`：RL 后接的局部搜索算法
- `--epochs`：训练轮数（默认 50）
- `--nodes --probability --colors`：随机图规模和颜色数
- `--tabu-iters --tabu-tenure`：Tabu 参数
- `--sa-iters --initial-temp --cooling-rate --min-temp`：SA 参数

## 3. 推理/求解

### 3.1 完整方案（RL + 局部搜索）

```bash
python runner.py test_graph.txt \
  --input checkpoints/policy.pth \
  --ablation full \
  --model-type gnn \
  --search-algorithm sa
```

### 3.2 消融：不使用 RL（只跑局部搜索）

```bash
python runner.py test_graph.txt \
  --ablation no_rl \
  --search-algorithm tabu \
  --tabu-iters 5000 \
  --tabu-tenure 20
```

结果默认输出到 `results/`，也可通过 `--output` 指定 JSON 路径：

```bash
python runner.py test_graph.txt --ablation no_rl --output results/result.json
```

## 4. 输入图格式

支持 DIMACS 风格边列表（见 `test_graph.txt`）：

```text
p edge <节点数> <边数>
e <u> <v>
```

注意：文件中节点编号从 `1` 开始，程序内部会自动转为从 `0` 开始。

## 5. 项目结构

- `trainer.py`：训练入口
- `runner.py`：推理/求解入口
- `gcp_env/gcp_env.py`：Gym 环境
- `network.py`：Actor/Critic 与 PPO 策略封装
- `simulated_annealing.py`：模拟退火
- `tabu_search.py`：禁忌搜索
- `test_graph.txt`：示例图数据
