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

依赖见 `requirements.txt`，核心库包括 `torch`、`tianshou`、`torch-geometric`、`networkx`、`matplotlib`。

## 2. 训练策略

> `trainer.py` 的第一个位置参数是**策略输出文件路径**，训练结束后会保存到该文件。

```bash
mkdir -p checkpoints
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

训练时会写入 TensorBoard 日志，其中新增 `eval/*` 指标（如 `final_conflicts`、`best_conflicts`、`action_entropy_mean`），用于观察策略质量与收敛过程。

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

## 4. 单次可视化与历史导出

`runner.py` 支持导出单次求解可视化和逐步历史：

- `--save-fig`：保存三联图（着色图 + 冲突曲线 + 颜色使用分布）
- `--save-history`：保存逐步历史 JSON（step、action、reward、conflicts 等）

示例：

```bash
python runner.py test_graph.txt \
  --input checkpoints/policy.pth \
  --ablation full \
  --save-fig results/plots/single_run.png \
  --save-history results/plots/single_run_history.json
```

## 5. 批量结果聚合可视化

新增脚本 `scripts/plot_results.py`，用于读取 `results/*.json` 并自动生成论文/汇报常用图表：

- 箱线图：`final_conflicts`（按 `ablation + model_type + search_algorithm` 分组）
- 散点图：`runtime_sec vs final_conflicts`（含帕累托前沿）
- 成功率柱状图：`conflicts == 0` 比例

```bash
python scripts/plot_results.py
```

可选参数：

```bash
python scripts/plot_results.py \
  --input-dir results \
  --output results/plots/results_dashboard.png \
  --summary-csv results/plots/results_summary.csv
```

输出：

- `results/plots/results_dashboard.png`
- `results/plots/results_summary.csv`

## 6. 快速端到端 Smoke Test

```bash
python trainer.py checkpoints/smoke_policy.pth \
  --epochs 1 --nodes 40 --probability 0.15 --colors 8 \
  --max_steps_RL 40 --max-steps 50 \
  --sa-iters 2000 --tabu-iters 200

python runner.py test_graph.txt \
  --input checkpoints/smoke_policy.pth \
  --ablation full --model-type gnn --search-algorithm sa \
  --save-fig results/plots/smoke_single.png \
  --save-history results/plots/smoke_history.json \
  --output results/smoke_full.json

python runner.py test_graph.txt \
  --ablation no_rl --search-algorithm tabu \
  --tabu-iters 1000 --tabu-tenure 20 \
  --output results/smoke_no_rl.json

python scripts/plot_results.py \
  --input-dir results \
  --output results/plots/results_dashboard.png \
  --summary-csv results/plots/results_summary.csv
```

## 7. 输入图格式

支持 DIMACS 风格边列表（见 `test_graph.txt`）：

```text
p edge <节点数> <边数>
e <u> <v>
```

注意：文件中节点编号从 `1` 开始，程序内部会自动转为从 `0` 开始。

## 8. 项目结构

- `trainer.py`：训练入口
- `runner.py`：推理/求解入口
- `scripts/plot_results.py`：批量结果聚合与可视化
- `gcp_env/gcp_env.py`：Gym 环境
- `network.py`：Actor/Critic 与 PPO 策略封装
- `simulated_annealing.py`：模拟退火
- `tabu_search.py`：禁忌搜索
- `test_graph.txt`：示例图数据
