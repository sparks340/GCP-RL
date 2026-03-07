# GCP-RL

基于 PPO 的图着色项目，支持两种求解方式：

- `RL + Local Search`
- `Local Search Only`

当前仓库里的局部搜索算法支持：

- `sa`：模拟退火
- `tabu`：禁忌搜索

## 环境

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

建议使用项目虚拟环境运行命令：

```bash
.\.venv\Scripts\python.exe <script> ...
```

## 数据

- `data/*.col`：DIMACS 格式的 DSJC 图
- `data/ReadMe.txt`：DSJC 图对应的颜色数
- `test_graph.txt`：单次测试用示例图

输入图格式：

```text
p edge <nodes> <edges>
e <u> <v>
```

文件中的节点编号从 `1` 开始，程序内部会转成从 `0` 开始。

## 训练

训练入口是 `trainer.py`，第一个位置参数是输出策略文件。

示例：

```bash
python trainer.py checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

继续训练：

```bash
python trainer.py checkpoints/policy_v2.pth \
  --input checkpoints/policy.pth \
  --model-type gnn \
  --search-algorithm sa
```

常用参数：

- `--model-type {gnn,mlp}`
- `--search-algorithm {sa,tabu}`
- `--epochs`
- `--nodes --probability --colors`
- `--train-env-num --test-env-num`
- `--step-per-epoch --step-per-collect --repeat-per-collect --batch-size`
- `--lr --vf-coef --ent-coef`
- `--sa-iters --initial-temp --cooling-rate --min-temp`
- `--tabu-iters --tabu-tenure`
- `--max_steps_RL`：RL 先跑多少步；达到该步数后执行一次局部搜索并结束该轮

## 单次求解

`runner.py` 用于单张图推理。

`RL + Local Search`：

```bash
python runner.py test_graph.txt \
  --input checkpoints/policy.pth \
  --ablation full \
  --model-type gnn \
  --search-algorithm sa
```

`Local Search Only`：

```bash
python runner.py test_graph.txt \
  --ablation no_rl \
  --search-algorithm tabu \
  --tabu-iters 5000 \
  --tabu-tenure 20
```

结果默认写到 `results/`，也可以显式指定：

```bash
python runner.py test_graph.txt \
  --ablation no_rl \
  --output results/result.json
```

如果要保存单次运行图和 RL 历史：

```bash
python runner.py test_graph.txt \
  --input checkpoints/policy.pth \
  --ablation full \
  --save-fig results/plots/single_run.png \
  --save-history results/plots/single_run_history.json
```

## 批量对比

`scripts/benchmark_compare.py` 用来比较：

- `rl_local_search`
- `local_search_only`

对比数据分两类：

- `random`：随机图，边概率来自 `data/ReadMe.txt` 对应配置，节点数默认在 `60-120` 之间随机采样，颜色数默认取 `nodes // 5`
- `dsjc`：`data/` 目录中实际存在的 DSJC 图，颜色数来自 `data/ReadMe.txt`

默认输出到 `results/benchmark_compare/`：

- `benchmark_summary.json`
- `benchmark_records.csv`
- `benchmark_overall_summary.csv`
- `benchmark_per_dataset_summary.csv`
- `benchmark_pairwise.csv`

### 运行示例

`SA` 版本：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_sa_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm sa \
  --random-instances-per-config 5 \
  --random-min-nodes 60 \
  --random-max-nodes 120 \
  --max-steps-rl 300 \
  --output-dir results/benchmark_compare_sa
```

`Tabu` 版本：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_tabu_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm tabu \
  --random-instances-per-config 5 \
  --random-min-nodes 60 \
  --random-max-nodes 120 \
  --max-steps-rl 300 \
  --output-dir results/benchmark_compare_tabu
```

只跑 DSJC：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_sa_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm sa \
  --include-dsjc \
  --output-dir results/benchmark_compare_dsjc
```

只跑随机图：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_sa_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm sa \
  --include-random \
  --random-min-nodes 60 \
  --random-max-nodes 120 \
  --output-dir results/benchmark_compare_random
```

小规模 smoke test：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_sa_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm sa \
  --include-dsjc \
  --dataset-names DSJC125.1 \
  --sa-iters 200 \
  --max-steps-rl 20 \
  --output-dir results/benchmark_compare_smoke
```

### 关键参数

- `--include-random`
- `--include-dsjc`
- `--dataset-names DSJC125.1 DSJC250.5`
- `--random-instances-per-config`
- `--random-min-nodes`
- `--random-max-nodes`
- `--sa-iters`
- `--tabu-iters`
- `--max-steps-rl`

## 结果说明

批量对比时最常看的文件：

- `benchmark_records.csv`：每次运行的明细
- `benchmark_overall_summary.csv`：按 `random/dsjc + method` 聚合后的成功率、平均冲突数、平均耗时
- `benchmark_per_dataset_summary.csv`：按具体数据集配置聚合
- `benchmark_pairwise.csv`：同一实例上 `RL + Local Search` 和 `Local Search Only` 的逐项比较

## 主要文件

- `trainer.py`：训练入口
- `runner.py`：单次求解入口
- `scripts/benchmark_compare.py`：批量对比脚本
- `gcp_env/gcp_env.py`：环境定义
- `network.py`：Actor / Critic / PPO policy
- `simulated_annealing.py`：模拟退火
- `tabu_search.py`：禁忌搜索
