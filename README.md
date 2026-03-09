# GCP-RL

基于 PPO 的图着色项目，支持两种求解方式：

- `RL + Local Search`
- `Local Search Only`

局部搜索算法：

- `sa`：模拟退火
- `tabu`：禁忌搜索

## 环境

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

建议使用项目虚拟环境运行：

```bash
.\.venv\Scripts\python.exe <script> ...
```

## 数据

- `data/*.col`：DIMACS 格式 DSJC 图
- `data/ReadMe.txt`：DSJC 图对应颜色数
- `test_graph.txt`：单次测试示例图

输入图格式：

```text
p edge <nodes> <edges>
e <u> <v>
```

节点编号从 `1` 开始，程序内部会转成从 `0` 开始。

## 训练

训练入口：`trainer.py`

### 1) 使用图库训练（推荐）

先生成训练图库（如果你没有现成 `.col` 数据）：

```bash
python scripts/generate_training_data.py \
  --output-dir data/train_data \
  --distribution 0.1:120,0.3:140,0.5:40 \
  --node-ranges 40-60:80,60-90:140,90-120:80
```

开始训练：

```bash
python trainer.py checkpoints/policy_library.pth \
  --model-type gnn \
  --search-algorithm sa \
  --input_data data/train_data
```

说明：

- 训练时每个 episode 会从图库中随机抽样图。
- 图库目录需包含 `.col` 与 `ReadMe.txt`，颜色数自动按 `ReadMe.txt` 加载。
- 默认路径：`--input_data` 默认即为 `data/train_data`。
- 推荐生成配置：
  - 概率分布：`p=0.1:120`、`p=0.3:140`、`p=0.5:40`
  - 节点区间分布：`40~60:80`、`60~90:140`、`90~120:80`
- 颜色规则（用于生成图库和随机图评测）：
  - `p=0.1`：`k=max(3, n//25 + 1)`
  - `p=0.3`：`k=max(4, n//15 + 1)`
  - `p=0.5`：`k=max(6, n//7 - 1)`

### 2) 从已有权重继续训练

```bash
python trainer.py checkpoints/policy_v2.pth \
  --input checkpoints/policy_library.pth \
  --model-type gnn \
  --search-algorithm sa \
  --input_data data/train_data
```

### `trainer.py` 参数

- 输出与加载：`output`、`--input`
- RL 步数：`--max_steps_RL`
- 模拟退火：`--sa-iters`、`--initial-temp`、`--cooling-rate`、`--min-temp`
- 禁忌搜索：`--tabu-iters`、`--tabu-tenure`
- 搜索模式：`--search-algorithm {sa,tabu}`
- 模型：`--model-type {gnn,mlp}`
- 训练轮数：`--epochs`
- 并行环境：`--train-env-num`、`--test-env-num`
- PPO 采样：`--step-per-epoch`、`--step-per-collect`、`--repeat-per-collect`、`--batch-size`、`--episode-per-test`
- 图与采样：
  - 图库图：`--input_data`
- 奖励与优化：`--beta`、`--stagnation-penalty`（仅惩罚真 no-op）、`--reward-scale`、`--actor-lr`、`--critic-lr`、`--lr`（兼容旧参数） 、`--vf-coef`、`--ent-coef`
- 设备：`--device {auto,cpu,cuda}`、`--split-gpus`、`--actor-device`、`--critic-device`

说明：

- `--max_steps_RL` 表示 RL 先运行多少步；达到该步数后执行一次局部搜索并结束这一轮。

## 单次求解

入口：`runner.py`

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

指定输出文件：

```bash
python runner.py test_graph.txt \
  --ablation no_rl \
  --output results/result.json
```

保存单次运行图和 RL 历史：

```bash
python runner.py test_graph.txt \
  --input checkpoints/policy.pth \
  --ablation full \
  --save-fig results/plots/single_run.png \
  --save-history results/plots/single_run_history.json
```

### `runner.py` 参数

- 输入与模式：`graph`、`--input`、`--ablation {full,no_rl}`、`--model-type {gnn,mlp}`
- 颜色与搜索：`--colors`、`--search-algorithm {sa,tabu}`
- 模拟退火：`--sa-iters`、`--initial-temp`、`--cooling-rate`、`--min-temp`
- 禁忌搜索：`--tabu-iters`、`--tabu-tenure`
- RL 相关：`--max-steps-RL`、`--beta`、`--stagnation-penalty`（仅惩罚真 no-op）、`--reward-scale`
- 输出：`--output`、`--output-dir`、`--save-fig`、`--save-history`
- 其他：`--render`

说明：

- `--colors` 不传时，`runner.py` 会默认使用 `max_degree + 1`。
- `--ablation full` 表示 `RL + Local Search`，`--ablation no_rl` 表示纯局部搜索。

## 批量对比

入口：`scripts/benchmark_compare.py`

对比两种方法：

- `rl_local_search`
- `local_search_only`

数据分两类：

- `random`：纯随机 Erdos-Renyi 图，节点数默认在 `60-120` 间随机采样，边概率默认取 `0.1/0.3/0.5`，每个概率配置独立随机生成 20 次，颜色数按概率规则计算
- `dsjc`：`data/` 中实际存在的 DSJC 图，颜色数来自 `data/ReadMe.txt`

默认输出目录：`results/benchmark_compare/`

输出文件：

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
  --random-instances-per-config 20 \
  --random-min-nodes 60 \
  --random-max-nodes 120 \
  --random-probabilities 0.1 0.3 0.5 \
  --max-steps-rl 300 \
  --output-dir results/benchmark_compare_sa
```

`Tabu` 版本：

```bash
.\.venv\Scripts\python.exe scripts\benchmark_compare.py \
  --policy checkpoints/gnn_tabu_train_150nodes_24colors_120_multi_policy.pth \
  --model-type gnn \
  --search-algorithm tabu \
  --random-instances-per-config 20 \
  --random-min-nodes 60 \
  --random-max-nodes 120 \
  --random-probabilities 0.1 0.3 0.5 \
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
  --random-probabilities 0.1 0.3 0.5 \
  --output-dir results/benchmark_compare_random
```

smoke test：

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

### `benchmark_compare.py` 参数

- 模型与搜索：`--policy`、`--model-type {gnn,mlp}`、`--search-algorithm {sa,tabu}`
- 数据路径：`--data-dir`、`--readme-path`
- 数据范围：`--include-random`、`--include-dsjc`、`--dataset-names`
- 随机图采样：`--random-instances-per-config`、`--random-min-nodes`、`--random-max-nodes`、`--random-probabilities`、`--seed`
- DSJC 重复运行：`--dsjc-runs-per-config`
- 模拟退火：`--sa-iters`、`--initial-temp`、`--cooling-rate`、`--min-temp`
- 禁忌搜索：`--tabu-iters`、`--tabu-tenure`
- RL 相关：`--beta`、`--stagnation-penalty`（仅惩罚真 no-op）、`--reward-scale`、`--max-steps-rl`
- 输出：`--output-dir`

说明：

- 随机图颜色数按概率规则：`p=0.1 -> k=max(3, n//25+1)`、`p=0.3 -> k=max(4, n//15+1)`、`p=0.5 -> k=max(6, n//7-1)`。
- DSJC 图颜色数取自 `data/ReadMe.txt`。
- `--max-steps-rl` 表示 RL 先运行多少步，然后接一次局部搜索并结束该实例。

## 结果说明

批量对比常看的文件：

- `benchmark_records.csv`：每次运行明细
- `benchmark_overall_summary.csv`：按 `random/dsjc + method` 聚合后的成功率（`x/总次数`）、平均冲突数、平均耗时
- `benchmark_per_dataset_summary.csv`：按具体数据集配置聚合；随机图配置名形如 `ER_n60-120_p0.1`
- `benchmark_pairwise.csv`：同一实例上 `RL + Local Search` 与 `Local Search Only` 的逐项对比

## 主要文件

- `trainer.py`：训练入口
- `runner.py`：单次求解入口
- `scripts/benchmark_compare.py`：批量对比脚本
- `gcp_env/gcp_env.py`：环境定义
- `network.py`：Actor / Critic / PPO policy
- `simulated_annealing.py`：模拟退火
- `tabu_search.py`：禁忌搜索
