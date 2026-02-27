import numpy as np
from gcp_env import GcpEnv
from gymnasium.envs.registration import register
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import json
import os
import sys

import torch
import gymnasium as gym
from tianshou.data import Collector
from tianshou.utils.net.common import ActorCritic
from tianshou.env import DummyVectorEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
from tianshou.policy import BasePolicy
from tianshou.data import Batch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("需要布尔值")

def read_graph_from_file(filename):
    edges = []
    n = 0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("p"):
                _, _, nodes, _ = line.split()
                n = int(nodes)
            elif line.startswith("e"):
                _, u, v = line.split()
                edges.append((int(u) - 1, int(v) - 1))

    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

def calculate_score(graph, solution):
    if solution is None:
        return float("inf")
    conflicts = 0
    for u, v in graph.edges():
        if solution[u] == solution[v]:
            conflicts += 1
    return conflicts

def get_conflict_edges(graph, coloring):
    if solution is None:
        return float("inf")
    conflict_edges = []
    for u, v in graph.edges():
        if solution[u] == solution[v]:
            conflict_edges.append((u,v))
    return conflict_edges

class RandomGCPPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs):
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图着色问题求解器")
    parser.add_argument("input", type=str, help="策略文件路径")
    parser.add_argument("graph", type=str, help="图文件路径")
    parser.add_argument(
        "-K",
        "--colors",
        type=int,
        dest="colors",
        default=None,
        help="颜色数量",
    )
    parser.add_argument(
        "-S",
        "--sa-iters",
        type=int,
        dest="sa_iters",
        default=5000000,
        help="模拟退火迭代次数",
    )
    parser.add_argument(
        "-T",
        "--initial-temp",
        type=float,
        dest="initial_temp",
        default=500.0,
        help="初始温度",
    )
    parser.add_argument(
        "-C",
        "--cooling-rate",
        type=float,
        dest="cooling_rate",
        default=0.995,
        help="冷却率",
    )
    parser.add_argument(
        "-M",
        "--min-temp",
        type=float,
        dest="min_temp",
        default=0.01,
        help="最小温度",
    )
    parser.add_argument(
        "-I",
        "--max-steps-RL",
        type=int,
        dest="max_steps_RL",
        default=300,
        help="每个episode的RL的最大步数",
    )
    parser.add_argument(
        "-L",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=320,
        help="每个episode的最大步数",
    )
    parser.add_argument(
        "-R",
        "--render",
        type=str2bool,
        dest="render",
        default=False,
        help="是否渲染",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        dest="output",
        default=None,
        help="输出文件路径",
    )
    parser.add_argument(
        "-B",
        "--beta",
        type=float,
        dest="beta",
        default=0.2,
        help="奖励计算中的beta参数",
    )

    args = parser.parse_args()

    # 加载图
    print(f"加载图: {args.graph}")
    sys.stdout.flush()

    # G = nx.read_edgelist(args.graph, nodetype=int)
    G = read_graph_from_file(args.graph)
    n = len(G)

    if args.colors is None:
        args.colors = max(G.degree())[1] + 1

    print(f"图信息: 节点数={n}, 边数={len(G.edges())}, 颜色数={args.colors}")
    sys.stdout.flush()

    # 创建环境
    env = GcpEnv(
        graph=G,
        k=args.colors,
        sa_iters=args.sa_iters,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp,
        beta=args.beta,
        max_episode_steps_RL=args.max_steps_RL,
        max_episode_steps = args.max_steps,
        render_mode="human" if args.render else None,
    )

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    sys.stdout.flush()

    # 计算状态和动作维度
    node_features = 3
    col_features = 3

    # 创建网络和策略
    actor = ActorNetwork(node_features, col_features, device=device).to(device)
    critic = CriticNetwork(node_features, col_features, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    dist = torch.distributions.Categorical
    policy = GCPPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        nodes=n,
        k=args.colors,
        action_space=env.action_space,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=True,
    )

    # 加载策略
    print(f"加载策略: {args.input}")
    sys.stdout.flush()
    policy.load_state_dict(torch.load(args.input, map_location=device))

    # 创建收集器
    print("创建环境和收集器...")
    sys.stdout.flush()
    VectorEnvenv = DummyVectorEnv([lambda: env])
    collector = Collector(policy, VectorEnvenv)
    
    
    print("重置收集器...")
    sys.stdout.flush()
    collector.reset()

    # 收集数据
    print("开始求解...")
    sys.stdout.flush()
    
    try:
        result = collector.collect(n_episode=1)
        print("收集完成，获取结果...")
        sys.stdout.flush()
        
        solution = collector.buffer.get(0,'info').best_solution
        conflicts = calculate_score(G,solution)
        
        print(f"求解完成! 冲突数: {conflicts}")
        print(f"解决方案: {solution}")
        sys.stdout.flush()

        if args.render:
            print("渲染图形...")
            sys.stdout.flush()
            # plt.clf()
            # pos = nx.spring_layout(G)
            # colors = [f"C{c}" for c in solution]
            # print(colors)
            # nx.draw(G, pos, node_color=colors, with_labels=True)
            # plt.title(f" Used Colors: {len(set(solution))}, Conflicts: {conflicts}")
            # plt.show()

            plt.figure(figsize=(10, 8))

            nodes = list(G.nodes())
            color_map = [solution[node] for node in nodes]  

            pos = nx.spring_layout(G, seed=42)  
            cmap = mcolors.ListedColormap(plt.cm.jet(np.linspace(0, 1, n)))

            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color_map, cmap=cmap)
            nx.draw_networkx_labels(G, pos, labels={node: node for node in nodes})

            conflict_edges = get_conflict_edges(G, solution)
            non_conflict_edges = [edge for edge in G.edges() if edge not in conflict_edges]

            nx.draw_networkx_edges(G, pos, edgelist=non_conflict_edges)
            nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, edge_color='red')

            plt.title(f" Used Colors: {len(set(solution))}, Conflicts: {conflicts}")
            plt.show()

    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    if args.output is not None:
        # 保存结果
        output = {
            "solution": solution.tolist() if isinstance(solution, np.ndarray) else solution,
            "conflicts": conflicts,
            "colors": args.colors,
            "nodes": n,
            "edges": len(G.edges()),
        }
        
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
            
        print(f"结果已保存到: {args.output}")
        sys.stdout.flush() 