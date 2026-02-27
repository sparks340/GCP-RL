import argparse
import json
import sys

import networkx as nx
import numpy as np
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic

from gcp_env import GcpEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
from simulated_annealing import SimulatedAnnealingSolver
from tabu_search import TabuSearchSolver


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("需要布尔值")


def read_graph_from_file(filename):
    edges = []
    n = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("p"):
                _, _, nodes, _ = line.split()
                n = int(nodes)
            elif line.startswith("e"):
                _, u, v = line.split()
                edges.append((int(u) - 1, int(v) - 1))
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


def calculate_score(graph, solution):
    if solution is None:
        return float("inf")
    conflicts = 0
    for u, v in graph.edges():
        if solution[u] == solution[v]:
            conflicts += 1
    return conflicts


def get_conflict_edges(graph, coloring):
    if coloring is None:
        return []
    return [(u, v) for (u, v) in graph.edges() if coloring[u] == coloring[v]]


class RandomGCPPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return type("Obj", (), {"act": action})


def run_no_rl(graph, colors, search_algorithm, sa_iters, initial_temp, cooling_rate, min_temp, tabu_iters, tabu_tenure):
    adj_list = [list(graph.neighbors(node)) for node in range(len(graph))]
    if search_algorithm == "tabu":
        solver = TabuSearchSolver(adj_list, colors, max_iterations=tabu_iters, tabu_tenure=tabu_tenure)
        return solver.solve()
    solver = SimulatedAnnealingSolver(
        adj_list,
        colors,
        max_iterations=sa_iters,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        min_temp=min_temp,
    )
    return solver.solve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图着色求解器（支持RL/无RL消融，GNN/MLP消融）")
    parser.add_argument("graph", type=str, help="图文件路径")
    parser.add_argument("--input", type=str, default=None, help="策略文件路径（RL模式必填）")
    parser.add_argument("--ablation", choices=["full", "no_rl"], default="full", help="消融模式")
    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn", help="模型类型")

    parser.add_argument("-K", "--colors", type=int, default=None, help="颜色数量")
    parser.add_argument("-S", "--sa-iters", type=int, default=5000000, help="模拟退火迭代次数")
    parser.add_argument("-T", "--initial-temp", type=float, default=500.0, help="初始温度")
    parser.add_argument("-C", "--cooling-rate", type=float, default=0.995, help="冷却率")
    parser.add_argument("-M", "--min-temp", type=float, default=0.01, help="最小温度")

    parser.add_argument("--tabu-iters", type=int, default=5000, help="禁忌搜索迭代次数")
    parser.add_argument("--tabu-tenure", type=int, default=20, help="禁忌长度")
    parser.add_argument("--search-algorithm", choices=["sa", "tabu"], default="sa", help="局部搜索算法")

    parser.add_argument("-I", "--max-steps-RL", type=int, default=300, help="每个episode的RL最大步数")
    parser.add_argument("-L", "--max-steps", type=int, default=320, help="每个episode的最大步数")
    parser.add_argument("-R", "--render", type=str2bool, default=False, help="是否渲染")
    parser.add_argument("-O", "--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("-B", "--beta", type=float, default=0.2, help="奖励中的局部搜索权重")
    args = parser.parse_args()

    graph = read_graph_from_file(args.graph)
    n = len(graph)
    if args.colors is None:
        args.colors = max(graph.degree())[1] + 1

    if args.ablation == "no_rl":
        solution, conflicts = run_no_rl(
            graph,
            args.colors,
            args.search_algorithm,
            args.sa_iters,
            args.initial_temp,
            args.cooling_rate,
            args.min_temp,
            args.tabu_iters,
            args.tabu_tenure,
        )
    else:
        if not args.input:
            raise ValueError("RL模式必须提供 --input 策略文件路径")

        env = GcpEnv(
            graph=graph,
            k=args.colors,
            sa_iters=args.sa_iters,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            min_temp=args.min_temp,
            tabu_iters=args.tabu_iters,
            tabu_tenure=args.tabu_tenure,
            search_algorithm=args.search_algorithm,
            beta=args.beta,
            max_episode_steps_RL=args.max_steps_RL,
            max_episode_steps=args.max_steps,
            render_mode="human" if args.render else None,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_gnn = args.model_type == "gnn"

        actor = ActorNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
        critic = CriticNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

        policy = GCPPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            nodes=n,
            k=args.colors,
            action_space=env.action_space,
            eps_clip=0.2,
            dual_clip=None,
            value_clip=True,
            advantage_normalization=True,
            recompute_advantage=True,
        )
        policy.load_state_dict(torch.load(args.input, map_location=device))

        collector = Collector(policy, DummyVectorEnv([lambda: env]))
        collector.reset()
        collector.collect(n_episode=1)
        solution = collector.buffer.get(0, "info").best_solution
        conflicts = calculate_score(graph, solution)

    print(f"求解完成! 冲突数: {conflicts}")
    print(f"解决方案: {solution}")
    sys.stdout.flush()

    if args.render and solution is not None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        nodes = list(graph.nodes())
        color_map = [solution[node] for node in nodes]

        pos = nx.spring_layout(graph, seed=42)
        cmap = mcolors.ListedColormap(plt.cm.jet(np.linspace(0, 1, n)))

        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=color_map, cmap=cmap)
        nx.draw_networkx_labels(graph, pos, labels={node: node for node in nodes})

        conflict_edges = get_conflict_edges(graph, solution)
        non_conflict_edges = [edge for edge in graph.edges() if edge not in conflict_edges]

        nx.draw_networkx_edges(graph, pos, edgelist=non_conflict_edges)
        nx.draw_networkx_edges(graph, pos, edgelist=conflict_edges, edge_color="red")

        plt.title(f"Used Colors: {len(set(solution))}, Conflicts: {conflicts}")
        plt.show()

    if args.output:
        output = {
            "solution": solution.tolist() if isinstance(solution, np.ndarray) else solution,
            "conflicts": conflicts,
            "colors": args.colors,
            "nodes": n,
            "edges": len(graph.edges()),
            "ablation": args.ablation,
            "model_type": args.model_type,
            "search_algorithm": args.search_algorithm,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"结果已保存到: {args.output}")
        sys.stdout.flush()
