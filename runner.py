import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from simulated_annealing import SimulatedAnnealingSolver
from tabu_search import TabuSearchSolver


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


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


def rollout_with_policy(env, policy):
    from tianshou.data import Batch

    obs, _ = env.reset()
    terminated = False
    truncated = False
    history = []

    while not (terminated or truncated):
        batched_obs = {
            "edge_index": np.expand_dims(obs["edge_index"], axis=0),
            "node_features": np.expand_dims(obs["node_features"], axis=0),
            "col_features": np.expand_dims(obs["col_features"], axis=0),
            "k": np.array([obs["k"]], dtype=np.int64),
        }
        batch = Batch(obs=batched_obs, info={})
        with torch.no_grad():
            result = policy(batch)

        logits = result.logits
        if isinstance(logits, torch.Tensor):
            action = int(torch.argmax(logits, dim=-1).detach().cpu().numpy()[0])
        else:
            action = int(np.asarray(result.act)[0])

        mapped_action = policy.map_action(np.array([action]))[0]
        node, color = int(mapped_action[0]), int(mapped_action[1])

        obs, reward, terminated, truncated, info = env.step(mapped_action)

        entropy = 0.0
        if hasattr(result, "dist") and result.dist is not None:
            entropy_tensor = result.dist.entropy()
            if isinstance(entropy_tensor, torch.Tensor):
                entropy = float(entropy_tensor.mean().item())
            else:
                entropy = float(np.mean(entropy_tensor))

        history.append(
            {
                "step": int(info.get("step", len(history) + 1)),
                "node": node,
                "color": color,
                "reward": float(reward),
                "immediate_reward": float(info.get("immediate_reward", 0.0)),
                "search_reward": float(info.get("search_reward", 0.0)),
                "total_reward": float(info.get("total_reward", reward)),
                "conflicts": float(info.get("conflicts", 0.0)),
                "best_conflicts": float(info.get("best_conflicts", 0.0)),
                "entropy": entropy,
            }
        )

    solution = env.get_solution()
    conflicts = calculate_score(env.get_graph(), solution)
    return solution, conflicts, history


def save_visualization(graph, solution, conflicts, history, colors, fig_path):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax_graph = axes[0]
    pos = nx.spring_layout(graph, seed=42)

    if solution is None:
        node_colors = [0 for _ in graph.nodes()]
    else:
        node_colors = [solution[node] for node in graph.nodes()]

    cmap = mcolors.ListedColormap(plt.cm.jet(np.linspace(0, 1, max(colors, 1))))
    nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_color=node_colors, cmap=cmap, ax=ax_graph)

    conflict_edges = get_conflict_edges(graph, solution)
    non_conflict_edges = [edge for edge in graph.edges() if edge not in conflict_edges]
    nx.draw_networkx_edges(graph, pos, edgelist=non_conflict_edges, alpha=0.7, ax=ax_graph)
    nx.draw_networkx_edges(graph, pos, edgelist=conflict_edges, edge_color="red", width=1.5, ax=ax_graph)
    ax_graph.set_title(f"Graph Coloring\nConflicts={conflicts}")
    ax_graph.axis("off")

    ax_curve = axes[1]
    if history:
        steps = [item["step"] for item in history]
        curr_conflicts = [item["conflicts"] for item in history]
        best_conflicts = [item["best_conflicts"] for item in history]
        ax_curve.plot(steps, curr_conflicts, label="conflicts")
        ax_curve.plot(steps, best_conflicts, label="best_conflicts")
        ax_curve.set_xlabel("Step")
        ax_curve.set_ylabel("Conflicts")
        ax_curve.set_title("Conflict Curve")
        ax_curve.legend()
        ax_curve.grid(alpha=0.3)
    else:
        ax_curve.text(0.5, 0.5, "No RL step history", ha="center", va="center")
        ax_curve.set_axis_off()

    ax_hist = axes[2]
    if solution is not None:
        used_colors = np.array(solution)
        bins = np.arange(-0.5, colors + 0.5, 1)
        ax_hist.hist(used_colors, bins=bins, rwidth=0.85)
        ax_hist.set_xlim(-0.5, colors - 0.5)
        ax_hist.set_xlabel("Color ID")
        ax_hist.set_ylabel("Node Count")
        ax_hist.set_title(f"Color Usage ({len(set(solution))}/{colors})")
        ax_hist.grid(axis="y", alpha=0.3)
    else:
        ax_hist.text(0.5, 0.5, "No solution", ha="center", va="center")
        ax_hist.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph coloring solver with RL/local-search ablations")
    parser.add_argument("graph", type=str, help="Path to graph file")
    parser.add_argument("--input", type=str, default=None, help="Path to policy file (required for RL mode)")
    parser.add_argument("--ablation", choices=["full", "no_rl"], default="full", help="Ablation mode")
    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn", help="Model type")

    parser.add_argument("-K", "--colors", type=int, default=None, help="Number of colors")
    parser.add_argument("-S", "--sa-iters", type=int, default=5000000, help="SA iterations")
    parser.add_argument("-T", "--initial-temp", type=float, default=500.0, help="Initial temperature")
    parser.add_argument("-C", "--cooling-rate", type=float, default=0.995, help="Cooling rate")
    parser.add_argument("-M", "--min-temp", type=float, default=0.01, help="Minimum temperature")

    parser.add_argument("--tabu-iters", type=int, default=5000, help="Tabu search iterations")
    parser.add_argument("--tabu-tenure", type=int, default=20, help="Tabu tenure")
    parser.add_argument("--search-algorithm", choices=["sa", "tabu"], default="sa", help="Local search algorithm")

    parser.add_argument("-I", "--max-steps-RL", type=int, default=300, help="Max RL steps per episode")
    parser.add_argument("-R", "--render", type=str2bool, default=False, help="Whether to display plot")
    parser.add_argument("-O", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--save-fig", type=str, default=None, help="Path to save visualization image")
    parser.add_argument("--save-history", type=str, default=None, help="Path to save per-step RL history JSON")
    parser.add_argument("-B", "--beta", type=float, default=0.2, help="Local search reward weight")
    args = parser.parse_args()

    graph = read_graph_from_file(args.graph)
    n = len(graph)
    if args.colors is None:
        args.colors = max(graph.degree())[1] + 1

    history = []
    start_time = time.perf_counter()

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
            raise ValueError("RL mode requires --input policy file")

        from tianshou.utils.net.common import ActorCritic

        from gcp_env import GcpEnv
        from network import ActorNetwork, CriticNetwork, GCPPPOPolicy

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
            dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
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

        solution, conflicts, history = rollout_with_policy(env, policy)

    elapsed_sec = time.perf_counter() - start_time

    print(f"Solve finished. Conflicts: {conflicts}")
    print(f"Solution: {solution}")
    sys.stdout.flush()

    if args.render:
        save_visualization(graph, solution, conflicts, history, args.colors, Path("results") / "last_render.png")
        import matplotlib.pyplot as plt

        img = plt.imread(Path("results") / "last_render.png")
        plt.figure(figsize=(14, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    if args.save_fig:
        save_visualization(graph, solution, conflicts, history, args.colors, args.save_fig)
        print(f"Visualization saved to: {args.save_fig}")

    if args.save_history:
        history_path = Path(args.save_history)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Step history saved to: {history_path}")

    output = {
        "solution": solution.tolist() if isinstance(solution, np.ndarray) else solution,
        "conflicts": conflicts,
        "colors": args.colors,
        "nodes": n,
        "edges": len(graph.edges()),
        "ablation": args.ablation,
        "model_type": args.model_type,
        "search_algorithm": args.search_algorithm,
        "runtime_sec": elapsed_sec,
        "history_len": len(history),
    }

    output_path = Path(args.output) if args.output else None
    if output_path and output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_path if output_path else Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_name = Path(args.graph).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_name = (
            f"{graph_name}_{args.ablation}_{args.model_type}_{args.search_algorithm}"
            f"_k{args.colors}_c{conflicts}_{timestamp}.json"
        )
        output_path = output_dir / auto_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Result saved to: {output_path}")
    sys.stdout.flush()
