import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import read_graph_from_file, rollout_with_policy, run_no_rl


@dataclass(frozen=True)
class DatasetSpec:
    dataset_type: str
    dataset_name: str
    config_name: str
    graph_path: Optional[Path]
    nodes: int
    probability: Optional[float]
    colors: int
    trial_id: int = 0
    graph_seed: Optional[int] = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dsjc_readme(readme_path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(readme_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = int(parts[1])
    return mapping


def parse_dsjc_name(name: str) -> Tuple[int, float]:
    stem = name[:-4] if name.endswith(".col") else name
    if not stem.startswith("DSJC"):
        raise ValueError(f"Unsupported DSJC name: {name}")
    body = stem[len("DSJC") :]
    nodes_str, probability_str = body.split(".")
    return int(nodes_str), float(f"0.{probability_str}")


def existing_dsjc_specs(data_dir: Path, readme_path: Path) -> List[DatasetSpec]:
    color_map = parse_dsjc_readme(readme_path)
    specs: List[DatasetSpec] = []
    for dataset_name, colors in sorted(color_map.items()):
        graph_path = data_dir / f"{dataset_name}.col"
        if not graph_path.exists():
            continue
        nodes, probability = parse_dsjc_name(dataset_name)
        specs.append(
            DatasetSpec(
                dataset_type="dsjc",
                dataset_name=dataset_name,
                config_name=dataset_name,
                graph_path=graph_path,
                nodes=nodes,
                probability=probability,
                colors=colors,
            )
        )
    return specs




def random_graph_colors(probability: float, nodes: int) -> int:
    if abs(probability - 0.1) < 1e-9:
        return max(3, nodes // 25 + 1)
    if abs(probability - 0.3) < 1e-9:
        return max(4, nodes // 15 + 1)
    if abs(probability - 0.5) < 1e-9:
        return max(6, nodes // 7 - 1)
    return max(1, nodes // 5)

def random_specs(
    min_nodes: int,
    max_nodes: int,
    probabilities: Sequence[float],
    instances_per_config: int,
    base_seed: int,
) -> List[DatasetSpec]:
    if min_nodes > max_nodes:
        raise ValueError(f"random node range is invalid: min_nodes={min_nodes}, max_nodes={max_nodes}")
    specs: List[DatasetSpec] = []
    for config_index, probability in enumerate(probabilities):
        config_name = f"ER_n{min_nodes}-{max_nodes}_p{probability:.1f}"
        for trial_id in range(instances_per_config):
            graph_seed = base_seed + config_index * 1000 + trial_id
            nodes_rng = random.Random(graph_seed)
            nodes = nodes_rng.randint(min_nodes, max_nodes)
            colors = random_graph_colors(probability, nodes)
            specs.append(
                DatasetSpec(
                    dataset_type="random",
                    dataset_name=f"{config_name}_trial{trial_id + 1}",
                    config_name=config_name,
                    graph_path=None,
                    nodes=nodes,
                    probability=probability,
                    colors=colors,
                    trial_id=trial_id,
                    graph_seed=graph_seed,
                )
            )
    return specs


def expand_dsjc_specs(base_specs: Sequence[DatasetSpec], runs_per_config: int) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for spec in base_specs:
        for trial_id in range(runs_per_config):
            specs.append(
                DatasetSpec(
                    dataset_type=spec.dataset_type,
                    dataset_name=spec.dataset_name,
                    config_name=spec.config_name,
                    graph_path=spec.graph_path,
                    nodes=spec.nodes,
                    probability=spec.probability,
                    colors=spec.colors,
                    trial_id=trial_id,
                    graph_seed=spec.graph_seed,
                )
            )
    return specs


def filter_specs_by_name(specs: Sequence[DatasetSpec], dataset_names: Optional[Sequence[str]]) -> List[DatasetSpec]:
    if not dataset_names:
        return list(specs)
    allowed = set(dataset_names)
    return [spec for spec in specs if spec.config_name in allowed]


def make_random_graph(nodes: int, probability: float, seed: int) -> nx.Graph:
    return nx.gnp_random_graph(nodes, probability, seed=seed)


def build_policy(policy_path: Path, nodes: int, colors: int, model_type: str, device: torch.device):
    from tianshou.utils.net.common import ActorCritic

    from gcp_env import GcpEnv
    from network import ActorNetwork, CriticNetwork, GCPPPOPolicy

    env = GcpEnv(graph=nx.empty_graph(nodes), k=colors)
    use_gnn = model_type == "gnn"

    actor = ActorNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    critic = CriticNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

    policy = GCPPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        nodes=nodes,
        k=colors,
        action_space=env.action_space,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=True,
    )
    state_dict = torch.load(policy_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def solve_with_rl(
    graph: nx.Graph,
    policy_path: Path,
    model_type: str,
    colors: int,
    search_algorithm: str,
    args: argparse.Namespace,
):
    from gcp_env import GcpEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(policy_path, len(graph), colors, model_type, device)
    env = GcpEnv(
        graph=graph,
        k=colors,
        sa_iters=args.sa_iters,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp,
        tabu_iters=args.tabu_iters,
        tabu_tenure=args.tabu_tenure,
        search_algorithm=search_algorithm,
        beta=args.beta,
        stagnation_penalty=args.stagnation_penalty,
        reward_scale=args.reward_scale,
        max_episode_steps_RL=args.max_steps_rl,
        render_mode=None,
    )
    solution, conflicts, history = rollout_with_policy(env, policy)
    return solution, conflicts, history


def solve_with_local_search(graph: nx.Graph, colors: int, search_algorithm: str, args: argparse.Namespace):
    solution, conflicts = run_no_rl(
        graph=graph,
        colors=colors,
        search_algorithm=search_algorithm,
        sa_iters=args.sa_iters,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp,
        tabu_iters=args.tabu_iters,
        tabu_tenure=args.tabu_tenure,
    )
    return solution, conflicts, []


def summarize(records: Sequence[Dict], group_keys: Sequence[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for record in records:
        key = tuple(record[k] for k in group_keys)
        groups.setdefault(key, []).append(record)

    rows: List[Dict] = []
    for key, items in sorted(groups.items()):
        conflicts = [float(item["conflicts"]) for item in items]
        runtimes = [float(item["runtime_sec"]) for item in items]
        success_count = sum(conflict == 0 for conflict in conflicts)
        row = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        row.update(
            {
                "runs": len(items),
                "success_count": success_count,
                "success_rate": f"{success_count}/{len(items)}",
                "success_rate_value": success_count / len(items),
                "mean_conflicts": float(np.mean(conflicts)),
                "median_conflicts": float(median(conflicts)),
                "min_conflicts": float(min(conflicts)),
                "max_conflicts": float(max(conflicts)),
                "mean_runtime_sec": float(np.mean(runtimes)),
                "median_runtime_sec": float(median(runtimes)),
            }
        )
        rows.append(row)
    return rows


def build_pairwise(records: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int], Dict[str, Dict]] = {}
    for record in records:
        key = (record["dataset_type"], record["config_name"], int(record["trial_id"]))
        grouped.setdefault(key, {})[record["method"]] = record

    rows: List[Dict] = []
    for key, methods in sorted(grouped.items()):
        rl_record = methods.get("rl_local_search")
        ls_record = methods.get("local_search_only")
        if rl_record is None or ls_record is None:
            continue
        rl_conflicts = float(rl_record["conflicts"])
        ls_conflicts = float(ls_record["conflicts"])
        rows.append(
            {
                "dataset_type": key[0],
                "config_name": key[1],
                "trial_id": key[2],
                "rl_conflicts": rl_conflicts,
                "local_conflicts": ls_conflicts,
                "conflict_delta": rl_conflicts - ls_conflicts,
                "rl_runtime_sec": float(rl_record["runtime_sec"]),
                "local_runtime_sec": float(ls_record["runtime_sec"]),
                "runtime_delta_sec": float(rl_record["runtime_sec"]) - float(ls_record["runtime_sec"]),
                "winner": (
                    "rl_local_search"
                    if rl_conflicts < ls_conflicts
                    else "local_search_only"
                    if rl_conflicts > ls_conflicts
                    else "tie"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(label: str, rows: Iterable[Dict]) -> None:
    print(label)
    for row in rows:
        print(
            f"- {row['dataset_type']}/{row.get('config_name', 'ALL')}/{row['method']}: "
            f"runs={row['runs']}, success_rate={row['success_rate']}, "
            f"mean_conflicts={row['mean_conflicts']:.3f}, mean_runtime={row['mean_runtime_sec']:.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RL+local-search vs local-search-only on random and DSJC graphs")
    parser.add_argument("--policy", type=str, required=True, help="Path to trained policy .pth")
    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn", help="Policy model type")
    parser.add_argument("--search-algorithm", choices=["sa", "tabu"], default="sa", help="Local search algorithm")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing DSJC .col files")
    parser.add_argument("--readme-path", type=str, default="data/ReadMe.txt", help="DSJC color mapping file")
    parser.add_argument("--include-random", action="store_true", help="Include random graphs")
    parser.add_argument("--include-dsjc", action="store_true", help="Include DSJC graphs")
    parser.add_argument("--random-instances-per-config", type=int, default=20, help="Random graphs per config")
    parser.add_argument("--dsjc-runs-per-config", type=int, default=20, help="Runs per DSJC config")
    parser.add_argument("--random-min-nodes", type=int, default=60, help="Minimum nodes for pure random graphs")
    parser.add_argument("--random-max-nodes", type=int, default=120, help="Maximum nodes for pure random graphs")
    parser.add_argument(
        "--random-probabilities",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Edge probabilities for pure random Erdos-Renyi graphs",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_compare", help="Output directory")
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help="Only run specific dataset configs, e.g. DSJC125.1 DSJC250.5",
    )

    parser.add_argument("--sa-iters", type=int, default=500000, help="SA iterations")
    parser.add_argument("--initial-temp", type=float, default=500.0, help="SA initial temperature")
    parser.add_argument("--cooling-rate", type=float, default=0.995, help="SA cooling rate")
    parser.add_argument("--min-temp", type=float, default=0.01, help="SA minimum temperature")
    parser.add_argument("--tabu-iters", type=int, default=5000, help="Tabu iterations")
    parser.add_argument("--tabu-tenure", type=int, default=20, help="Tabu tenure")
    parser.add_argument("--beta", type=float, default=0.1, help="Local search reward weight for RL mode")
    parser.add_argument("--stagnation-penalty", type=float, default=1e-4, help="Penalty applied on true no-op actions (same color, unchanged conflicts)")
    parser.add_argument("--reward-scale", type=float, default=20.0, help="Global multiplier applied after reward normalization")
    parser.add_argument("--max-steps-rl", type=int, default=300, help="Max RL steps before local search")
    args = parser.parse_args()

    if not args.include_random and not args.include_dsjc:
        args.include_random = True
        args.include_dsjc = True

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise SystemExit(f"Policy file not found: {policy_path}")

    data_dir = Path(args.data_dir)
    readme_path = Path(args.readme_path)
    output_dir = Path(args.output_dir)

    if args.random_min_nodes <= 0 or args.random_max_nodes <= 0:
        raise SystemExit("--random-min-nodes and --random-max-nodes must be positive integers.")
    if args.random_min_nodes > args.random_max_nodes:
        raise SystemExit("--random-min-nodes must be less than or equal to --random-max-nodes.")
    if any(probability <= 0.0 or probability >= 1.0 for probability in args.random_probabilities):
        raise SystemExit("--random-probabilities must all be between 0 and 1.")

    dsjc_specs = (
        expand_dsjc_specs(existing_dsjc_specs(data_dir, readme_path), args.dsjc_runs_per_config)
        if args.include_dsjc
        else []
    )
    random_dataset_specs = (
        random_specs(
            args.random_min_nodes,
            args.random_max_nodes,
            args.random_probabilities,
            args.random_instances_per_config,
            args.seed,
        )
        if args.include_random
        else []
    )
    dsjc_specs = filter_specs_by_name(dsjc_specs, args.dataset_names)
    random_dataset_specs = filter_specs_by_name(random_dataset_specs, args.dataset_names)

    specs = [*random_dataset_specs, *dsjc_specs]
    if not specs:
        raise SystemExit("No datasets found. Check data directory and ReadMe.txt.")

    records: List[Dict] = []
    total_runs = len(specs) * 2
    run_index = 0

    for spec in specs:
        if spec.dataset_type == "random":
            graph = make_random_graph(spec.nodes, float(spec.probability), int(spec.graph_seed))
        else:
            graph = read_graph_from_file(str(spec.graph_path))

        for method in ("rl_local_search", "local_search_only"):
            run_index += 1
            run_seed = args.seed + run_index * 97
            set_global_seed(run_seed)

            start = time.perf_counter()
            if method == "rl_local_search":
                solution, conflicts, history = solve_with_rl(
                    graph.copy(),
                    policy_path=policy_path,
                    model_type=args.model_type,
                    colors=spec.colors,
                    search_algorithm=args.search_algorithm,
                    args=args,
                )
            else:
                solution, conflicts, history = solve_with_local_search(
                    graph.copy(),
                    colors=spec.colors,
                    search_algorithm=args.search_algorithm,
                    args=args,
                )
            runtime_sec = time.perf_counter() - start

            record = {
                "dataset_type": spec.dataset_type,
                "dataset_name": spec.dataset_name,
                "config_name": spec.config_name,
                "trial_id": spec.trial_id,
                "graph_seed": spec.graph_seed,
                "method": method,
                "ablation": "full" if method == "rl_local_search" else "no_rl",
                "model_type": args.model_type,
                "search_algorithm": args.search_algorithm,
                "nodes": spec.nodes,
                "probability": spec.probability,
                "colors": spec.colors,
                "edges": graph.number_of_edges(),
                "conflicts": int(conflicts),
                "success": int(conflicts == 0),
                "runtime_sec": runtime_sec,
                "history_len": len(history),
                "used_colors": len(set(solution)) if solution is not None else None,
            }
            records.append(record)
            print(
                f"[{run_index}/{total_runs}] {spec.dataset_type}/{spec.config_name}/trial{spec.trial_id + 1} "
                f"{method}: conflicts={conflicts}, runtime={runtime_sec:.3f}s"
            )

    overall_rows = summarize(records, ["dataset_type", "method"])
    per_dataset_rows = summarize(records, ["dataset_type", "config_name", "method"])
    pairwise_rows = build_pairwise(records)

    summary_payload = {
        "config": vars(args),
        "records": records,
        "overall_summary": overall_rows,
        "per_dataset_summary": per_dataset_rows,
        "pairwise_comparison": pairwise_rows,
    }

    write_json(output_dir / "benchmark_summary.json", summary_payload)
    write_csv(output_dir / "benchmark_records.csv", records)
    write_csv(output_dir / "benchmark_overall_summary.csv", overall_rows)
    write_csv(output_dir / "benchmark_per_dataset_summary.csv", per_dataset_rows)
    write_csv(output_dir / "benchmark_pairwise.csv", pairwise_rows)

    print_summary("Overall summary:", overall_rows)
    print(f"Saved JSON summary to: {output_dir / 'benchmark_summary.json'}")
    print(f"Saved detailed CSV to: {output_dir / 'benchmark_records.csv'}")
    print(f"Saved overall CSV to: {output_dir / 'benchmark_overall_summary.csv'}")
    print(f"Saved per-dataset CSV to: {output_dir / 'benchmark_per_dataset_summary.csv'}")
    print(f"Saved pairwise CSV to: {output_dir / 'benchmark_pairwise.csv'}")


if __name__ == "__main__":
    main()
