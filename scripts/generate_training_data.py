import argparse
from pathlib import Path

import networkx as nx
import numpy as np


def parse_distribution(raw: str):
    pairs = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        p_str, c_str = token.split(":", 1)
        probability = float(p_str)
        count = int(c_str)
        if not 0 < probability < 1:
            raise ValueError(f"invalid probability: {probability}")
        if count <= 0:
            raise ValueError(f"invalid count: {count}")
        pairs.append((probability, count))
    if not pairs:
        raise ValueError("empty distribution")
    return pairs


def sample_color_count(probability: float, nodes: int) -> int:
    if abs(probability - 0.1) < 1e-9:
        return max(3, nodes // 25 + 1)
    if abs(probability - 0.3) < 1e-9:
        return max(4, nodes // 15 + 1)
    if abs(probability - 0.5) < 1e-9:
        return max(6, nodes // 7 - 1)
    raise ValueError(f"Unsupported probability for color rule: p={probability}")


def write_dimacs(path: Path, graph: nx.Graph):
    edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p edge {graph.number_of_nodes()} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u + 1} {v + 1}\n")


def build_node_samples(node_ranges, rng: np.random.Generator):
    samples = []
    for range_index, ((low, high), count) in enumerate(node_ranges):
        for _ in range(count):
            n = int(rng.integers(low, high + 1))
            samples.append((range_index, n))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate graph library in DIMACS format with fixed p/node distributions")
    parser.add_argument("--output-dir", type=str, default="data/library_train", help="target directory")
    parser.add_argument("--distribution", type=str, default="0.1:120,0.3:140,0.5:40", help="probability counts")
    parser.add_argument(
        "--node-ranges",
        type=str,
        default="40-60:80,60-90:140,90-120:80",
        help="node-range counts, format low-high:count,...",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prob_dist = parse_distribution(args.distribution)
    node_ranges = []
    for token in args.node_ranges.split(","):
        span, count_str = token.split(":", 1)
        low_str, high_str = span.split("-", 1)
        low = int(low_str)
        high = int(high_str)
        count = int(count_str)
        if low <= 0 or high < low:
            raise ValueError(f"invalid node range: {span}")
        if count <= 0:
            raise ValueError(f"invalid node-range count: {count}")
        node_ranges.append(((low, high), count))

    total_prob = sum(count for _, count in prob_dist)
    total_nodes = sum(count for _, count in node_ranges)
    if total_prob != total_nodes:
        raise ValueError(f"count mismatch: probability total={total_prob}, node-range total={total_nodes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    probs = []
    for probability, count in prob_dist:
        probs.extend([probability] * count)

    node_samples = build_node_samples(node_ranges, rng)
    rng.shuffle(probs)
    rng.shuffle(node_samples)

    pair_records = list(zip(probs, node_samples))

    readme_lines = []
    p_summary = {p: 0 for p, _ in prob_dist}
    r_summary = {idx: 0 for idx in range(len(node_ranges))}

    for idx, (probability, (range_idx, n)) in enumerate(pair_records, start=1):
        p_summary[probability] += 1
        r_summary[range_idx] += 1

        graph_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        graph = nx.gnp_random_graph(n, probability, seed=graph_seed)
        name = f"ER_p{probability:.1f}_n{n}_id{idx:03d}".replace(".", "_")
        write_dimacs(output_dir / f"{name}.col", graph)

        colors = sample_color_count(probability, n)
        readme_lines.append(f"{name} {colors}\n")

    with open(output_dir / "ReadMe.txt", "w", encoding="utf-8") as f:
        f.writelines(readme_lines)

    p_text = ", ".join(f"p={p:.1f}:{c}" for p, c in sorted(p_summary.items()))
    r_text = ", ".join(
        f"{node_ranges[i][0][0]}-{node_ranges[i][0][1]}:{r_summary[i]}" for i in range(len(node_ranges))
    )
    print(f"Generated {len(readme_lines)} graphs in {output_dir}")
    print(f"Probability counts -> {p_text}")
    print(f"Node-range counts -> {r_text}")


if __name__ == "__main__":
    main()
