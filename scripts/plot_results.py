import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_records(input_dir: Path):
    records = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        conflicts = data.get("conflicts")
        if conflicts is None:
            continue

        runtime = data.get("runtime_sec")
        ablation = str(data.get("ablation", "unknown"))
        model_type = str(data.get("model_type", "unknown"))
        search_algorithm = str(data.get("search_algorithm", "unknown"))

        try:
            conflicts = float(conflicts)
        except (TypeError, ValueError):
            continue

        try:
            runtime = float(runtime) if runtime is not None else np.nan
        except (TypeError, ValueError):
            runtime = np.nan

        records.append(
            {
                "file": path.name,
                "group": f"{ablation}+{model_type}+{search_algorithm}",
                "ablation": ablation,
                "model_type": model_type,
                "search_algorithm": search_algorithm,
                "final_conflicts": conflicts,
                "runtime_sec": runtime,
            }
        )
    return records


def group_order(records):
    groups = sorted({r["group"] for r in records})
    return groups


def pareto_front(points):
    if not points:
        return []
    # Minimize runtime and conflicts.
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_conflict = float("inf")
    for runtime, conflict, group in sorted_points:
        if conflict < best_conflict:
            front.append((runtime, conflict, group))
            best_conflict = conflict
    return front


def save_summary_csv(records, groups, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "n",
                "mean_final_conflicts",
                "median_final_conflicts",
                "success_rate",
                "mean_runtime_sec",
            ]
        )
        for g in groups:
            items = [r for r in records if r["group"] == g]
            conflicts = np.array([r["final_conflicts"] for r in items], dtype=float)
            runtimes = np.array([r["runtime_sec"] for r in items], dtype=float)
            success_rate = float(np.mean(conflicts == 0)) if len(conflicts) else 0.0
            mean_runtime = float(np.nanmean(runtimes)) if np.any(~np.isnan(runtimes)) else np.nan
            writer.writerow(
                [
                    g,
                    len(items),
                    float(np.mean(conflicts)) if len(conflicts) else np.nan,
                    float(np.median(conflicts)) if len(conflicts) else np.nan,
                    success_rate,
                    mean_runtime,
                ]
            )


def plot(records, output_png: Path):
    groups = group_order(records)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1) Boxplot: final_conflicts by group.
    ax1 = axes[0]
    box_data = [np.array([r["final_conflicts"] for r in records if r["group"] == g], dtype=float) for g in groups]
    ax1.boxplot(box_data, labels=groups, showfliers=True)
    ax1.set_title("Final Conflicts by Group")
    ax1.set_ylabel("Final Conflicts")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.3)

    # 2) Scatter: runtime vs conflicts + Pareto front.
    ax2 = axes[1]
    cmap = plt.cm.get_cmap("tab10", max(len(groups), 1))
    all_points = []
    for i, g in enumerate(groups):
        items = [r for r in records if r["group"] == g and not np.isnan(r["runtime_sec"])]
        if not items:
            continue
        x = [r["runtime_sec"] for r in items]
        y = [r["final_conflicts"] for r in items]
        ax2.scatter(x, y, s=35, alpha=0.8, color=cmap(i), label=g)
        all_points.extend([(r["runtime_sec"], r["final_conflicts"], g) for r in items])

    front = pareto_front(all_points)
    if front:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax2.plot(fx, fy, color="black", linewidth=2, linestyle="--", label="pareto_front")

    ax2.set_title("Runtime vs Final Conflicts")
    ax2.set_xlabel("Runtime (s)")
    ax2.set_ylabel("Final Conflicts")
    ax2.grid(alpha=0.3)
    if all_points:
        ax2.legend(fontsize=8)

    # 3) Success rate bar: conflicts == 0 by group.
    ax3 = axes[2]
    success_rates = []
    for g in groups:
        conflicts = np.array([r["final_conflicts"] for r in records if r["group"] == g], dtype=float)
        rate = float(np.mean(conflicts == 0)) if len(conflicts) else 0.0
        success_rates.append(rate)

    x_pos = np.arange(len(groups))
    bars = ax3.bar(x_pos, success_rates, color=[cmap(i) for i in range(len(groups))])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(groups, rotation=30)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Success Rate")
    ax3.set_title("Success Rate (conflicts == 0)")
    ax3.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, success_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close(fig)


def print_summary(records, groups):
    print("Group summary:")
    for g in groups:
        items = [r for r in records if r["group"] == g]
        conflicts = np.array([r["final_conflicts"] for r in items], dtype=float)
        runtimes = np.array([r["runtime_sec"] for r in items], dtype=float)
        success_rate = float(np.mean(conflicts == 0)) if len(conflicts) else 0.0
        mean_runtime = float(np.nanmean(runtimes)) if np.any(~np.isnan(runtimes)) else float("nan")
        print(
            f"- {g}: n={len(items)}, mean_conflicts={np.mean(conflicts):.3f}, "
            f"median_conflicts={np.median(conflicts):.3f}, success_rate={success_rate:.3f}, "
            f"mean_runtime={mean_runtime:.3f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from results/*.json")
    parser.add_argument("--input-dir", type=str, default="results", help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default="results/plots/results_dashboard.png", help="Output image path")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/plots/results_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    records = load_records(input_dir)
    if not records:
        raise SystemExit(f"No valid JSON records found in: {input_dir}")

    groups = group_order(records)
    plot(records, Path(args.output))
    save_summary_csv(records, groups, Path(args.summary_csv))
    print_summary(records, groups)
    print(f"Saved plot to: {args.output}")
    print(f"Saved summary CSV to: {args.summary_csv}")


if __name__ == "__main__":
    main()
