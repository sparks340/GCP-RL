import os
import random
from typing import List, Tuple, Optional


class TabuSearchSolver:
    def __init__(
        self,
        adj_list: List[List[int]],
        k: int,
        max_iterations: int = 10000,
        tabu_tenure: int = 20,
    ):
        self.adj_list = adj_list
        self.k = k
        self.n = len(adj_list)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure

    def calculate_conflicts(self, solution: List[int]) -> int:
        conflicts = 0
        for node in range(self.n):
            for neighbor in self.adj_list[node]:
                if solution[node] == solution[neighbor]:
                    conflicts += 1
        return conflicts // 2

    def generate_initial_solution(self) -> List[int]:
        return [random.randint(0, self.k - 1) for _ in range(self.n)]

    def solve(self, initial_solution: Optional[List[int]] = None) -> Tuple[List[int], int]:
        current = (
            initial_solution.copy()
            if initial_solution is not None
            else self.generate_initial_solution()
        )

        # neighbor_color_counts[node][color]
        # 表示 node 的邻居中，颜色为 color 的节点个数
        neighbor_color_counts = [[0] * self.k for _ in range(self.n)]
        for node in range(self.n):
            for neighbor in self.adj_list[node]:
                neighbor_color_counts[node][current[neighbor]] += 1

        current_score = self.calculate_conflicts(current)
        best = current.copy()
        best_score = current_score

        # tabu_until[node][color] = 在该迭代数之前，node 不能被染成 color
        tabu_until = [[-1] * self.k for _ in range(self.n)]

        for it in range(self.max_iterations):
            if best_score == 0:
                break

            best_delta = float("inf")
            candidate_moves = []

            # 只考虑当前有冲突的点
            conflicting_nodes = [
                node for node in range(self.n)
                if neighbor_color_counts[node][current[node]] > 0
            ]

            if not conflicting_nodes:
                break

            for node in conflicting_nodes:
                old_color = current[node]

                for new_color in range(self.k):
                    if new_color == old_color:
                        continue

                    # 改色后的冲突变化量
                    delta = (
                        neighbor_color_counts[node][new_color]
                        - neighbor_color_counts[node][old_color]
                    )
                    cand_score = current_score + delta

                    # 禁忌判断
                    # 若当前迭代 it <= tabu_until[node][new_color]，则该 move 是 tabu
                    is_tabu = it <= tabu_until[node][new_color]

                    # aspiration: 候选解优于历史最优则允许破禁忌
                    if is_tabu and cand_score >= best_score:
                        continue

                    if delta < best_delta:
                        best_delta = delta
                        candidate_moves = [(node, new_color)]
                    elif delta == best_delta:
                        candidate_moves.append((node, new_color))

            if not candidate_moves:
                break

            # 并列最优里随机选一个
            node, new_color = random.choice(candidate_moves)
            old_color = current[node]

            # 执行 move
            current[node] = new_color
            current_score += best_delta

            # 更新邻居颜色统计表
            for neighbor in self.adj_list[node]:
                neighbor_color_counts[neighbor][old_color] -= 1
                neighbor_color_counts[neighbor][new_color] += 1

            # 关键：禁止短期内把 node 改回 old_color
            # 这里参考你的 KColor，加一点随机扰动
            random_offset = random.randint(0, 9)
            tabu_until[node][old_color] = it + self.tabu_tenure + random_offset

            if current_score < best_score:
                best_score = current_score
                best = current.copy()

        return best, best_score


def read_col_graph(file_path: str) -> List[List[int]]:
    """
    读取 DIMACS .col 图文件
    格式示例：
        c comment
        p edge 125 6961
        e 1 2
        e 1 3
    """
    n = 0
    edges = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("c"):
                continue

            parts = line.split()

            if parts[0] == "p":
                # p edge n m
                n = int(parts[2])
            elif parts[0] == "e":
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u != v:
                    edges.append((u, v))

    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    return adj_list


if __name__ == "__main__":

    graph_file = r"C:\Users\JZSK\Desktop\GCP-RL\data\DSJC125\DSJC125.5.col"
    k = 17

    adj_list = read_col_graph(graph_file)

    print(f"Graph file: {graph_file}")
    print(f"Number of vertices: {len(adj_list)}")
    print(f"Number of edges: {sum(len(nei) for nei in adj_list) // 2}")
    print(f"Number of colors k: {k}")

    solver = TabuSearchSolver(
        adj_list=adj_list,
        k=k,
        max_iterations=10000,
        tabu_tenure=20,
    )

    best_solution, best_score = solver.solve()

    print(f"Best conflict count: {best_score}")
    print(f"Best solution: {best_solution}")