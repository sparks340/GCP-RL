import os
import random
import math
from typing import List, Tuple, Optional


class SimulatedAnnealingSolver:
    def __init__(
        self,
        adj_list: List[List[int]],
        k: int,
        max_iterations: int = 10000,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.99,
        min_temp: float = 1e-8,
    ):
        self.adj_list = adj_list
        self.k = k
        self.n = len(adj_list)
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

        self.solution: List[int] = [0] * self.n
        self.best_solution: List[int] = [0] * self.n
        self.current_score: int = 0
        self.best_score: int = float("inf")

        # neighbor_color_counts[i][c] = 顶点 i 的邻居中，颜色为 c 的数量
        self.neighbor_color_counts = [[0] * self.k for _ in range(self.n)]

    def generate_initial_solution(self) -> List[int]:
        return [random.randint(0, self.k - 1) for _ in range(self.n)]

    def initialize(self, initial_solution: Optional[List[int]] = None) -> None:
        if initial_solution is not None:
            self.solution = initial_solution.copy()
        else:
            self.solution = self.generate_initial_solution()

        self.best_solution = self.solution.copy()

        # 初始化邻接颜色统计表
        self.neighbor_color_counts = [[0] * self.k for _ in range(self.n)]
        for node in range(self.n):
            for neighbor in self.adj_list[node]:
                neighbor_color_counts_of_node = self.neighbor_color_counts[node]
                neighbor_color = self.solution[neighbor]
                neighbor_color_counts_of_node[neighbor_color] += 1

        self.current_score = self.calculate_conflicts_from_table()
        self.best_score = self.current_score

    def calculate_conflicts(self, solution: List[int]) -> int:
        conflicts = 0
        for node in range(self.n):
            for neighbor in self.adj_list[node]:
                if solution[node] == solution[neighbor]:
                    conflicts += 1
        return conflicts // 2

    def calculate_conflicts_from_table(self) -> int:
        conflicts = 0
        for node in range(self.n):
            conflicts += self.neighbor_color_counts[node][self.solution[node]]
        return conflicts // 2

    def select_move(self) -> Tuple[int, int]:
        """
        参考 KColorRLSA：选择一个单点改色动作
        这里优先从冲突点中选；若没有冲突点，则随机选点
        """
        conflicting_nodes = [
            node
            for node in range(self.n)
            if self.neighbor_color_counts[node][self.solution[node]] > 0
        ]

        if conflicting_nodes:
            vertex = random.choice(conflicting_nodes)
        else:
            vertex = random.randint(0, self.n - 1)

        old_color = self.solution[vertex]
        new_color = random.randint(0, self.k - 1)
        while new_color == old_color:
            new_color = random.randint(0, self.k - 1)

        return vertex, new_color

    def calculate_delta_conflicts(self, vertex: int, new_color: int) -> int:
        """
        增量计算：
        把 vertex 从 old_color 改成 new_color 后，冲突变化量为
        邻居中 new_color 的数量 - 邻居中 old_color 的数量
        """
        old_color = self.solution[vertex]
        delta = (
            self.neighbor_color_counts[vertex][new_color]
            - self.neighbor_color_counts[vertex][old_color]
        )
        return delta

    def accept_move(self, delta: int, temperature: float) -> bool:
        """
        Metropolis 准则
        """
        if delta <= 0:
            return True
        probability = math.exp(-delta / temperature)
        return random.random() < probability

    def make_move(self, vertex: int, new_color: int, delta: int) -> None:
        """
        执行 move，并增量更新：
        1. 当前解
        2. 当前冲突数
        3. 邻接颜色统计表
        """
        old_color = self.solution[vertex]
        self.solution[vertex] = new_color
        self.current_score += delta

        # 更新所有邻居的邻接颜色统计
        for neighbor in self.adj_list[vertex]:
            self.neighbor_color_counts[neighbor][old_color] -= 1
            self.neighbor_color_counts[neighbor][new_color] += 1

        if self.current_score < self.best_score:
            self.best_score = self.current_score
            self.best_solution = self.solution.copy()

    def solve(self, initial_solution: Optional[List[int]] = None) -> Tuple[List[int], int]:
        self.initialize(initial_solution)

        temperature = self.initial_temp
        iteration = 0

        while iteration < self.max_iterations and temperature > self.min_temp:
            if self.best_score == 0:
                break

            vertex, new_color = self.select_move()
            delta = self.calculate_delta_conflicts(vertex, new_color)

            if self.accept_move(delta, temperature):
                self.make_move(vertex, new_color, delta)

            temperature *= self.cooling_rate
            iteration += 1

        return self.best_solution, self.best_score


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

    graph_file = r"C:\Users\JZSK\Desktop\GCP-RL\data\DSJC125\DSJC125.1.col"
    k = 5

    adj_list = read_col_graph(graph_file)

    print(f"Graph file: {graph_file}")
    print(f"Number of vertices: {len(adj_list)}")
    print(f"Number of edges: {sum(len(nei) for nei in adj_list) // 2}")
    print(f"Number of colors k: {k}")

    solver = SimulatedAnnealingSolver(
        adj_list=adj_list,
        k=k,
        max_iterations=10000,
        initial_temp=1000.0,
        cooling_rate=0.99,
        min_temp=1e-8,
    )

    best_solution, best_score = solver.solve()

    print(f"Best conflict count: {best_score}")
    print(f"Best solution: {best_solution}")