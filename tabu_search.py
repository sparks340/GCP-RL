import random
from collections import deque
from typing import List, Tuple


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

    def solve(self, initial_solution: List[int] = None) -> Tuple[List[int], int]:
        current = (
            initial_solution.copy()
            if initial_solution is not None
            else self.generate_initial_solution()
        )
        current_score = self.calculate_conflicts(current)
        best = current.copy()
        best_score = current_score

        tabu_queue = deque(maxlen=self.tabu_tenure)
        tabu_set = set()

        for _ in range(self.max_iterations):
            if best_score == 0:
                break

            best_move = None
            best_move_score = float("inf")

            for node in range(self.n):
                old_color = current[node]
                for color in range(self.k):
                    if color == old_color:
                        continue
                    move = (node, color)

                    candidate = current.copy()
                    candidate[node] = color
                    cand_score = self.calculate_conflicts(candidate)

                    is_tabu = move in tabu_set
                    aspiration = cand_score < best_score
                    if is_tabu and not aspiration:
                        continue

                    if cand_score < best_move_score:
                        best_move_score = cand_score
                        best_move = move

            if best_move is None:
                break

            node, color = best_move
            current[node] = color
            current_score = best_move_score

            tabu_queue.append(best_move)
            tabu_set = set(tabu_queue)

            if current_score < best_score:
                best_score = current_score
                best = current.copy()

        return best, best_score
