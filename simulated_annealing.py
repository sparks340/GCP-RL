import numpy as np
import random
from typing import List, Tuple

class SimulatedAnnealingSolver:
    def __init__(
        self,
        adj_list: List[List[int]],
        k: int,
        max_iterations: int = 10000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01
    ):
        self.adj_list = adj_list
        self.k = k
        self.n = len(adj_list)
        self.max_iterations = max_iterations
        self.solution = None
        self.best_solution = None
        self.best_score = float('inf')
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def calculate_conflicts(self, solution: List[int]) -> int:
        """计算当前解的冲突数"""
        conflicts = 0
        for node in range(self.n):
            for neighbor in self.adj_list[node]:
                if solution[node] == solution[neighbor]:
                    conflicts += 1
        return conflicts // 2  # 每个冲突被计算了两次

    def generate_initial_solution(self) -> List[int]:
        """生成初始解"""
        return [random.randint(0, self.k - 1) for _ in range(self.n)]

    def get_neighbor_solution(self, current_solution: List[int]) -> List[int]:
        """生成邻域解"""
        neighbor = current_solution.copy()
        node = random.randint(0, self.n - 1)
        new_color = random.randint(0, self.k - 1)
        while new_color == neighbor[node]:
            new_color = random.randint(0, self.k - 1)
        neighbor[node] = new_color
        return neighbor

    def accept_solution(self, current_score: int, new_score: int, temperature: float) -> bool:
        """根据Metropolis准则决定是否接受新解"""
        if new_score <= current_score:
            return True
        delta = new_score - current_score
        probability = np.exp(-delta / temperature)
        return random.random() < probability

    def solve(self) -> Tuple[List[int], int]:
        """运行模拟退火算法求解图着色问题"""
        self.solution = self.generate_initial_solution()
        self.best_solution = self.solution.copy()
        current_score = self.calculate_conflicts(self.solution)
        self.best_score = current_score
        
        temperature = self.initial_temp
        iteration = 0
        
        while iteration < self.max_iterations and temperature > self.min_temp:
            neighbor_solution = self.get_neighbor_solution(self.solution)
            neighbor_score = self.calculate_conflicts(neighbor_solution)
            
            if self.accept_solution(current_score, neighbor_score, temperature):
                self.solution = neighbor_solution
                current_score = neighbor_score
                
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_solution = self.solution.copy()
                    
                    if current_score == 0:  # 找到可行解
                        break
            
            temperature *= self.cooling_rate
            iteration += 1
        
        return self.best_solution, self.best_score 