import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
import random
from simulated_annealing import SimulatedAnnealingSolver
from tabu_search import TabuSearchSolver


class GcpEnv(gym.Env):
    metadata = {"render_modes": ["human", "file"], "render_fps": 60}

    def __init__(
        self,
        graph,
        k,
        sa_iters=50000,
        initial_temp=500.0,
        cooling_rate=0.995,
        min_temp=0.01,
        tabu_iters=5000,
        tabu_tenure=20,
        search_algorithm="sa",
        beta=0.2,
        render_mode=None,
        base_filename=None,
        sa_init=False,
        max_episode_steps_RL=300,
        max_episode_steps=1000,
    ):
        super().__init__()
        self._graph = graph
        self._k = k
        self._adj_matrix = nx.to_numpy_array(graph, dtype=np.int32)
        self._adj_list = [list(self._graph.neighbors(node)) for node in range(len(self._graph))]

        edges = np.array(list(self._graph.edges), dtype=np.int64)
        if len(edges) == 0:
            self._edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            reverse_edges = edges[:, ::-1]
            bidirectional_edges = np.vstack([edges, reverse_edges])
            self._edge_index = bidirectional_edges.T

        self._max_episode_steps_RL = max_episode_steps_RL
        self._max_episode_steps = max_episode_steps

        n = len(self._graph)
        self.observation_space = spaces.Dict(
            {
                "edge_index": spaces.Sequence(spaces.Box(0, n - 1, shape=(2,), dtype=np.int32)),
                "node_features": spaces.Box(low=-float("inf"), high=float("inf"), shape=(n, 3), dtype=np.float32),
                "col_features": spaces.Box(low=-float("inf"), high=float("inf"), shape=(n, k, 3), dtype=np.float32),
                "k": spaces.Discrete(k + 1),
            }
        )

        self.action_space = spaces.Discrete(n * k)

        self._solution = None
        self._best_solution = None
        self._best_score = float("inf")

        self._sa_solver = SimulatedAnnealingSolver(
            self._adj_list,
            k,
            max_iterations=sa_iters,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
        )
        self._tabu_solver = TabuSearchSolver(
            self._adj_list,
            k,
            max_iterations=tabu_iters,
            tabu_tenure=tabu_tenure,
        )

        self._search_algorithm = search_algorithm
        self._beta = beta
        self._sa_init = sa_init
        self.render_mode = render_mode
        self._base_filename = base_filename
        self._episode = 0
        self._step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._sa_init and self._search_algorithm == "sa":
            self._solution, _ = self._sa_solver.solve()
        else:
            self._solution = [random.randint(0, self._k - 1) for _ in range(len(self._graph))]

        self._best_solution = self._solution.copy()
        self._best_score = self._calculate_conflicts()
        self._episode += 1
        self._step = 0

        return self._get_obs(), {}

    def _calculate_conflicts(self):
        conflicts = 0
        for i in range(len(self._graph)):
            for j in self._adj_list[i]:
                if self._solution[i] == self._solution[j]:
                    conflicts += 1
        return conflicts // 2

    def _get_obs(self):
        n = len(self._graph)
        node_features = np.zeros((n, 3), dtype=np.float32)
        col_features = np.zeros((n, self._k, 3), dtype=np.float32)

        for i in range(n):
            conflicts = sum(1 for j in self._adj_list[i] if self._solution[i] == self._solution[j])
            node_features[i] = [
                conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,
                len(self._adj_list[i]) / n,
                self._solution[i] / self._k,
            ]

        for i in range(n):
            for c in range(self._k):
                conflicts = sum(1 for j in self._adj_list[i] if self._solution[j] == c)
                col_features[i, c] = [
                    conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,
                    1 if self._solution[i] == c else 0,
                    c / self._k,
                ]

        return {
            "edge_index": self._edge_index,
            "node_features": node_features,
            "col_features": col_features,
            "k": self._k,
        }

    def _run_local_search(self):
        if self._search_algorithm == "sa":
            self._sa_solver.solution = self._solution.copy()
            return self._sa_solver.solve()
        if self._search_algorithm == "tabu":
            return self._tabu_solver.solve(initial_solution=self._solution)
        return self._solution.copy(), self._calculate_conflicts()

    def step(self, action):
        self._step += 1

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 2:
                node, color = action
            else:
                action = action[0] if isinstance(action, np.ndarray) else action
                node = action // self._k
                color = action % self._k
        else:
            node = action // self._k
            color = action % self._k

        node = int(node)
        color = int(color)

        old_score = self._calculate_conflicts()
        self._solution[node] = color
        new_score = self._calculate_conflicts()

        immediate_reward = old_score - new_score

        if new_score < self._best_score:
            self._best_score = new_score
            self._best_solution = self._solution.copy()

        search_reward = 0
        if self._step >= self._max_episode_steps_RL:
            old_search_score = self._calculate_conflicts()
            self._solution, _ = self._run_local_search()
            new_search_score = self._calculate_conflicts()
            search_reward = old_search_score - new_search_score
            if new_search_score < self._best_score:
                self._best_score = new_search_score
                self._best_solution = self._solution.copy()

        reward = immediate_reward + self._beta * search_reward

        terminated = new_score == 0
        truncated = self._step >= self._max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, {
            "best_solution": self._best_solution,
            "conflicts": new_score,
            "best_conflicts": self._best_score,
        }

    def get_graph(self):
        return self._graph

    def get_solution(self):
        return self._best_solution.copy() if self._best_solution is not None else None

    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt

        plt.clf()
        pos = nx.spring_layout(self._graph)
        colors = [f"C{c}" for c in self._solution]
        nx.draw(self._graph, pos, node_color=colors, with_labels=True)

        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "file" and self._base_filename:
            filename = f"{self._base_filename}_{self._episode}_{self._step}.png"
            plt.savefig(filename)
