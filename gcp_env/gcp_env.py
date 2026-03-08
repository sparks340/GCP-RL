import random

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from simulated_annealing import SimulatedAnnealingSolver
from tabu_search import TabuSearchSolver


class GcpEnv(gym.Env):
    metadata = {"render_modes": ["human", "file"], "render_fps": 60}

    def __init__(
        self,
        graph=None,
        k=24,
        sa_iters=50000,
        initial_temp=500.0,
        cooling_rate=0.995,
        min_temp=0.01,
        tabu_iters=5000,
        tabu_tenure=20,
        search_algorithm="sa",
        beta=0.1,
        stagnation_penalty=0.001,
        render_mode=None,
        base_filename=None,
        sa_init=False,
        max_episode_steps_RL=300,
        graph_sampler=None,
        resample_on_reset=False,
        max_nodes=None,
        max_colors=None,
        k_sampler=None,
    ):
        super().__init__()
        if graph is None and graph_sampler is None:
            raise ValueError("Either graph or graph_sampler must be provided")

        self._graph_sampler = graph_sampler
        self._resample_on_reset = resample_on_reset and graph_sampler is not None
        self._k_sampler = k_sampler

        self._k = k
        self._max_episode_steps_RL = max_episode_steps_RL

        self._n = None
        self._graph = None
        self._adj_matrix = None
        self._adj_list = None
        self._edge_index = None

        initial_graph = graph if graph is not None else graph_sampler()

        self._max_nodes = max_nodes if max_nodes is not None else len(initial_graph)
        self._max_colors = max_colors if max_colors is not None else k
        if self._max_nodes <= 0 or self._max_colors <= 0:
            raise ValueError("max_nodes and max_colors must be positive")

        if self._k > self._max_colors:
            raise ValueError(f"k ({self._k}) cannot be greater than max_colors ({self._max_colors})")

        self._max_directed_edges = self._max_nodes * (self._max_nodes - 1)

        self._set_graph(initial_graph)
        self._configure_k_from_graph()

        self.observation_space = spaces.Dict(
            {
                "edge_index": spaces.Box(
                    low=-1,
                    high=self._max_nodes - 1,
                    shape=(2, self._max_directed_edges),
                    dtype=np.int32,
                ),
                "node_features": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self._max_nodes, 3),
                    dtype=np.float32,
                ),
                "col_features": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self._max_nodes, self._max_colors, 3),
                    dtype=np.float32,
                ),
                "k": spaces.Discrete(self._max_colors + 1),
                "n": spaces.Discrete(self._max_nodes + 1),
            }
        )

        self.action_space = spaces.Discrete(self._max_nodes * self._max_colors)

        self._solution = None
        self._best_solution = None
        self._best_score = float("inf")

        self._sa_solver = SimulatedAnnealingSolver(
            self._adj_list,
            self._k,
            max_iterations=sa_iters,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
        )
        self._tabu_solver = TabuSearchSolver(
            self._adj_list,
            self._k,
            max_iterations=tabu_iters,
            tabu_tenure=tabu_tenure,
        )

        self._search_algorithm = search_algorithm
        self._beta = beta
        self._stagnation_penalty = stagnation_penalty
        self._sa_init = sa_init
        self.render_mode = render_mode
        self._base_filename = base_filename
        self._episode = 0
        self._step = 0

    def _configure_k_from_graph(self):
        if self._k_sampler is not None:
            try:
                sampled_k = int(self._k_sampler(self._n, self._graph))
            except TypeError:
                sampled_k = int(self._k_sampler(self._n))
            self._k = max(1, sampled_k)
        if self._k > self._max_colors:
            raise ValueError(f"Sampled k ({self._k}) exceeds max_colors ({self._max_colors})")

        if hasattr(self, "_sa_solver") and self._sa_solver is not None:
            self._sa_solver.k = self._k
            self._sa_solver.n = self._n
        if hasattr(self, "_tabu_solver") and self._tabu_solver is not None:
            self._tabu_solver.k = self._k
            self._tabu_solver.n = self._n

    def _set_graph(self, graph):
        n = len(graph)
        if n > self._max_nodes:
            raise ValueError(f"Graph has {n} nodes, which exceeds max_nodes={self._max_nodes}")

        self._n = n
        self._graph = graph
        self._adj_matrix = nx.to_numpy_array(graph, dtype=np.int32)
        self._adj_list = [list(self._graph.neighbors(node)) for node in range(n)]

        edges = np.array(list(self._graph.edges), dtype=np.int64)
        if len(edges) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            reverse_edges = edges[:, ::-1]
            bidirectional_edges = np.vstack([edges, reverse_edges])
            edge_index = bidirectional_edges.T

        if edge_index.shape[1] > self._max_directed_edges:
            raise ValueError(
                f"Graph has {edge_index.shape[1]} directed edges, which exceeds max_directed_edges={self._max_directed_edges}"
            )

        self._edge_index = np.full((2, self._max_directed_edges), -1, dtype=np.int64)
        self._edge_index[:, : edge_index.shape[1]] = edge_index

        if hasattr(self, "_sa_solver") and self._sa_solver is not None:
            self._sa_solver.adj_list = self._adj_list
            self._sa_solver.n = n
        if hasattr(self, "_tabu_solver") and self._tabu_solver is not None:
            self._tabu_solver.adj_list = self._adj_list
            self._tabu_solver.n = n

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._resample_on_reset and self._graph_sampler is not None:
            self._set_graph(self._graph_sampler())

        self._configure_k_from_graph()

        if self._sa_init and self._search_algorithm == "sa":
            self._solution, _ = self._sa_solver.solve()
        else:
            self._solution = [random.randint(0, self._k - 1) for _ in range(self._n)]

        self._best_solution = self._solution.copy()
        self._best_score = self._calculate_conflicts()
        self._episode += 1
        self._step = 0

        return self._get_obs(), {}

    def _calculate_conflicts(self):
        conflicts = 0
        for i in range(self._n):
            for j in self._adj_list[i]:
                if self._solution[i] == self._solution[j]:
                    conflicts += 1
        return conflicts // 2

    def _reward_normalizer(self):
        edge_count = self._graph.number_of_edges()
        return edge_count if edge_count > 0 else 1

    def _get_obs(self):
        node_features = np.zeros((self._max_nodes, 3), dtype=np.float32)
        col_features = np.zeros((self._max_nodes, self._max_colors, 3), dtype=np.float32)

        for i in range(self._n):
            conflicts = sum(1 for j in self._adj_list[i] if self._solution[i] == self._solution[j])
            node_features[i] = [
                conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,
                len(self._adj_list[i]) / self._n,
                self._solution[i] / self._k,
            ]

        for i in range(self._n):
            for c in range(self._k):
                conflicts = sum(1 for j in self._adj_list[i] if self._solution[j] == c)
                col_features[i, c] = [
                    conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,
                    1 if self._solution[i] == c else 0,
                    c / self._k,
                ]

        return {
            "edge_index": self._edge_index.astype(np.int32, copy=False),
            "node_features": node_features,
            "col_features": col_features,
            "k": self._k,
            "n": self._n,
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

        action_k = self._max_colors
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 2:
                node, color = action
            else:
                action = action[0] if isinstance(action, np.ndarray) else action
                node = action // action_k
                color = action % action_k
        else:
            node = action // action_k
            color = action % action_k

        node = int(node)
        color = int(color)

        old_score = self._calculate_conflicts()
        reward_normalizer = self._reward_normalizer()

        if node >= self._n or color >= self._k:
            immediate_reward = -1.0 / reward_normalizer
            new_score = old_score
        else:
            self._solution[node] = color
            new_score = self._calculate_conflicts()
            immediate_reward = (old_score - new_score) / reward_normalizer
            if new_score == old_score:
                immediate_reward -= self._stagnation_penalty

            if new_score < self._best_score:
                self._best_score = new_score
                self._best_solution = self._solution.copy()

        search_reward = 0
        local_search_finished = False
        # Run local search exactly once after the RL budget is exhausted, then terminate the episode.
        if self._step == self._max_episode_steps_RL:
            old_search_score = self._calculate_conflicts()
            self._solution, _ = self._run_local_search()
            new_search_score = self._calculate_conflicts()
            search_reward = (old_search_score - new_search_score) / reward_normalizer
            local_search_finished = True
            if new_search_score < self._best_score:
                self._best_score = new_search_score
                self._best_solution = self._solution.copy()

        reward = immediate_reward + self._beta * search_reward

        final_score = self._calculate_conflicts()
        terminated = final_score == 0 or local_search_finished
        truncated = False

        padded_best_solution = np.full(self._max_nodes, -1, dtype=np.int32)
        if self._best_solution is not None:
            padded_best_solution[: len(self._best_solution)] = np.asarray(self._best_solution, dtype=np.int32)

        return self._get_obs(), float(reward), terminated, truncated, {
            "step": self._step,
            "best_solution": padded_best_solution,
            "conflicts": final_score,
            "best_conflicts": self._best_score,
            "immediate_reward": float(immediate_reward),
            "search_reward": float(search_reward),
            "total_reward": float(reward),
            "n": self._n,
            "k": self._k,
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
