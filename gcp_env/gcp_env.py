import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealingSolver
from PIL import Image


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
        beta=0.2,
        render_mode=None,
        base_filename=None,
        sa_init=False,
        max_episode_steps_RL=300,
        max_episode_steps = 1000
    ):
        super().__init__()
        self._graph = graph
        self._k = k
        self._adj_matrix = nx.to_numpy_array(graph, dtype=np.int32)
        self._adj_list = [list(self._graph.neighbors(node)) for node in range(len(self._graph))]
        self._edge_list = self._graph.edges
        self._max_episode_steps_RL = max_episode_steps_RL
        self._max_episode_steps = max_episode_steps
        
        n = len(self._graph)
        # 修改观察空间为字典空间
        self.observation_space = spaces.Dict({
            "edge_index": spaces.Sequence(
                spaces.Box(0, n-1, shape=(2,), dtype=np.int32)
            ),
            "node_features": spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(n, 3),
                dtype=np.float32
            ),
            "col_features": spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(n, k, 3),
                dtype=np.float32
            ),
            "k": spaces.Discrete(k + 1)
        })
        
        # 修改动作空间为单一离散空间
        self.action_space = spaces.Discrete(n * k)
        
        self._solution = None
        self._best_solution = None
        self._best_score = float('inf')
        
        # 初始化模拟退火求解器
        self._sa_solver = SimulatedAnnealingSolver(
            self._adj_list,
            k,
            max_iterations=sa_iters,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp
        )
        
        self._beta = beta
        self._sa_init = sa_init
        self.render_mode = render_mode
        self._base_filename = base_filename
        self._episode = 0
        self._step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self._sa_init:
            # 使用模拟退火生成初始解
            self._solution, _ = self._sa_solver.solve()
        else:
            # 随机生成初始解
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
        
        # 计算节点特征
        for i in range(n):
            conflicts = sum(1 for j in self._adj_list[i] if self._solution[i] == self._solution[j])
            node_features[i] = [
                conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,  # 冲突率
                len(self._adj_list[i]) / n,  # 归一化度
                self._solution[i] / self._k,  # 归一化颜色
            ]
        
        # 计算颜色特征
        for i in range(n):
            for c in range(self._k):
                conflicts = sum(1 for j in self._adj_list[i] if self._solution[j] == c)
                col_features[i, c] = [
                    conflicts / len(self._adj_list[i]) if self._adj_list[i] else 0,  # 邻居中使用该颜色的比例
                    1 if self._solution[i] == c else 0,  # 是否使用该颜色
                    c / self._k,  # 归一化颜色索引
                ]
        
        return {
            "edge_index":self._edge_list,
            "node_features": node_features,
            "col_features": col_features,
            "k": self._k
        }
    
    def step(self, action):
        self._step += 1
        # print('step',self._step, end='\r')
        # print('step',self._step)
        # 处理动作
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
            
        # 确保是整数
        node = int(node)
        color = int(color)
        
        # 保存旧状态
        old_color = self._solution[node]
        old_score = self._calculate_conflicts()
        
        # 应用动作
        self._solution[node] = color
        
        # 计算新状态
        new_score = self._calculate_conflicts()
        
        # 计算即时奖励（负的冲突数变化）
        immediate_reward = old_score - new_score
        
        # 更新最佳解
        if new_score < self._best_score:
            self._best_score = new_score
            self._best_solution = self._solution.copy()
        
        # 如果达到最大步数，运行完整的模拟退火
        sa_reward = 0
        if self._step >= self._max_episode_steps_RL:
            old_sa_score = self._calculate_conflicts()
            self._sa_solver.solution = self._solution.copy()
            self._solution, _ = self._sa_solver.solve()
            new_sa_score = self._calculate_conflicts()
            sa_reward = old_sa_score - new_sa_score
            
            # 更新最佳解
            if new_sa_score < self._best_score:
                self._best_score = new_sa_score
                self._best_solution = self._solution.copy()
        
        # 计算总奖励
        reward = immediate_reward + self._beta * sa_reward
        
        # 检查是否结束
        terminated = new_score == 0  # 找到可行解
        truncated = self._step >= self._max_episode_steps   # 达到最大步数
        
        return self._get_obs(), float(reward), terminated, truncated, {
            'best_solution':self._best_solution,
            "conflicts": new_score,
            "best_conflicts": self._best_score
        }
    
    def get_graph(self):
        return self._graph
    
    def get_solution(self):
        return self._best_solution.copy() if self._best_solution is not None else None
    
    def render(self):
        if self.render_mode is None:
            return
        
        try:
            print(f"正在渲染图形，模式: {self.render_mode}")
            print(f"图信息: 节点数={len(self._graph)}, 当前解={self._solution}")
            
            plt.clf()
            pos = nx.spring_layout(self._graph)
            colors = [f"C{c}" for c in self._solution]
            nx.draw(self._graph, pos, node_color=colors, with_labels=True)
            
            if self.render_mode == "human":
                print("显示图形...")
                plt.show()
            elif self.render_mode == "file" and self._base_filename:
                filename = f"{self._base_filename}_{self._episode}_{self._step}.png"
                print(f"保存图形到文件: {filename}")
                plt.savefig(filename)
            
            return None
        except Exception as e:
            print(f"渲染时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None