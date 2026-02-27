from torch import nn
import torch
from tianshou.policy.modelfree.ppo import PPOPolicy
from typing import Any, Optional, Union, Type
from tianshou.data import Batch
from tianshou.data.buffer.base import ReplayBuffer
import numpy as np

from torch_geometric.nn import GCNConv  

class NodeNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = GCNConv(input_size, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = nn.Linear(16, 1)

    def forward(self, data_tuple):
        # x shape: (batch_size, num_nodes, input_size)
        # edge_index shape: (2, num_edges)

        x, edge_index = data_tuple
        
        # 处理图结构
        batch_size, num_nodes, _ = x.shape
        x = x.view(-1, x.size(-1))  # 
        edge_index = edge_index.expand(batch_size, *edge_index.shape).reshape(2, -1)
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = x.view(batch_size, num_nodes, -1)  
        return self.fc(x)  

class ColNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class ActorNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu"):
        super().__init__()
        self.node_model = nn.Sequential(
            NodeNetwork(node_features),
            nn.Softmax(dim=1),
        )
        self.col_model = nn.Sequential(
            ColNetwork(col_features),
            nn.Softmax(dim=2),
        )
        self.device = device

    def forward(self, obs, state=None, info={}):
        # 添加edge_index到观测数据中
        obs["node_features"] = torch.as_tensor(
            obs["node_features"], device=self.device, dtype=torch.float32
        )
        obs["col_features"] = torch.as_tensor(
            obs["col_features"], device=self.device, dtype=torch.float32
        )
        edge_index = torch.as_tensor(
            obs["edge_index"], device=self.device, dtype=torch.long
        )

        # 使用GNN处理节点特征
        node_probs = torch.squeeze(
            self.node_model((obs["node_features"], edge_index)), -1
        )
        col_probs = torch.squeeze(self.col_model(obs["col_features"]), -1)

        return (
            torch.flatten(
                torch.transpose(
                    (torch.transpose(col_probs, -1, -2) * node_probs[:, None]), -1, -2
                ),
                start_dim=1,
            ),
            state,
        )

class CriticNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu"):
        super().__init__()
        self.node_model = NodeNetwork(node_features)
        self.col_model = ColNetwork(col_features)
        self.fc = nn.Linear(25, 16)  # 修改输入维度为25（根据实际输入维度调整）
        self.out_layer = nn.Linear(16, 1)  # 最终输出层
        self.device = device

    def forward(self, obs, **kwargs):
        obs["node_features"] = torch.as_tensor(
            obs["node_features"], device=self.device, dtype=torch.float32
        )
        obs["col_features"] = torch.as_tensor(
            obs["col_features"], device=self.device, dtype=torch.float32
        )
        edge_index = torch.as_tensor(
            obs["edge_index"], device=self.device, dtype=torch.long
        )

        # 使用GNN处理节点特征
        node_features = self.node_model((obs["node_features"], edge_index))
        col_features = self.col_model(obs["col_features"])
        print('node_features.shape',node_features.shape,'col_features.shape', col_features.shape)
        
        # 调整维度并合并特征
        node_features = node_features.squeeze(-1)  # 移除最后一个维度
        col_features = col_features.squeeze(-1)  # 移除最后一个维度
        print('node_features.shape',node_features.shape,'col_features.shape', col_features.shape)
        
        # 分别计算平均值并确保维度一致
        node_avg = node_features.mean(1)  # 对节点维度取平均
        col_avg = col_features.mean(1)    # 对列维度取平均
        print('node_avg.shape',node_avg.shape,'col_avg.shape', col_avg.shape)
        
        # 确保两个张量都是2维的 [batch_size, features]
        if node_avg.dim() == 1:
            node_avg = node_avg.unsqueeze(-1)
        if col_avg.dim() == 1:
            col_avg = col_avg.unsqueeze(-1)
        print('node_avg.shape',node_avg.shape,'col_avg.shape', col_avg.shape)
            
        # 拼接特征
        combined = torch.cat([node_avg, col_avg], dim=1)  # 在特征维度上拼接
        print('combined.shape',combined.shape)
        
        # 通过额外的全连接层
        x = torch.relu(self.fc(combined))
        return self.out_layer(x)

class GCPPPOPolicy(PPOPolicy):
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        nodes: int,
        k: int,
        action_space: Any,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any
    ) -> None:
        self.k = k
        self.n = nodes
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            action_scaling=False,
            action_bound_method=None,
            discount_factor=0.99,
            eps_clip=eps_clip,
            vf_coef=0.5,
            ent_coef=0.01,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            **kwargs
        )

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        def mapper(x):
            node = x // self.k
            col = x % self.k
            return np.array([node, col])
        
        if isinstance(act, Batch):
            return Batch(act=np.array([mapper(x) for x in act.act]))
        return np.array([mapper(x) for x in act])

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return super().process_fn(batch, buffer, indices) 