from typing import Any, Optional, Type, Union

import numpy as np
import torch
from torch import nn
from tianshou.data import Batch
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.policy.modelfree.ppo import PPOPolicy
from torch_geometric.nn import GCNConv


class GNNNodeNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = GCNConv(input_size, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = nn.Linear(16, 1)

    def forward(self, data_tuple):
        x, edge_index = data_tuple
        batch_size, num_nodes, _ = x.shape
        x = x.view(-1, x.size(-1))

        # Support both a shared graph edge index [2, E] and batched edge index [B, 2, E].
        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
        elif edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.expand(batch_size, -1, -1)
        elif edge_index.dim() != 3 or edge_index.size(0) != batch_size:
            raise ValueError(f"Unexpected edge_index shape: {tuple(edge_index.shape)}")

        # Offset node ids for each graph in the batch after flattening node features.
        offsets = torch.arange(batch_size, device=edge_index.device).view(batch_size, 1, 1) * num_nodes
        edge_index = edge_index + offsets
        edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = x.view(batch_size, num_nodes, -1)
        return self.fc(x)


class MLPNodeNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, data_tuple):
        x, _ = data_tuple
        return self.network(x)


class ColNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


class ActorNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu", use_gnn=True):
        super().__init__()
        self.node_model = GNNNodeNetwork(node_features) if use_gnn else MLPNodeNetwork(node_features)
        self.col_model = ColNetwork(col_features)
        self.device = device

    def forward(self, obs, state=None, info=None):
        obs["node_features"] = torch.as_tensor(obs["node_features"], device=self.device, dtype=torch.float32)
        obs["col_features"] = torch.as_tensor(obs["col_features"], device=self.device, dtype=torch.float32)
        edge_index = torch.as_tensor(obs["edge_index"], device=self.device, dtype=torch.long)

        node_logits = torch.squeeze(self.node_model((obs["node_features"], edge_index)), -1)
        col_logits = torch.squeeze(self.col_model(obs["col_features"]), -1)

        # Factorized action logits: (node, color) logit = node_logit + color_logit.
        # Keeping logits (instead of pre-softmax probabilities) stabilizes PPO updates
        # and allows torch.distributions.Categorical(logits=...) to handle normalization.
        joint_logits = node_logits.unsqueeze(-1) + col_logits
        logits = torch.flatten(joint_logits, start_dim=1)
        return logits, state


class CriticNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu", output_device=None, use_gnn=True):
        super().__init__()
        self.node_model = GNNNodeNetwork(node_features) if use_gnn else MLPNodeNetwork(node_features)
        self.col_model = ColNetwork(col_features)
        self.fc = nn.Linear(2, 16)
        self.out_layer = nn.Linear(16, 1)
        self.device = device
        self.output_device = output_device

    def forward(self, obs, **kwargs):
        obs["node_features"] = torch.as_tensor(obs["node_features"], device=self.device, dtype=torch.float32)
        obs["col_features"] = torch.as_tensor(obs["col_features"], device=self.device, dtype=torch.float32)
        edge_index = torch.as_tensor(obs["edge_index"], device=self.device, dtype=torch.long)

        node_features = self.node_model((obs["node_features"], edge_index)).squeeze(-1)
        col_features = self.col_model(obs["col_features"]).squeeze(-1)

        node_avg = node_features.mean(dim=tuple(range(1, node_features.dim())))
        col_avg = col_features.mean(dim=tuple(range(1, col_features.dim())))

        if node_avg.dim() == 1:
            node_avg = node_avg.unsqueeze(-1)
        if col_avg.dim() == 1:
            col_avg = col_avg.unsqueeze(-1)

        combined = torch.cat([node_avg, col_avg], dim=1)
        x = torch.relu(self.fc(combined))
        value = self.out_layer(x)
        if self.output_device is not None and value.device != torch.device(self.output_device):
            value = value.to(self.output_device)
        return value


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
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        **kwargs: Any,
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
            action_bound_method="",
            discount_factor=0.99,
            eps_clip=eps_clip,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            **kwargs,
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
