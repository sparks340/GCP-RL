import argparse
import sys
import time

import gymnasium as gym
import networkx as nx
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic

from gcp_env.gcp_env import GcpEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于强化学习的图着色训练器（支持SA/Tabu与GNN消融）")
    parser.add_argument("output", type=str, help="策略输出文件路径")
    parser.add_argument("--input", type=str, default=None, help="策略输入文件路径")
    parser.add_argument("-I", "--max_steps_RL", type=int, default=300, help="每个episode的RL最大步数")
    parser.add_argument("-L", "--max-steps", type=int, dest="max_steps", default=320, help="每个episode的最大步数")

    parser.add_argument("-S", "--sa-iters", type=int, default=500000, help="模拟退火迭代次数")
    parser.add_argument("-T", "--initial-temp", type=float, default=500.0, help="模拟退火初始温度")
    parser.add_argument("-C", "--cooling-rate", type=float, default=0.995, help="模拟退火冷却率")
    parser.add_argument("-M", "--min-temp", type=float, default=0.01, help="模拟退火最小温度")

    parser.add_argument("--tabu-iters", type=int, default=5000, help="禁忌搜索迭代次数")
    parser.add_argument("--tabu-tenure", type=int, default=20, help="禁忌长度")
    parser.add_argument("--search-algorithm", choices=["sa", "tabu"], default="sa", help="RL后的局部搜索算法")

    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn", help="策略网络结构（用于GNN消融）")

    parser.add_argument("-E", "--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("-N", "--nodes", type=int, default=250, help="训练图节点数")
    parser.add_argument("-P", "--probability", type=float, default=0.5, help="随机图边概率")
    parser.add_argument("-K", "--colors", type=int, default=24, help="颜色数")
    parser.add_argument("-B", "--beta", type=float, default=0.2, help="奖励中的局部搜索权重")
    args = parser.parse_args()

    gym.register(id="GcpEnvMaxIters-v0", entry_point="gcp_env.gcp_env:GcpEnv", max_episode_steps=args.max_steps)

    nodes, probability, colors = args.nodes, args.probability, args.colors

    def build_env():
        return GcpEnv(
            graph=nx.gnp_random_graph(nodes, probability),
            k=colors,
            sa_iters=args.sa_iters,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            min_temp=args.min_temp,
            tabu_iters=args.tabu_iters,
            tabu_tenure=args.tabu_tenure,
            search_algorithm=args.search_algorithm,
            beta=args.beta,
            max_episode_steps_RL=args.max_steps_RL,
            max_episode_steps=args.max_steps,
        )

    env = build_env()
    train_envs = SubprocVectorEnv([build_env for _ in range(1)])
    test_envs = SubprocVectorEnv([build_env for _ in range(1)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gnn = args.model_type == "gnn"

    actor = ActorNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    critic = CriticNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    policy = GCPPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        nodes=nodes,
        k=colors,
        action_space=env.action_space,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=True,
    )

    if args.input:
        print(f"加载策略: {args.input}")
        policy.load_state_dict(torch.load(args.input, map_location=device))
        sys.stdout.flush()

    replay_buffer = VectorReplayBuffer(20000, len(train_envs))
    train_collector = Collector(policy, train_envs, replay_buffer)
    test_collector = Collector(policy, test_envs)

    log_path = (
        f"./log/gcp_train_{args.nodes}nodes_{args.colors}colors_"
        f"{args.epochs}epochs_{args.model_type}_{args.search_algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "参数配置",
        f"nodes={args.nodes}\ncolors={args.colors}\nepochs={args.epochs}\n"
        f"model_type={args.model_type}\nsearch_algorithm={args.search_algorithm}\n"
        f"sa_iters={args.sa_iters}\ntabu_iters={args.tabu_iters}\nbeta={args.beta}",
    )
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=1000,
        repeat_per_collect=5,
        episode_per_test=2,
        batch_size=128,
        step_per_collect=500,
        logger=logger,
    )

    print("开始训练...")
    sys.stdout.flush()
    result = trainer.run()
    print(f"训练完成: {result}")
    torch.save(policy.state_dict(), args.output)
    writer.close()
