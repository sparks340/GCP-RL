import argparse
import sys
import time

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic

from gcp_env.gcp_env import GcpEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy


def run_eval_episode(eval_env, policy):
    obs, _ = eval_env.reset()
    total_reward = 0.0
    history = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        batched_obs = {
            "edge_index": np.expand_dims(obs["edge_index"], axis=0),
            "node_features": np.expand_dims(obs["node_features"], axis=0),
            "col_features": np.expand_dims(obs["col_features"], axis=0),
            "k": np.array([obs["k"]], dtype=np.int64),
        }
        batch = Batch(obs=batched_obs, info={})
        with torch.no_grad():
            result = policy(batch)

        logits = result.logits
        if isinstance(logits, torch.Tensor):
            action = int(torch.argmax(logits, dim=-1).detach().cpu().numpy()[0])
        else:
            action = int(np.asarray(result.act)[0])

        mapped_action = policy.map_action(np.array([action]))[0]
        obs, reward, terminated, truncated, info = eval_env.step(mapped_action)

        entropy = 0.0
        if hasattr(result, "dist") and result.dist is not None:
            entropy_tensor = result.dist.entropy()
            if isinstance(entropy_tensor, torch.Tensor):
                entropy = float(entropy_tensor.mean().item())
            else:
                entropy = float(np.mean(entropy_tensor))

        history.append(
            {
                "conflicts": float(info.get("conflicts", 0.0)),
                "best_conflicts": float(info.get("best_conflicts", 0.0)),
                "immediate_reward": float(info.get("immediate_reward", 0.0)),
                "search_reward": float(info.get("search_reward", 0.0)),
                "total_reward": float(info.get("total_reward", reward)),
                "entropy": entropy,
            }
        )
        total_reward += float(reward)

    return history, total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO trainer for graph coloring")
    parser.add_argument("output", type=str, help="Path to output policy file")
    parser.add_argument("--input", type=str, default=None, help="Path to input policy file")
    parser.add_argument("-I", "--max_steps_RL", type=int, default=300, help="Max RL steps in each episode")
    parser.add_argument("-L", "--max-steps", type=int, dest="max_steps", default=320, help="Max total steps in each episode")

    parser.add_argument("-S", "--sa-iters", type=int, default=500000, help="SA iterations")
    parser.add_argument("-T", "--initial-temp", type=float, default=500.0, help="SA initial temperature")
    parser.add_argument("-C", "--cooling-rate", type=float, default=0.995, help="SA cooling rate")
    parser.add_argument("-M", "--min-temp", type=float, default=0.01, help="SA minimum temperature")

    parser.add_argument("--tabu-iters", type=int, default=5000, help="Tabu search iterations")
    parser.add_argument("--tabu-tenure", type=int, default=20, help="Tabu tenure")
    parser.add_argument("--search-algorithm", choices=["sa", "tabu"], default="sa", help="Local search algorithm")

    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn", help="Policy network type")

    parser.add_argument("-E", "--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--train-env-num", type=int, default=8, help="Number of parallel training environments")
    parser.add_argument("--test-env-num", type=int, default=4, help="Number of parallel test environments")
    parser.add_argument("--step-per-epoch", type=int, default=5000, help="Environment steps collected per epoch")
    parser.add_argument("--step-per-collect", type=int, default=2000, help="Environment steps collected before each update")
    parser.add_argument("--repeat-per-collect", type=int, default=10, help="Gradient update rounds per collection")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--episode-per-test", type=int, default=8, help="Evaluation episodes per epoch")
    parser.add_argument("-N", "--nodes", type=int, default=250, help="Training graph node count")
    parser.add_argument("-P", "--probability", type=float, default=0.5, help="Erdos-Renyi edge probability")
    parser.add_argument("-K", "--colors", type=int, default=24, help="Number of colors")
    parser.add_argument("-B", "--beta", type=float, default=0.2, help="Local search reward weight")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--vf-coef", type=float, default=0.25, help="Value loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy bonus coefficient")
    args = parser.parse_args()

    gym.register(id="GcpEnvMaxIters-v0", entry_point="gcp_env.gcp_env:GcpEnv", max_episode_steps=args.max_steps)

    nodes, probability, colors = args.nodes, args.probability, args.colors
    base_graph = nx.gnp_random_graph(nodes, probability)

    def build_env():
        return GcpEnv(
            graph=base_graph.copy(),
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
    train_envs = DummyVectorEnv([build_env for _ in range(args.train_env_num)])
    test_envs = DummyVectorEnv([build_env for _ in range(args.test_env_num)])
    eval_env = build_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gnn = args.model_type == "gnn"

    actor = ActorNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    critic = CriticNetwork(3, 3, device=device, use_gnn=use_gnn).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    policy = GCPPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        nodes=nodes,
        k=colors,
        action_space=env.action_space,
        eps_clip=0.2,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=True,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
    )

    if args.input:
        print(f"Loading policy: {args.input}")
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
        "config",
        f"nodes={args.nodes}\ncolors={args.colors}\nepochs={args.epochs}\n"
        f"model_type={args.model_type}\nsearch_algorithm={args.search_algorithm}\n"
        f"sa_iters={args.sa_iters}\ntabu_iters={args.tabu_iters}\nbeta={args.beta}",
    )
    logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

    def test_fn(epoch, env_step):
        history, episode_reward = run_eval_episode(eval_env, policy)
        if not history:
            return

        conflicts = np.array([item["conflicts"] for item in history], dtype=np.float32)
        best_conflicts = np.array([item["best_conflicts"] for item in history], dtype=np.float32)
        immediate_rewards = np.array([item["immediate_reward"] for item in history], dtype=np.float32)
        search_rewards = np.array([item["search_reward"] for item in history], dtype=np.float32)
        total_rewards = np.array([item["total_reward"] for item in history], dtype=np.float32)
        entropies = np.array([item["entropy"] for item in history], dtype=np.float32)

        writer.add_scalar("eval/final_conflicts", float(conflicts[-1]), global_step=env_step)
        writer.add_scalar("eval/best_conflicts", float(best_conflicts.min()), global_step=env_step)
        writer.add_scalar("eval/episode_len", len(history), global_step=env_step)
        writer.add_scalar("eval/episode_reward", float(episode_reward), global_step=env_step)
        writer.add_scalar("eval/immediate_reward_mean", float(immediate_rewards.mean()), global_step=env_step)
        writer.add_scalar("eval/search_reward_mean", float(search_rewards.mean()), global_step=env_step)
        writer.add_scalar("eval/total_reward_mean", float(total_rewards.mean()), global_step=env_step)
        writer.add_scalar("eval/action_entropy_mean", float(entropies.mean()), global_step=env_step)

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        logger=logger,
        test_fn=test_fn,
    )

    print("Start training...")
    sys.stdout.flush()
    result = trainer.run()
    print(f"Training finished: {result}")
    torch.save(policy.state_dict(), args.output)
    writer.close()
