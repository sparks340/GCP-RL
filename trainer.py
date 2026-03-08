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
            "n": np.array([obs["n"]], dtype=np.int64),
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


def build_graph_factory(nodes, probability, seed=None, fixed_edge_count=False):
    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    max_undirected_edges = nodes * (nodes - 1) // 2
    target_edges = int(round(probability * max_undirected_edges))
    target_edges = max(0, min(target_edges, max_undirected_edges))

    def sample_graph():
        graph_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        if fixed_edge_count:
            return nx.gnm_random_graph(nodes, target_edges, seed=graph_seed)
        return nx.gnp_random_graph(nodes, probability, seed=graph_seed)

    return sample_graph


PROBABILITY_COLOR_RULES = (
    (0.1, 25),
    (0.5, 7),
    (0.9, 3),
)


def build_variable_graph_factory(min_nodes, max_nodes, seed=None):
    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)

    def sample_graph():
        nodes = int(rng.integers(min_nodes, max_nodes + 1))
        rule_idx = int(rng.integers(0, len(PROBABILITY_COLOR_RULES)))
        probability, color_divisor = PROBABILITY_COLOR_RULES[rule_idx]
        graph_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        graph = nx.gnp_random_graph(nodes, probability, seed=graph_seed)
        graph.graph["edge_probability"] = float(probability)
        graph.graph["color_divisor"] = int(color_divisor)
        return graph

    return sample_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO trainer for graph coloring")
    parser.add_argument("output", type=str, help="Path to output policy file")
    parser.add_argument("--input", type=str, default=None, help="Path to input policy file")
    parser.add_argument("-I", "--max_steps_RL", type=int, default=300, help="Max RL steps in each episode")

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
    parser.add_argument("--random-nodes", action="store_true", help="Randomize node count every training episode")
    parser.add_argument("--min-nodes", type=int, default=60, help="Minimum node count when --random-nodes is enabled")
    parser.add_argument("--max-nodes", type=int, default=150, help="Maximum node count when --random-nodes is enabled")
    parser.add_argument("-B", "--beta", type=float, default=0.1, help="Local search reward weight")
    parser.add_argument("--stagnation-penalty", type=float, default=1e-4, help="Penalty applied on true no-op actions (same color, unchanged conflicts)")
    parser.add_argument("--reward-scale", type=float, default=20.0, help="Global multiplier applied after reward normalization")
    parser.add_argument("--actor-lr", type=float, default=5e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=2e-4, help="Critic learning rate")
    parser.add_argument("--lr", type=float, default=None, help="Deprecated unified learning rate; overrides both actor/critic rates if set")
    parser.add_argument("--vf-coef", type=float, default=0.25, help="Value loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy bonus coefficient")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device selection")
    parser.add_argument("--split-gpus", action="store_true", help="Split actor/critic across cuda:0 and cuda:1 when available")
    parser.add_argument("--actor-device", type=str, default=None, help="Override actor device, e.g. cuda:0")
    parser.add_argument("--critic-device", type=str, default=None, help="Override critic device, e.g. cuda:1")
    parser.add_argument(
        "--train-graph-mode",
        choices=["single", "multi"],
        default="single",
        help="single: fixed graph for all training episodes; multi: resample a new graph each reset (with fixed edge count)",
    )
    parser.add_argument(
        "--eval-graph-mode",
        choices=["single", "multi"],
        default="single",
        help="single: fixed evaluation graph(s); multi: resample new graph at each evaluation reset (with fixed edge count)",
    )
    parser.add_argument("--graph-seed", type=int, default=None, help="Random seed for graph generation")
    args = parser.parse_args()

    gym.register(id="GcpEnvMaxIters-v0", entry_point="gcp_env.gcp_env:GcpEnv", max_episode_steps=args.max_steps_RL)

    if args.random_nodes:
        if args.min_nodes <= 0 or args.max_nodes <= 0:
            raise ValueError("--min-nodes and --max-nodes must be positive")
        if args.min_nodes > args.max_nodes:
            raise ValueError(f"--min-nodes ({args.min_nodes}) cannot be greater than --max-nodes ({args.max_nodes})")

    if args.random_nodes and args.train_graph_mode != "multi":
        print("[Info] --random-nodes enabled: overriding --train-graph-mode to multi for per-episode resampling.")
        args.train_graph_mode = "multi"

    probability = args.probability
    use_fixed_edge_count = (args.train_graph_mode == "multi" or args.eval_graph_mode == "multi") and not args.random_nodes

    if args.random_nodes:
        min_nodes = args.min_nodes
        max_nodes = args.max_nodes
        max_colors = max(1, max_nodes // 3)
        graph_factory = build_variable_graph_factory(
            min_nodes,
            max_nodes,
            args.graph_seed,
        )
        base_graph = graph_factory()
        nodes = max_nodes
        colors = max_colors
    else:
        nodes = args.nodes
        colors = args.colors
        max_nodes = nodes
        max_colors = colors
        graph_factory = build_graph_factory(
            nodes,
            probability,
            args.graph_seed,
            fixed_edge_count=use_fixed_edge_count,
        )
        base_graph = graph_factory()

    env_kwargs = dict(
        k=max_colors,
        sa_iters=args.sa_iters,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp,
        tabu_iters=args.tabu_iters,
        tabu_tenure=args.tabu_tenure,
        search_algorithm=args.search_algorithm,
        beta=args.beta,
        stagnation_penalty=args.stagnation_penalty,
        reward_scale=args.reward_scale,
        max_episode_steps_RL=args.max_steps_RL,
        max_nodes=max_nodes,
        max_colors=max_colors,
        k_sampler=(lambda n, g: max(1, n // int(g.graph.get("color_divisor", 5)))) if args.random_nodes else None,
    )

    def make_train_env_fn():
        if args.train_graph_mode == "multi":
            return GcpEnv(graph=graph_factory(), graph_sampler=graph_factory, resample_on_reset=True, **env_kwargs)
        return GcpEnv(graph=base_graph.copy(), **env_kwargs)

    def make_test_env_fn():
        if args.eval_graph_mode == "multi":
            return GcpEnv(graph=graph_factory(), graph_sampler=graph_factory, resample_on_reset=True, **env_kwargs)
        return GcpEnv(graph=base_graph.copy(), **env_kwargs)

    env = make_train_env_fn()
    train_envs = DummyVectorEnv([make_train_env_fn for _ in range(args.train_env_num)])
    test_envs = DummyVectorEnv([make_test_env_fn for _ in range(args.test_env_num)])
    eval_env = make_test_env_fn()

    if args.device == "cpu":
        actor_device = torch.device("cpu")
        critic_device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    elif not torch.cuda.is_available():
        actor_device = torch.device("cpu")
        critic_device = torch.device("cpu")
    else:
        gpu_count = torch.cuda.device_count()
        if args.actor_device:
            actor_device = torch.device(args.actor_device)
        else:
            actor_device = torch.device("cuda:0")

        if args.critic_device:
            critic_device = torch.device(args.critic_device)
        elif args.split_gpus and gpu_count >= 2:
            critic_device = torch.device("cuda:1")
        else:
            critic_device = actor_device

    use_gnn = args.model_type == "gnn"

    print(f"Using actor_device={actor_device}, critic_device={critic_device}")
    if args.random_nodes:
        print(
            f"Training graph size: per-episode random nodes in [{args.min_nodes}, {args.max_nodes}], "
            f"rules: p=0.5->n//7, p=0.1->n//25, p=0.9->n//3 (max_colors={max_colors})"
        )
    else:
        print(f"Training graph size: nodes={nodes}, colors={colors} (random_nodes={args.random_nodes})")
    print(
        f"Graph mode: train={args.train_graph_mode}, eval={args.eval_graph_mode}, "
        f"graph_seed={args.graph_seed}, fixed_edge_count={use_fixed_edge_count}"
    )
    sys.stdout.flush()

    actor_lr = args.lr if args.lr is not None else args.actor_lr
    critic_lr = args.lr if args.lr is not None else args.critic_lr

    actor = ActorNetwork(3, 3, device=actor_device, use_gnn=use_gnn).to(actor_device)
    critic = CriticNetwork(
        3,
        3,
        device=critic_device,
        output_device=actor_device,
        use_gnn=use_gnn,
    ).to(critic_device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": actor_lr},
            {"params": critic.parameters(), "lr": critic_lr},
        ]
    )

    policy = GCPPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        nodes=max_nodes,
        k=max_colors,
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
        policy.load_state_dict(torch.load(args.input, map_location=actor_device))
        sys.stdout.flush()

    replay_buffer = VectorReplayBuffer(20000, len(train_envs))
    train_collector = Collector(policy, train_envs, replay_buffer)
    test_collector = Collector(policy, test_envs)

    node_log_name = f"{args.min_nodes}-{args.max_nodes}" if args.random_nodes else str(nodes)
    color_log_name = "profiled_colors" if args.random_nodes else str(colors)
    log_path = (
        f"./log/gcp_train_{node_log_name}nodes_{color_log_name}colors_"
        f"{args.epochs}epochs_{args.model_type}_{args.search_algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "config",
        f"nodes={nodes}\ncolors={colors}\nepochs={args.epochs}\n"
        f"model_type={args.model_type}\nsearch_algorithm={args.search_algorithm}\n"
        f"sa_iters={args.sa_iters}\ntabu_iters={args.tabu_iters}\nbeta={args.beta}\n"
        f"stagnation_penalty={args.stagnation_penalty}\nreward_scale={args.reward_scale}\nactor_lr={actor_lr}\ncritic_lr={critic_lr}\n"
        f"actor_device={actor_device}\ncritic_device={critic_device}\n"
        f"random_nodes={args.random_nodes}\nmin_nodes={args.min_nodes}\nmax_nodes={args.max_nodes}\n"
        f"train_graph_mode={args.train_graph_mode}\neval_graph_mode={args.eval_graph_mode}\n"
        f"graph_seed={args.graph_seed}\nfixed_edge_count={use_fixed_edge_count}",
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
