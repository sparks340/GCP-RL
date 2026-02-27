import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.trainer import OnpolicyTrainer
import gymnasium as gym
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
import networkx as nx
from gcp_env.gcp_env import GcpEnv
from tianshou.env import SubprocVectorEnv
import argparse
import sys
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import time

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="基于强化学习的模拟退火图着色算法训练器"
        )
        parser.add_argument("output", type=str, help="策略输出文件路径")
        parser.add_argument(
            "--input", type=str, default=None, help="策略输入文件路径"
        )
        parser.add_argument(
            "-I",
            "--max_steps_RL",
            type=int,
            dest="max_steps_RL",
            default=300,
            help="每个episode的RL最大步数",
        )
        parser.add_argument(
            "-L",
            "--max-steps",
            type=int,
            dest="max_steps",
            default=320,
            help="每个episode的最大步数",
        )
        parser.add_argument(
            "-S",
            "--sa-iters",
            type=int,
            dest="sa_iters",
            default=500000,
            help="模拟退火算法的迭代次数",
        )
        parser.add_argument(
            "-T",
            "--initial-temp",
            type=float,
            dest="initial_temp",
            default=500.0,
            help="模拟退火初始温度",
        )
        parser.add_argument(
            "-C",
            "--cooling-rate",
            type=float,
            dest="cooling_rate",
            default=0.995,
            help="模拟退火冷却率",
        )
        parser.add_argument(
            "-M",
            "--min-temp",
            type=float,
            dest="min_temp",
            default=0.01,
            help="模拟退火最小温度",
        )
        parser.add_argument(
            "-E",
            "--epochs",
            type=int,
            dest="epochs",
            default=50,
            help="训练轮数",
        )
        parser.add_argument(
            "-N",
            "--nodes",
            type=int,
            dest="nodes",
            default=250,
            help="训练图中的节点数",
        )
        parser.add_argument(
            "-P",
            "--probability",
            type=float,
            dest="probability",
            default=0.5,
            help="训练图中节点间的边概率",
        )
        parser.add_argument(
            "-K",
            "--colors",
            type=int,
            dest="colors",
            default=24,
            help="允许使用的颜色数",
        )
        parser.add_argument(
            "-B",
            "--beta",
            type=float,
            dest="beta",
            default=0.2,
            help="奖励计算中的beta参数",
        )

        args = parser.parse_args()

        # 注册环境
        gym.register(
            id="GcpEnvMaxIters-v0",
            entry_point="gcp_env.gcp_env:GcpEnv",
            max_episode_steps=args.max_steps,
        )

        print("解析参数完成...")
        sys.stdout.flush()

        nodes = args.nodes
        probability = args.probability
        colors = args.colors

        print(f"创建环境，节点数: {nodes}, 边概率: {probability}, 颜色数: {colors}")
        sys.stdout.flush()

        env = GcpEnv(
            graph=nx.gnp_random_graph(nodes, probability),
            k=colors,
            sa_iters=args.sa_iters,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            min_temp=args.min_temp,
            beta=args.beta,
            max_episode_steps_RL= args.max_steps_RL,
            max_episode_steps = args.max_steps
        )

        print("创建环境完成...")
        sys.stdout.flush()

        print("设置训练和测试环境...")
        sys.stdout.flush()

        train_envs = SubprocVectorEnv(
            [
                lambda: GcpEnv(
                    graph=nx.gnp_random_graph(nodes, probability),
                    k=colors,
                    sa_iters=args.sa_iters,
                    initial_temp=args.initial_temp,
                    cooling_rate=args.cooling_rate,
                    min_temp=args.min_temp,
                    beta=args.beta,
                    max_episode_steps_RL= args.max_steps_RL,
                    max_episode_steps = args.max_steps

                )
                for _ in range(1)
            ]
        )
        test_envs = SubprocVectorEnv(
            [
                lambda: GcpEnv(
                    graph=nx.gnp_random_graph(nodes, probability),
                    k=colors,
                    sa_iters=args.sa_iters,
                    initial_temp=args.initial_temp,
                    cooling_rate=args.cooling_rate,
                    min_temp=args.min_temp,
                    beta=args.beta,
                    max_episode_steps_RL= args.max_steps_RL,
                    max_episode_steps = args.max_steps
                )
                for _ in range(1)
            ]
        )

        print("环境设置完成...")
        sys.stdout.flush()

        print("设置策略...")
        sys.stdout.flush()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        sys.stdout.flush()

        # 修改为使用原始的特征维度
        node_features = 3
        col_features = 3

        print("设置网络...")
        sys.stdout.flush()

        actor = ActorNetwork(node_features, col_features, device=device).to(device)
        critic = CriticNetwork(node_features, col_features, device=device).to(device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

        print("创建策略...")
        sys.stdout.flush()

        dist = torch.distributions.Categorical
        policy = GCPPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            nodes=nodes,
            k=colors,
            action_space=env.action_space,
            eps_clip=0.2,
            dual_clip=None,
            value_clip=True,
            advantage_normalization=True,
            recompute_advantage=True,
        )

        if args.input is not None:
            print(f"加载策略: {args.input}")
            policy.load_state_dict(torch.load(args.input, map_location=device))

        print("设置回放缓冲区和收集器...")
        sys.stdout.flush()

        buffer_size = 20000  # 增加缓冲区大小
        replay_buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        train_collector = Collector(policy, train_envs, replay_buffer)
        test_collector = Collector(policy, test_envs)

        print("设置日志记录器...")
        sys.stdout.flush()

        log_path = f"./log/gcp_train_{args.nodes}nodes_{args.colors}colors_{args.epochs}epochs_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_path)
        writer.add_text("参数配置", 
            f"节点数: {args.nodes}\n" +
            f"颜色数: {args.colors}\n" +
            f"训练轮数: {args.epochs}\n" +
            f"模拟退火迭代次数: {args.sa_iters}\n" +
            f"初始温度: {args.initial_temp}\n" +
            f"冷却率: {args.cooling_rate}\n" +
            f"最小温度: {args.min_temp}\n" +
            f"beta: {args.beta}"
        )
        logger = TensorboardLogger(writer, train_interval=1, update_interval=1)

        print("开始训练，进度将实时显示...")
        sys.stdout.flush()

        # 训练参数设置
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

        try:
            print("开始执行训练循环...")
            sys.stdout.flush()
            
            # 设置进度回调函数
            def progress_callback(epoch, env_step, gradient_step):
                print(f"当前轮次: {epoch}, 环境步数: {env_step}, 梯度步数: {gradient_step}")
                sys.stdout.flush()
            
            result = trainer.run(progress_callback)
            
            print("训练完成！")
            print(f"训练结果: {result}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

        print("保存策略...")
        sys.stdout.flush()

        torch.save(policy.state_dict(), args.output)

        print("关闭日志记录器...")
        sys.stdout.flush()
        writer.close()

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush() 