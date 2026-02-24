#!/usr/bin/env python3

import datetime
import os
import pprint

import numpy as np
import torch
from env_model import YBGCEnv  # 导入你自定义的环境
from sensai.util import logging

from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic
from tianshou.env import DummyVectorEnv  # 导入 DummyVectorEnv

log = logging.getLogger(__name__)

# 设置环境
def make_env():
    return YBGCEnv(
        num_agent=10,
        d_limit=50.0,
        l_max=865.0,
        piancha=50.0,
        max_steps_per_episode=6,
        target_r=0.0,
    )


# 训练主函数
def main(
    task: str = "YBGC",  # 任务名称
    persistence_base_dir: str = "log",
    seed: int = 0,
    hidden_sizes: list | None = None,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.95,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    update_actor_freq: int = 2,
    start_timesteps: int = 2000,
    epoch: int = 200,
    epoch_num_steps: int = 1000,
    collection_step_num_env_steps: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    batch_size: int = 128,
    buffer_size: int = 2000000,
    num_training_envs: int = 1,
    num_test_envs: int = 10,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "mujoco.benchmark",
    watch: bool = False,
    render: float = 0.0,
) -> None:
    # Set defaults for mutable arguments
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all local variables as config
    params_log_info = locals()
    log.info(f"Starting training with config:\n{pprint.pformat(params_log_info)}")

    # 创建训练和测试环境，使用 DummyVectorEnv 包装多个环境实例
    training_envs = DummyVectorEnv([make_env for _ in range(num_training_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(num_test_envs)])

    # --------------------------
    # 修复1：正确获取向量环境的空间维度
    # --------------------------
    single_env = training_envs.workers[0].env
    state_shape = single_env.observation_space.shape[0]
    action_shape = single_env.action_space.shape[0]
    max_action = 1.0  # 标准化为1.0
    exploration_noise = exploration_noise * max_action
    policy_noise = policy_noise * max_action
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    log.info(f"Action range (normalized): 0.0, {max_action}")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------
    # 修复2：网络创建（仅模型to(device)，优化器工厂不处理）
    # --------------------------
    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes).to(device)  # 提前to(device)
    actor = ContinuousActorDeterministic(
        preprocess_net=net_a, action_shape=action_shape, max_action=max_action
    ).to(device)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)

    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    ).to(device)  # 提前to(device)
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
    critic1_optim = AdamOptimizerFactory(lr=critic_lr)

    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    ).to(device)  # 提前to(device)
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
    critic2_optim = AdamOptimizerFactory(lr=critic_lr)

    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=exploration_noise),
        action_space=single_env.action_space,
    )

    # --------------------------
    # 修复3：TD3算法（移除device参数，适配极旧版本）
    # --------------------------
    algorithm: TD3 = TD3(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        policy_noise=policy_noise,
        update_actor_freq=update_actor_freq,
        noise_clip=noise_clip,
        n_step_return_horizon=n_step,
        # 关键：删除 device=device 参数（旧版本TD3无此参数）
    )

    # 加载之前的策略
    if resume_path:
        log.info(f"Loaded agent from: {resume_path}")
        algorithm.load_state_dict(torch.load(resume_path, map_location=device))

    # --------------------------
    # 修复4：缓冲区创建适配向量环境
    # --------------------------
    buffer: VectorReplayBuffer | ReplayBuffer
    if num_training_envs > 1:
        buffer = VectorReplayBuffer(buffer_size, len(training_envs))
    else:
        buffer = ReplayBuffer(buffer_size)

    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    training_collector.reset()
    training_collector.collect(n_step=start_timesteps, random=True)

    # 日志记录设置
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "td3"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)
    os.makedirs(log_path, exist_ok=True)

    logger_factory = LoggerFactoryDefault()
    if logger_type == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=resume_id,
        config_dict=params_log_info,
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not watch:
        # 训练过程
        result = algorithm.run_training(
            OffPolicyTrainerParams(
                training_collector=training_collector,
                test_collector=test_collector,
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
                test_step_num_episodes=num_test_envs,
                batch_size=batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
                update_step_num_gradient_steps_per_sample=update_per_step,
                test_in_training=False,
            )
        )
        pprint.pprint(result)


    test_collector.reset()

    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(collector_stats)

if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)



