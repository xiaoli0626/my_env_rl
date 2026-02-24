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
# 补充导入：优先向量经验池
from tianshou.data import (
    Collector, CollectStats, ReplayBuffer, VectorReplayBuffer,
    PrioritizedReplayBuffer, PrioritizedVectorReplayBuffer
)
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
    # --------------------------
    # 1. 对齐优化器学习率（你的手写版：actor=3e-4, critic=5e-4）
    # --------------------------
    actor_lr: float = 3e-4,
    critic_lr: float = 5e-4,
    # --------------------------
    # 2. 对齐TD3核心超参数
    # --------------------------
    gamma: float = 0.95,        # 对齐你的 gamma=0.95
    tau: float = 0.005,         # 对齐你的 tau=0.005
    exploration_noise: float = 0.05,  # 对齐你的探索噪声 sigma=0.05（原0.1）
    policy_noise: float = 0.05,       # 对齐你的目标策略噪声 sigma=0.05（原0.2）
    noise_clip: float = 0.2,          # 对齐你的噪声裁剪 c=0.2（原0.5）
    update_actor_freq: int = 3,       # 对齐你的 actor延迟更新 d=3（原2）
    # --------------------------
    # 3. 对齐训练起步参数
    # --------------------------
    start_timesteps: int = 50000,  # 对齐你的随机探索步数 50000（原2000）
    # --------------------------
    # 4. 对齐训练总步数（关键！你的手写版是20万episode×6步≈120万步）
    #    epoch=2000, epoch_num_steps=600 → 总训练步数=2000×600=120万步
    # --------------------------
    epoch: int = 200000,           # 训练轮数（原200）
    epoch_num_steps: int = 6,  # 每轮步数（原1000）
    collection_step_num_env_steps: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    # --------------------------
    # 5. 对齐Batch Size（你的手写版 batch_size=256）
    # --------------------------
    batch_size: int = 256,
    # --------------------------
    # 6. 对齐经验池配置（你的手写版 buffer_size=1200000）
    # --------------------------
    buffer_size: int = 1200000,
    num_training_envs: int = 1,  # 你的手写版是单环境训练
    num_test_envs: int = 10,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "mujoco.benchmark",
    watch: bool = False,
    render: float = 0.0,
    # --------------------------
    # 7. 对齐PER参数（你的手写版PER配置）
    # --------------------------
    alpha: float = 0.6,        # 你的手写版 alpha=0.6
    beta: float = 0.4,         # 你的手写版 beta=0.4
    weight_norm: bool = True,  # 你的手写版权重归一化（weights/weights.max()）
    beta_anneal_step: int = 100000,  # 你的手写版 beta_increment=1e-6 → 10万步到1.0
    beta_final: float = 1.0     # 你的手写版 beta最终到1.0
) -> None:
    # --------------------------
    # 8. 对齐网络结构（你的手写版：Actor/Critic是[256,128]，fuse_hidden=128）
    #    Tianshou的Net是单序列，用[256,128]对齐你的双层结构
    # --------------------------
    if hidden_sizes is None:
        hidden_sizes = [256, 128]  # 原[256,256] → 改为[256,128]
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
    # 核心修改：完善优先经验池创建（补充源码要求的参数）
    # --------------------------
    buffer: PrioritizedReplayBuffer | PrioritizedVectorReplayBuffer
    if num_training_envs > 1:
        # 多环境：使用优先向量经验池（参数与普通优先池一致）
        buffer = PrioritizedVectorReplayBuffer(
            buffer_size,
            len(training_envs),
            alpha=alpha,          # 对应源码 __init__ 的 alpha
            beta=beta,            # 对应源码 __init__ 的 beta
            weight_norm=weight_norm  # 对应源码 __init__ 的 weight_norm
        )
    else:
        # 单环境：使用普通优先经验池（严格对应源码的 __init__ 参数）
        buffer = PrioritizedReplayBuffer(
            buffer_size,
            alpha=alpha,
            beta=beta,
            weight_norm=weight_norm
        )

    # 可选扩展：beta 退火（训练过程中逐步提升 beta 到 1.0）
    # 定义 beta 退火函数（训练时调用）
    def update_beta(step: int):
        if step > beta_anneal_step:
            current_beta = beta_final
        else:
            # 线性退火
            current_beta = beta + (beta_final - beta) * (step / beta_anneal_step)
        buffer.set_beta(current_beta)  # 调用源码的 set_beta 方法更新

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
        # 训练过程：添加 beta 退火逻辑（可选）
        total_steps = 0
        # 自定义训练循环（可选，若用原生 run_training 可通过回调实现）
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
                test_in_training=False
            )
        )
        pprint.pprint(result)

    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(collector_stats)

if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)