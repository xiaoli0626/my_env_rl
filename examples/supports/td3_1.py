#!/usr/bin/env python3

import datetime
import os
import pprint
from collections.abc import Callable

import numpy as np
import torch
import tianshou as ts
from env_model3 import YBGCEnv
from sensai.util import logging

from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import (
    Batch,
    Collector,
    CollectStats,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic

log = logging.getLogger(__name__)


def make_env_model3_env(
    seed: int,
    num_training_envs: int,
    num_test_envs: int,
    num_agent: int,
    d_limit: float,
    l_max: float,
    max_steps_per_episode: int,
    min_gc_init: float,
    w1: float,
    w2: float,
    success_r1_threshold: float | None,
) -> tuple[YBGCEnv, ts.env.DummyVectorEnv, ts.env.DummyVectorEnv]:
    def make_env_fn(env_seed: int) -> Callable[[], YBGCEnv]:
        def _init() -> YBGCEnv:
            return YBGCEnv(
                num_agent=num_agent,
                d_limit=d_limit,
                l_max=l_max,
                max_steps_per_episode=max_steps_per_episode,
                min_gc_init=min_gc_init,
                w1=w1,
                w2=w2,
                success_r1_threshold=success_r1_threshold,
                seed=env_seed,
            )

        return _init

    env = make_env_fn(seed)()
    training_envs = ts.env.DummyVectorEnv([make_env_fn(seed + i) for i in range(num_training_envs)])
    test_envs = ts.env.DummyVectorEnv(
        [make_env_fn(seed + num_training_envs + i) for i in range(num_test_envs)]
    )
    training_envs.seed(seed)
    test_envs.seed(seed)
    return env, training_envs, test_envs


def evaluate_success_metrics(
    algorithm: TD3,
    env_factory: Callable[[int], YBGCEnv],
    episodes: int,
    seed: int,
) -> dict[str, float]:
    algorithm.eval()
    success_num = 0
    episode_returns: list[float] = []
    success_returns: list[float] = []

    for ep in range(episodes):
        env = env_factory(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_batch = Batch(obs=np.expand_dims(obs, axis=0), info=np.array([{}], dtype=object))
            with torch.no_grad():
                act_batch = algorithm.policy(obs_batch)
            act = act_batch.act
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()
            act = np.asarray(act)[0]
            act = algorithm.policy.map_action(act)

            obs, rew, terminated, truncated, info = env.step(act)
            ep_return += float(rew)
            done = bool(terminated or truncated)
            if done and bool(info.get("success", info.get("is_success", False))):
                success_num += 1
                success_returns.append(ep_return)

        episode_returns.append(ep_return)

    success_rate = success_num / max(episodes, 1)
    avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    avg_success_return = float(np.mean(success_returns)) if success_returns else 0.0
    return {
        "episodes": float(episodes),
        "success_rate": float(success_rate),
        "avg_return": avg_return,
        "avg_success_return": avg_success_return,
    }


def main(
    persistence_base_dir: str = "log",
    seed: int = 0,
    buffer_size: int = 1500000,
    hidden_sizes: list | None = None,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.995,
    tau: float = 0.01,
    exploration_noise: float = 0.06,
    policy_noise: float = 0.10,
    noise_clip: float = 0.35,
    update_actor_freq: int = 2,
    start_timesteps: int = 300000,
    epoch: int = 200,
    epoch_num_steps: int = 8000,
    collection_step_num_env_steps: int = 60,
    update_per_step: int = 2,
    n_step: int = 1,
    batch_size: int = 512,
    num_training_envs: int = 8,
    num_test_envs: int = 4,
    render: float = 0.0,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "env_model3.td3",
    watch: bool = False,
    test_only: bool = False,
    test_episode_num: int = 6,
    success_eval_episodes: int = 100,
    num_agent: int = 50,
    d_limit: float = 50.0,
    l_max: float = 865.0,
    success_r1_threshold: float | None = None,
    w1: float = 0.8,
    w2: float = 0.2,
    max_steps_per_episode: int = 6,
    min_gc_init: float = 500.0,
) -> None:
    if hidden_sizes is None:
        hidden_sizes = [384, 384]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if success_r1_threshold is None:
        success_r1_threshold = 1 - 100 / ((num_agent - 1) * d_limit + l_max)


    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, training_envs, test_envs = make_env_model3_env(
        seed=seed,
        num_training_envs=num_training_envs,
        num_test_envs=num_test_envs,
        num_agent=num_agent,
        d_limit=d_limit,
        l_max=l_max,
        max_steps_per_episode=max_steps_per_episode,
        min_gc_init=min_gc_init,
        w1=w1,
        w2=w2,
        success_r1_threshold=success_r1_threshold
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = float(np.max(env.action_space.high))
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    log.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorDeterministic(
        preprocess_net=net_a,
        action_shape=action_shape,
        max_action=max_action,
    ).to(device)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)

    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
    critic1_optim = AdamOptimizerFactory(lr=critic_lr)
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
    critic2_optim = AdamOptimizerFactory(lr=critic_lr)

    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=exploration_noise * max_action),
        action_space=env.action_space,
    )

    algorithm: TD3 = TD3(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        policy_noise=policy_noise * max_action,
        update_actor_freq=update_actor_freq,
        noise_clip=noise_clip * max_action,
        n_step_return_horizon=n_step,
    )

    if resume_path:
        algorithm.load_state_dict(torch.load(resume_path, map_location=device))
        log.info(f"Loaded agent from: {resume_path}")

    if test_only and not resume_path:
        raise ValueError("test_only=True 时必须提供 resume_path，用于加载已训练好的模型。")


    buffer: VectorReplayBuffer | ReplayBuffer
    if num_training_envs > 1:
        buffer = VectorReplayBuffer(buffer_size, len(training_envs))
    else:
        buffer = ReplayBuffer(buffer_size)

    training_collector = Collector[CollectStats](
        algorithm,
        training_envs,
        buffer,
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    training_collector.reset()
    training_collector.collect(n_step=start_timesteps, random=True)

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    task = "env_model3"
    algo_name = "td3"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)

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
        torch.save(policy.state_dict(), os.path.join(log_path, "50buf_policy.pth"))

    if not watch and not test_only:
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

    def build_eval_env(eval_seed: int) -> YBGCEnv:
        return YBGCEnv(
            num_agent=num_agent,
            d_limit=d_limit,
            l_max=l_max,
            max_steps_per_episode=max_steps_per_episode,
            min_gc_init=min_gc_init,
            w1=w1,
            w2=w2,
            success_r1_threshold=success_r1_threshold,
            seed=eval_seed,
        )

    if test_only:
        test_metrics = evaluate_success_metrics(
            algorithm=algorithm,
            env_factory=build_eval_env,
            episodes=test_episode_num,
            seed=seed,
        )
        log.info(f"Test-only metrics: {test_metrics}")
        return

    algorithm.eval()
    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(collector_stats)

    success_metrics = evaluate_success_metrics(
        algorithm=algorithm,
        env_factory=build_eval_env,
        episodes=success_eval_episodes,
        seed=seed + 100000,
    )
    log.info(f"Success-priority metrics: {success_metrics}")


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)