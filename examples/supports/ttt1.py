
import datetime
import importlib.util
import os
import pprint
from pathlib import Path

import numpy as np
import torch
from sensai.util import logging

from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic
from env_model2 import YBGCEnv
log = logging.getLogger(__name__)

#
# def _load_ybgc_env_cls() -> type:
#     module_path = Path(__file__).resolve().parents[2] / "docs" / "ybgc_env_standardized.py"
#     spec = importlib.util.spec_from_file_location("ybgc_env_standardized", module_path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Cannot load env module from {module_path}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module.YBGCEnv
def _load_ybgc_env_cls() -> type:
    return YBGCEnv


def make_ybgc_env(
    seed: int,
    num_training_envs: int,
    num_test_envs: int,
    num_agent: int,
    d_limit: float,
    l_max: float,
    max_steps_per_episode: int,
    min_gc_init: float,
):
    YBGCEnv = _load_ybgc_env_cls()

    def make_single(env_seed: int):
        return YBGCEnv(
            num_agent=num_agent,
            d_limit=d_limit,
            l_max=l_max,
            max_steps_per_episode=max_steps_per_episode,
            min_gc_init=min_gc_init,
            seed=env_seed,
            target_r=0.9,
            w1=0.4,
            w2=0.6,
        )

    env = make_single(seed)
    training_envs = DummyVectorEnv([lambda i=i: make_single(seed + i + 1) for i in range(num_training_envs)])
    test_envs = DummyVectorEnv(
        [lambda i=i: make_single(seed + 10_000 + i + 1) for i in range(num_test_envs)]
    )
    training_envs.seed(seed)
    test_envs.seed(seed)
    return env, training_envs, test_envs


def main(
    persistence_base_dir: str = "log",
    seed: int = 0,
    hidden_sizes: list | None = None,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.98,
    tau: float = 0.005,
    exploration_noise: float = 0.05,
    policy_noise: float = 0.10,
    noise_clip: float = 0.20,
    update_actor_freq: int = 2,
    start_timesteps: int = 4000,
    epoch: int = 120,
    epoch_num_steps: int = 3000,
    collection_step_num_env_steps: int = 100,
    update_per_step: float = 2.0,
    n_step: int = 1,
    batch_size: int = 256,
    buffer_size: int = 200000,
    num_training_envs: int = 8,
    num_test_envs: int = 8,
    num_agent: int = 8,
    d_limit: float = 50.0,
    l_max: float = 865.0,
    min_gc_init: float = 500.0,
    max_steps_per_episode: int = 6,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "ybgc.td3",
    watch: bool = False,
    render: float = 0.0,
    print_test_info: bool = True,
    info_print_episodes: int = 3,
) -> None:
    """TD3 on custom YBGCEnv.

    当前超参数偏向“稳态高分”（在你锁定环境参数的前提下）：
    - 减小探索噪声
    - 提高并行采样和 warmup
    - 提升训练总步数与更新覆盖率，尽量把每轮测试回报推高
    """

    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, training_envs, test_envs = make_ybgc_env(
        seed=seed,
        num_training_envs=num_training_envs,
        num_test_envs=num_test_envs,
        num_agent=num_agent,
        d_limit=d_limit,
        l_max=l_max,
        max_steps_per_episode=max_steps_per_episode,
        min_gc_init=min_gc_init,
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = float(env.action_space.high[0])
    exploration_noise_scaled = exploration_noise * max_action
    policy_noise_scaled = policy_noise * max_action

    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    log.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorDeterministic(
        preprocess_net=net_a, action_shape=action_shape, max_action=max_action
    ).to(device)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)

    net_c1 = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes, concat=True)
    net_c2 = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes, concat=True)
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
    critic1_optim = AdamOptimizerFactory(lr=critic_lr)
    critic2_optim = AdamOptimizerFactory(lr=critic_lr)

    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=exploration_noise_scaled),
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
        policy_noise=policy_noise_scaled,
        update_actor_freq=update_actor_freq,
        noise_clip=noise_clip,
        n_step_return_horizon=n_step,
    )

    if resume_path:
        log.info(f"Loaded agent from: {resume_path}")

    buffer = (
        VectorReplayBuffer(buffer_size, len(training_envs))
        if num_training_envs > 1
        else ReplayBuffer(buffer_size)
    )
    training_collector = Collector[CollectStats](algorithm, training_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    training_collector.reset()
    training_collector.collect(n_step=start_timesteps, random=True)

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "td3"
    task = "YBGCEnv"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)

    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = "wandb" if logger_type == "wandb" else "tensorboard"
    if logger_type == "wandb":
        logger_factory.wandb_project = wandb_project

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=resume_id,
        config_dict=params_log_info,
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not watch:
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

    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(collector_stats)

    if print_test_info:
        log.info("Start detailed test rollout with per-step env info printing...")
        YBGCEnv = _load_ybgc_env_cls()
        eval_env = YBGCEnv(
            num_agent=num_agent,
            d_limit=d_limit,
            l_max=l_max,
            max_steps_per_episode=max_steps_per_episode,
            min_gc_init=min_gc_init,
            seed=seed,
            target_r=0.9,
            w1=0.4,
            w2=0.6,
        )

        algorithm.policy.eval()
        for ep in range(info_print_episodes):
            obs, info = eval_env.reset(seed=seed + ep)
            done = False
            step_idx = 0
            ep_rew = 0.0
            log.info(f"[Eval episode {ep + 1}] reset info={info}")
            while not done:
                obs_batch = Batch(obs=np.expand_dims(obs, axis=0), info=np.array([{}], dtype=object))
                with torch.no_grad():
                    act_batch = algorithm.policy(obs_batch)
                act = act_batch.act
                if isinstance(act, torch.Tensor):
                    act = act.detach().cpu().numpy()
                act = np.asarray(act)[0]
                act = algorithm.policy.map_action(act)

                obs, rew, terminated, truncated, info = eval_env.step(act)
                ep_rew += float(rew)
                step_idx += 1
                log.info(
                    f"[Eval episode {ep + 1} step {step_idx}] reward={rew:.6f}, "
                    f"terminated={terminated}, truncated={truncated}, info={info}"
                )
                done = bool(terminated or truncated)
            log.info(f"[Eval episode {ep + 1}] total_reward={ep_rew:.6f}")


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)