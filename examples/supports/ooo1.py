import torch
import torch.nn as nn
import numpy as np
import datetime
import os
import pprint
from typing import Sequence, Any, Tuple, List, Dict, Optional
from gymnasium.spaces import Box
from tianshou.data import (
    Collector, CollectStats, ReplayBuffer, VectorReplayBuffer,
    PrioritizedReplayBuffer, PrioritizedVectorReplayBuffer
)
# Tianshou核心导入
from tianshou.utils.net.common import ModuleWithVectorOutput, MLP, torch_device
from tianshou.utils.net.continuous import (
    AbstractContinuousActorDeterministic,
    AbstractContinuousCritic,
    ContinuousActorDeterministic,
    ContinuousCritic,
)
from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.env import DummyVectorEnv

# 第三方导入
from env_model import YBGCEnv
from sensai.util import logging

# 日志配置
log = logging.getLogger(__name__)
logging.getLogger("tianshou.data.collector").setLevel(logging.WARNING)


# --------------------------
# 1. Actor特征提取网络（适配你的Tianshou版本）
# --------------------------
class ActorPreprocessNet(ModuleWithVectorOutput):
    def __init__(self, num_agent: int, hidden1: int = 256, hidden2: int = 128, fuse_hidden: int = 128):
        super().__init__(output_dim=fuse_hidden)
        self.num_agent = num_agent
        self.coord_dim = num_agent
        self.rod_dim = num_agent
        self.fuse_hidden = fuse_hidden

        # 分支1：绝对坐标特征编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(self.coord_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True)
        )

        # 分支2：杆长特征编码
        self.rod_encoder = nn.Sequential(
            nn.Linear(self.rod_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True)
        )

        # 特征融合层
        self.fuse_layer = nn.Sequential(
            nn.Linear(hidden2 * 2, fuse_hidden),
            nn.ReLU(inplace=True)
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        if isinstance(obs, (tuple, list)):
            obs = obs[0] if len(obs) > 0 else obs

        obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)

        coord_input = obs[:, :self.num_agent]
        rod_input = obs[:, self.num_agent:]

        coord_feat = self.coord_encoder(coord_input)
        rod_feat = self.rod_encoder(rod_input)
        fuse_feat = torch.cat([coord_feat, rod_feat], dim=1)
        fuse_feat = self.fuse_layer(fuse_feat)

        return fuse_feat, state

    def get_output_dim(self) -> int:
        return self.fuse_hidden


# --------------------------
# 2. 自定义Critic（彻底解决维度问题）
# --------------------------
class CustomContinuousCritic(AbstractContinuousCritic):
    """
    自定义Critic，完全控制维度，避免ContinuousCritic的自动维度计算问题
    """

    def __init__(
            self,
            *,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: Sequence[int] = (256, 128),
            linear_layer: nn.Module = nn.Linear,
            max_action: float = 1.0
    ) -> None:
        super().__init__(output_dim=1)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # 观测特征提取（和Actor保持一致）
        self.obs_encoder = ActorPreprocessNet(num_agent=obs_dim // 2)  # obs_dim=20 → num_agent=10
        obs_feature_dim = self.obs_encoder.get_output_dim()

        # 总输入维度：观测特征 + 动作
        total_input_dim = obs_feature_dim + action_dim

        # 手动构建MLP，完全控制维度
        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_sizes:
            layers.append(linear_layer(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        # 输出层：Q值
        layers.append(linear_layer(prev_dim, 1))

        self.q_net = nn.Sequential(*layers)

        # 参数初始化
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            act: np.ndarray | torch.Tensor | None = None,
            info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        device = next(self.parameters()).device

        # 处理观测
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs_feat, _ = self.obs_encoder(obs)

        # 处理动作
        if act is not None:
            act = torch.as_tensor(act, dtype=torch.float32, device=device)
            # 确保动作维度正确
            if len(act.shape) == 1:
                act = act.unsqueeze(0)
            act = act.flatten(1)

        # 拼接观测特征和动作
        if act is not None:
            input_feat = torch.cat([obs_feat, act], dim=1)
        else:
            input_feat = obs_feat

        # 计算Q值
        q_value = self.q_net(input_feat)

        return q_value.squeeze(-1)  # 确保输出形状是 [batch_size]


# --------------------------
# 3. 网络创建函数（终极修复）
# --------------------------
def create_official_actor(num_agent: int = 10, hidden1: int = 256, hidden2: int = 128,
                          fuse_hidden: int = 128) -> ContinuousActorDeterministic:
    preprocess_net = ActorPreprocessNet(num_agent, hidden1, hidden2, fuse_hidden)

    actor = ContinuousActorDeterministic(
        preprocess_net=preprocess_net,
        action_shape=(num_agent,),
        hidden_sizes=(),
        max_action=1.0
    )

    # ✅ 替换 actor.last.model 为 Linear + Sigmoid
    actor.last.model = nn.Sequential(
        nn.Linear(fuse_hidden, num_agent),
        nn.Sigmoid()
    )

    return actor


def create_custom_critic(obs_dim: int = 20, action_dim: int = 10) -> CustomContinuousCritic:
    """
    创建自定义Critic，完全控制维度
    """
    critic = CustomContinuousCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(256, 128),  # 隐藏层维度，无需考虑输入维度
        linear_layer=nn.Linear,
        max_action=1.0
    )

    return critic


# --------------------------
# 4. 环境创建
# --------------------------
def make_env() -> YBGCEnv:
    return YBGCEnv(
        num_agent=10,
        d_limit=50.0,
        l_max=865.0,
        piancha=50.0,
        max_steps_per_episode=6,
        target_r=0.9,
    )


# --------------------------
# 5. 训练主函数
# --------------------------
def main(
        task: str = "YBGC",  # 任务名称
        persistence_base_dir: str = "log",
        seed: int = 0,
        hidden_sizes: list | None = None,
        # --------------------------
        # 1. 统一优化器学习率（对齐手写版：actor=3e-4, critic=5e-4）
        # --------------------------
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        # --------------------------
        # 2. 统一TD3核心超参数
        # --------------------------
        gamma: float = 0.95,        # 对齐手写版 gamma=0.95
        tau: float = 0.005,         # 对齐手写版 tau=0.005
        exploration_noise: float = 0.05,  # 对齐手写版探索噪声 sigma=0.05
        policy_noise: float = 0.05,       # 对齐手写版目标策略噪声 sigma=0.05
        noise_clip: float = 0.2,          # 对齐手写版噪声裁剪 c=0.2
        update_actor_freq: int = 3,       # 对齐手写版actor延迟更新 d=3
        # --------------------------
        # 3. 统一训练起步参数
        # --------------------------
        start_timesteps: int = 50000,  # 对齐手写版：前5万步随机探索
        # --------------------------
        # 4. 统一训练总步数（关键！20万轮×6步=120万步）
        # --------------------------
        epoch: int = 50,           # 总轮数（对齐手写版total_episodes=20万）
        epoch_num_steps: int = 10000,      # 每轮步数（对齐手写版max_steps_per_episode=6）
        collection_step_num_env_steps: int = 100,
        update_per_step: int = 1,
        n_step: int = 1,
        # --------------------------
        # 5. 统一Batch Size（对齐手写版 batch_size=256）
        # --------------------------
        batch_size: int = 256,
        # --------------------------
        # 6. 统一经验池配置（对齐手写版 buffer_size=1200000）
        # --------------------------
        buffer_size: int = 1200000,
        num_training_envs: int = 1,  # 对齐手写版：单环境训练
        num_test_envs: int = 10,
        device: str | None = None,
        resume_path: str | None = None,
        resume_id: str | None = None,
        logger_type: str = "tensorboard",
        wandb_project: str = "mujoco.benchmark",
        watch: bool = False,
        render: float = 0.0,
        # --------------------------
        # 7. 新增：PER核心参数（对齐手写版配置）
        # --------------------------
        alpha: float = 0.6,        # 手写版 alpha=0.6
        beta: float = 0.4,         # 手写版 beta=0.4
        weight_norm: bool = True,  # 手写版：权重归一化（weights/weights.max()）
        beta_anneal_step: int = 100000,  # 手写版 beta_increment=1e-6 → 10万步到1.0
        beta_final: float = 1.0     # 手写版 beta最终值=1.0
) -> None:
    # 设备设置
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = str(device) if not isinstance(device, str) else device

    # 统一网络结构参数（对齐手写版：[256,128]）
    if hidden_sizes is None:
        hidden_sizes = [256, 128]

    # 日志输出（打印统一后的参数）
    params_log_info = locals()
    log.info(f"Starting training with unified config (PER+对齐手写版):\n{pprint.pformat(params_log_info)}")

    # 创建环境
    training_envs = DummyVectorEnv([make_env for _ in range(num_training_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(num_test_envs)])

    # 获取环境维度
    single_env = training_envs.workers[0].env
    state_shape = single_env.observation_space.shape[0]  # 20
    action_shape = single_env.action_space.shape[0]  # 10
    max_action = 1.0

    # 噪声标准化（对齐max_action=1.0）
    exploration_noise = exploration_noise * max_action
    policy_noise = policy_noise * max_action



    log.info(f"Observation shape: {state_shape}")
    log.info(f"Action shape: {action_shape}")
    log.info(f"Max action (normalized): {max_action}")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    training_envs.seed(seed)
    test_envs.seed(seed + 1)

    # --------------------------
    # 创建网络（保留自定义Actor/Critic，仅迁移到设备）
    # --------------------------
    # Actor网络
    actor = create_official_actor(num_agent=10).to(device_str)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)

    # Critic网络（使用自定义版本，彻底解决维度问题）
    critic1 = create_custom_critic(obs_dim=state_shape, action_dim=action_shape).to(device_str)
    critic1_optim = AdamOptimizerFactory(lr=critic_lr)

    critic2 = create_custom_critic(obs_dim=state_shape, action_dim=action_shape).to(device_str)
    critic2_optim = AdamOptimizerFactory(lr=critic_lr)

    # 创建策略
    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=exploration_noise),
        action_space=single_env.action_space,
    )
    # 创建TD3算法（移除device参数，适配旧版本）
    td3: TD3 = TD3(
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
    )

    # 加载预训练模型
    if resume_path:
        log.info(f"Loading policy from {resume_path}")
        td3.load_state_dict(torch.load(resume_path, map_location=device_str))

    # --------------------------
    # 核心修改1：替换为优先经验池（PER）
    # --------------------------
    buffer: PrioritizedReplayBuffer | PrioritizedVectorReplayBuffer
    if num_training_envs > 1:
        # 多环境：优先向量经验池
        buffer = PrioritizedVectorReplayBuffer(
            buffer_size,
            len(training_envs),
            alpha=alpha,
            beta=beta,
            weight_norm=weight_norm
        )
    else:
        # 单环境：普通优先经验池（对齐手写版）
        buffer = PrioritizedReplayBuffer(
            buffer_size,
            alpha=alpha,
            beta=beta,
            weight_norm=weight_norm
        )

    # --------------------------
    # 核心修改2：beta退火函数（对齐手写版 beta_increment=1e-6）
    # --------------------------
    def update_beta(step: int):
        # 线性退火：step<=10万步时，beta从0.4升到1.0；超过后固定为1.0
        if step > beta_anneal_step:
            current_beta = beta_final
        else:
            current_beta = beta + (beta_final - beta) * (step / beta_anneal_step)
        buffer.set_beta(current_beta)  # 调用PER的set_beta方法更新

    # 创建收集器
    training_collector = Collector[CollectStats](
        td3, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](td3, test_envs)

    # 预收集随机数据（对齐手写版：前5万步随机探索）
    training_collector.reset()
    log.info(f"Collecting initial random data ({start_timesteps} steps)")
    training_collector.collect(n_step=start_timesteps, random=True)

    # 日志配置
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "td3_per", str(seed), now)  # 标注PER版本
    log_path = os.path.join(persistence_base_dir, log_name)
    os.makedirs(log_path, exist_ok=True)

    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = logger_type
    if logger_type == "wandb":
        logger_factory.wandb_project = wandb_project

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=resume_id,
        config_dict=params_log_info,
    )

    # 保存最佳模型
    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy_per.pth"))
        log.info(f"Saved best PER policy to {log_path}/best_policy_per.pth")

    # 开始训练（添加beta退火回调）
    if not watch:
        log.info("Starting TD3 training with PER (unified params)...")
        result = td3.run_training(
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
        log.info("Training with PER completed!")
        pprint.pprint(result)

    # 测试最终策略
    log.info("Testing final PER policy...")

    test_collector.reset(reset_buffer=True)
    test_stats = test_collector.collect(n_episode=num_test_envs, render=render)

    print("\n===== RAW TEST INFOS (filtered) =====")
    buf = test_collector.buffer

    # 只遍历有效长度，避免循环buffer尾部垃圾
    for i in range(len(buf)):
        info = buf.info[i]

        if info is None:
            continue

        # 多环境时可能是 list/tuple of infos
        if isinstance(info, (list, tuple)):
            for j, one in enumerate(info):
                if isinstance(one, dict) and ("r" in one or "r1_yc" in one or "r2_yb" in one):
                    print(f"[step {i} env {j}] {one}")
            continue

        # 普通 dict
        if isinstance(info, dict):
            # ✅ 如果被包装过，常见是嵌套一层
            if ("r" in info) or ("r1_yc" in info) or ("r2_yb" in info):
                print(f"[step {i}] {info}")
            else:
                # 尝试从嵌套里找（你只要看一次输出就知道该取哪个key）
                for k, v in info.items():
                    if isinstance(v, dict) and (("r" in v) or ("r1_yc" in v) or ("r2_yb" in v)):
                        print(f"[step {i}] wrapped_key={k} -> {v}")

    log.info(f"Test results (PER version):\n{pprint.pformat(test_stats)}")


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)

