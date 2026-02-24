import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import List, Tuple, Dict, Any, Optional

def env_mode(
    arrange_yc_list: List[float],
    arrange_gc_list: List[float],
    action_list: List[float],
    l_max: float,
    d_limit: float
) -> Tuple[List[float], List[float], List[float]]:
    """
    你的动力学：输入 (yc, gc, action) -> 输出 (new_yc, new_gc, new_yb)
    保留 while 收敛写法
    """
    n = len(action_list)

    new_yc_list = [arrange_yc_list[i] + action_list[i] for i in range(n)]
    mqgc_list   = [arrange_gc_list[i] - action_list[i] for i in range(n)]
    mqgb_list   = [new_yc_list[i] + mqgc_list[i] for i in range(n)]
    l_max_list  = [l_max - mqgc_list[i] for i in range(n)]

    action_sum = [0.0] * n
    saturated  = [False] * n

    def apply_delta(i: int, delta: float) -> None:
        if saturated[i]:
            return
        remain = l_max_list[i] - action_sum[i]
        if remain <= 0:
            saturated[i] = True
            return

        if action_sum[i] + delta <= l_max_list[i]:
            actual = delta
        else:
            actual = remain
            saturated[i] = True

        mqgb_list[i] += actual
        action_sum[i] += actual

    prev_action_sum = None
    while True:
        if prev_action_sum is not None and action_sum == prev_action_sum:
            break
        prev_action_sum = action_sum.copy()

        for i in range(n):
            if saturated[i]:
                continue
            if n == 1:
                saturated[i] = True
                continue

            if i == 0:
                delta = d_limit - mqgb_list[i] + mqgb_list[i + 1]
                apply_delta(i, delta)
            elif i == n - 1:
                delta = d_limit - mqgb_list[i] + mqgb_list[i - 1]
                apply_delta(i, delta)
            else:
                action1 = d_limit - mqgb_list[i] + mqgb_list[i + 1]
                action2 = d_limit - mqgb_list[i] + mqgb_list[i - 1]
                delta = action2 if action1 > action2 else action1
                apply_delta(i, delta)

    zxgc_list = [mqgc_list[i] + action_sum[i] for i in range(n)]
    return new_yc_list, zxgc_list, mqgb_list

class YBGCEnv(gym.Env):
    """
    你标定的标准环境：
    - obs/state = [arrange_yb_list, arrange_gc_list] 拼接
    - action = action_phys (每个agent一个动作，物理量)
    - arrange_yc_list 仅用于 reward 中间计算，不作为状态输出
    - done:
        truncated: step_count >= max_steps_per_episode
        terminated: 达到目标阈值（例如 reward >= target_reward）
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_agent: int = 10,
        d_limit: float = 50.0,
        l_max: float = 865.0,
        piancha: float = 50.0,
        max_steps_per_episode: int = 6,
        # 目标阈值
        target_r: float = 0.6,
        w1: float = 0.4,
        w2: float = 0.6,
        seed: Optional[int] = None,
        # 动作上界：为了让 action_space 固定（Gym要求），设一个硬上界
        # 如果你希望动作可到 gc[i]，那就用连续动作[0,1]再映射；这里先按“物理动作整数”做
        action_max: float = 865.0
    ):
        super().__init__()
        self.num_agent = int(num_agent)
        self.d_limit = float(d_limit)
        self.l_max = float(l_max)
        self.piancha = float(piancha)
        self.max_steps = int(max_steps_per_episode)
        self.target_r = target_r
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.action_max = float(action_max)
        self.rng = np.random.default_rng(seed)

        # obs = [yb, gc] -> shape (2n,)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(2 * self.num_agent,), dtype=np.float32
        )

        # action = action_phys，每个 agent 一个动作，范围 [0, action_max]
        # 你原来用 randint(0, gc[i])，那是“随 state 变化的动作上限”，Gym 不允许动态 action_space；
        # 所以这里用固定上界 action_max，并在 step 里 clip 到 gc[i]
        self.action_space = Box(
            low=0.0, high=1.0, shape=(self.num_agent,), dtype=np.float32
        )

        # internal state
        self.yb: List[float] = []
        self.gc: List[float] = []
        self.step_count: int = 0

    # ---------- helpers ----------
    def _compute_yc(self, yb: List[float], gc: List[float]) -> List[float]:
        # yc = yb - gc
        return [yb[i] - gc[i] for i in range(self.num_agent)]

    def _get_obs(self) -> np.ndarray:
        # state is (yb, gc)
        yb_arr = np.asarray(self.yb, dtype=np.float32)
        yb_norm = yb_arr - yb_arr[0]  # 消除绝对偏移，第一个元素变为0

        # 2. gc归一化：每个元素除以l_max
        gc_arr = np.asarray(self.gc, dtype=np.float32)
        gc_norm = gc_arr / self.l_max  # 归一化到[0,1]区间

        # 3. 拼接归一化后的yb和gc作为最终观测
        obs = np.concatenate([yb_norm, gc_norm], axis=0)
        return obs

    def _reward(self, yc: List[float], yb: List[float]) -> Tuple[float, Dict[str, float]]:

        n = self.num_agent
        yc_diff = max(yc) - min(yc)
        if yc_diff > (2 * self.d_limit):
            r1 = 1.0 - yc_diff / (2 * self.d_limit)
        else:
            sum1 = sum(abs(yc[i + 1] - yc[i]) for i in range(n - 1))
            r1 = 1.0 - (sum1 / (2 * (n - 1) * self.d_limit))
        yb_diff = max(yb) - min(yb)
        if yb_diff > (2 * self.d_limit):
            r2 = 1.0 - yb_diff / (2 * self.d_limit)
        else:
            sum2 = sum(abs(yb[i + 1] - yb[i]) for i in range(n - 1))
            r2 = 1.0 - (sum2 / (2 * (n - 1) * self.d_limit))
        r = self.w1 * r1 + self.w2 * r2

        info = {"r": float(r), "r1_yc": r1, "r2_yb": r2, "yc_diff": float(yc_diff), "yb_diff": float(yb_diff)}

        return float(r), info

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # 初始化 yb：差分在 [-d_limit, d_limit]，第一个为0
        csgb = self.rng.integers(low=-int(self.d_limit), high=int(self.d_limit) + 1, size=(self.num_agent - 1,))
        csgb = np.concatenate([np.array([0], dtype=int), csgb], axis=0)
        yb = np.cumsum(csgb).astype(float).tolist()

        gc = self.rng.integers(low=500, high=int(self.l_max) + 1, size=(self.num_agent,)).astype(float).tolist()
        self.yb = yb
        self.gc = gc
        obs = self._get_obs()
        info = {}
        return obs, info
    def step(self, action: np.ndarray):
        self.step_count += 1
        # action_phys: clip 到 [0, action_max]，再逐元素 clip 到当前 gc[i]
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, 0.0, 1.0)
        gc_arr = np.asarray(self.gc, dtype=np.float32)
        action_phys = np.floor(gc_arr * a).astype(np.int32)

        yc = self._compute_yc(self.yb, self.gc)
        new_yc, new_gc, new_yb = env_mode(
            yc, self.gc, action_phys, self.l_max, self.d_limit
        )
        self.yb = list(map(float, new_yb))
        self.gc = list(map(float, new_gc))
        reward, rinfo = self._reward(new_yc, self.yb)
        # done 规则
        terminated = bool(reward >= self.target_r)
        truncated = bool(self.step_count >= self.max_steps)
        obs = self._get_obs()
        info = {
            **rinfo,
            "step": self.step_count,
            "terminated_by_threshold": terminated,
        }
        return obs, reward, terminated, truncated, info
