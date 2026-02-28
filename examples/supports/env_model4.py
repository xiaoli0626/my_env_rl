import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Any, Dict, List, Optional, Tuple


def env_mode(
    arrange_yc_list: List[float],
    arrange_gc_list: List[float],
    action_list: List[float],
    l_max: float,
    d_limit: float,
    *,
    max_iter: int = 200,
    atol: float = 1e-7,
) -> Tuple[List[float], List[float], List[float]]:
    """动力学推进：输入 (yc, gc, action) -> 输出 (new_yc, new_gc, new_yb)。"""
    n = len(action_list)
    new_yc_list = [arrange_yc_list[i] + action_list[i] for i in range(n)]
    mqgc_list = [arrange_gc_list[i] - action_list[i] for i in range(n)]
    mqgb_list = [new_yc_list[i] + mqgc_list[i] for i in range(n)]
    l_max_list = [l_max - mqgc_list[i] for i in range(n)]

    action_sum = np.zeros(n, dtype=np.float64)
    saturated = np.zeros(n, dtype=bool)

    def apply_delta(i: int, delta: float) -> None:
        if saturated[i]:
            return
        delta = max(float(delta), 0.0)
        remain = l_max_list[i] - action_sum[i]
        if remain <= 0:
            saturated[i] = True
            return
        actual = min(delta, remain)
        if actual < remain:
            saturated[i] = False
        else:
            saturated[i] = True
        mqgb_list[i] += actual
        action_sum[i] += actual

    prev = None
    for _ in range(max_iter):
        if prev is not None and np.allclose(action_sum, prev, atol=atol, rtol=0.0):
            break
        prev = action_sum.copy()

        for i in range(n):
            if saturated[i]:
                continue
            if n == 1:
                saturated[i] = True
                continue
            if i == 0:
                delta = d_limit - mqgb_list[i] + mqgb_list[i + 1]
            elif i == n - 1:
                delta = d_limit - mqgb_list[i] + mqgb_list[i - 1]
            else:
                action1 = d_limit - mqgb_list[i] + mqgb_list[i + 1]
                action2 = d_limit - mqgb_list[i] + mqgb_list[i - 1]
                delta = min(action1, action2)
            apply_delta(i, delta)

    zxgc_list = [mqgc_list[i] + float(action_sum[i]) for i in range(n)]
    return new_yc_list, zxgc_list, mqgb_list


class YBGCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_agent: int = 10,
        d_limit: float = 50.0,
        l_max: float = 865.0,
        max_steps_per_episode: int = 6,
        success_r1_threshold: Optional[float] = None,
        w1: float = 0.8,
        w2: float = 0.2,
        seed: Optional[int] = None,
        min_gc_init: float = 500.0,
    ) -> None:
        super().__init__()
        self.num_agent = int(num_agent)
        self.d_limit = float(d_limit)
        self.l_max = float(l_max)
        self.max_steps = int(max_steps_per_episode)
        if success_r1_threshold is None:
            success_r1_threshold = 1 - 100 / ((self.num_agent - 1) * self.d_limit + self.l_max)
        self.success_r1_threshold = float(success_r1_threshold)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.min_gc_init = float(min_gc_init)

        yb_low = np.full((self.num_agent,), -self.num_agent * self.d_limit, dtype=np.float32)
        yb_high = np.full((self.num_agent,), self.num_agent * self.d_limit, dtype=np.float32)
        gc_low = np.zeros((self.num_agent,), dtype=np.float32)
        gc_high = np.ones((self.num_agent,), dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate([yb_low, gc_low]),
            high=np.concatenate([yb_high, gc_high]),
            dtype=np.float32,
        )
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_agent,), dtype=np.float32)

        self.yb: List[float] = []
        self.gc: List[float] = []
        self.step_count = 0
        self.episode_return = 0.0
        self.r1_reached_once = False
        self.rng = np.random.default_rng(seed)

    def _compute_yc(self, yb: List[float], gc: List[float]) -> List[float]:
        return [yb[i] - gc[i] for i in range(self.num_agent)]

    def _get_obs(self) -> np.ndarray:
        yb_arr = np.asarray(self.yb, dtype=np.float32)
        yb_rel = yb_arr - yb_arr[0]
        gc_arr = np.asarray(self.gc, dtype=np.float32)
        gc_norm = np.clip(gc_arr / self.l_max, 0.0, 1.0)
        obs = np.concatenate([yb_rel, gc_norm], axis=0).astype(np.float32)
        return obs

    def _reward(self, yc: List[float], a: List[float]) -> Tuple[float, Dict[str, float]]:
        n = self.num_agent
        yc_diff = max(yc) - min(yc)
        r1 = 1.0 - yc_diff / ((n - 1) * self.d_limit + self.l_max)
        sum1 = sum(a)
        r2 = sum1 / n
        r = np.clip(self.w1 * r1 + self.w2 * r2, 0.0, 1.0)
        info = {"r": float(r), "r1": float(r1), "r2": float(r2)}
        return float(r), info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.episode_return = 0.0
        self.r1_reached_once = False

        csgb = self.rng.integers(
            low=-int(self.d_limit),
            high=int(self.d_limit) + 1,
            size=(self.num_agent - 1,),
        )
        csgb = np.concatenate([np.array([0], dtype=np.int64), csgb], axis=0)
        self.yb = np.cumsum(csgb).astype(float).tolist()
        self.gc = self.rng.integers(
            low=int(self.min_gc_init),
            high=int(self.l_max) + 1,
            size=(self.num_agent,),
        ).astype(float).tolist()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (self.num_agent,):
            raise ValueError(f"action shape must be ({self.num_agent},), got {a.shape}")
        a = np.clip(a, 0.0, 1.0)

        gc_arr = np.asarray(self.gc, dtype=np.float32)
        action_phys = np.floor(gc_arr * a).astype(np.int32).tolist()
        yc = self._compute_yc(self.yb, self.gc)
        new_yc, new_gc, new_yb = env_mode(yc, self.gc, action_phys, self.l_max, self.d_limit)

        self.yb = list(map(float, new_yb))
        self.gc = [max(0.0, float(x)) for x in new_gc]

        reward, rinfo = self._reward(new_yc, a)
        self.episode_return += float(reward)

        success_now = bool(rinfo["r1"] >= self.success_r1_threshold)
        if success_now:
            self.r1_reached_once = True

        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        obs = self._get_obs()
        done = bool(terminated or truncated)

        info = {
            **rinfo,
            "step": self.step_count,
            "terminated_by_threshold": terminated,
            "terminated_by_r1_threshold": False,
            "success_r1_threshold": self.success_r1_threshold,
            "success_now": success_now,
            "is_success": float(self.r1_reached_once),
            "success": bool(self.r1_reached_once),
            "r1_reached_once": bool(self.r1_reached_once),
        }
        if done:
            info["episode_return"] = float(self.episode_return)

        return obs, reward, terminated, truncated, info


#
# def main() -> None:
#     # 1) 初始化环境
#     env = YBGCEnv(num_agent=10, max_steps_per_episode=6, seed=42)
#
#
#     obs, info = env.reset(seed=42)
#     print(obs)
#     print(info)
#     total_reward = 0.6*(1-100/((10 - 1) * 50 + 865))+0.4*(800/865)
#     max_iter = 20
#     for t in range(max_iter):
#         action = env.action_space.sample()  # 每维在 [0,1]
#         next_obs, reward, terminated, truncated, step_info = env.step(action)
#         total_reward += reward
#         assert env.observation_space.contains(next_obs), "obs 超出 observation_space"
#         assert np.all((action >= 0.0) & (action <= 1.0)), "随机动作不在 [0,1]"
#         if terminated or truncated:
#             break
# if __name__ == "__main__":
#     main()