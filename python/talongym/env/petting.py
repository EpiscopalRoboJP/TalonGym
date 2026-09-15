"""Phase 4 unused PettingZoo parallel wrapper around FTCAutoEnv / World."""

from __future__ import annotations

from typing import Any

import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import LoadedPresets

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover
    class ParallelEnv:  # type: ignore[no-redef]
        metadata: dict = {}


class FTCAutoParallelEnv(ParallelEnv):
    metadata = {"name": "talongym_ftc_auto_v0", "render_modes": ["none"]}

    def __init__(
        self,
        bundle: LoadedPresets | None = None,
        opponent_mode: str = "scripted",
        live_teammate: bool = True,
        shared_alliance_reward: bool = False,
        action_tier: str = "high_level_waypoint",
    ) -> None:
        super().__init__()
        self._gym = FTCAutoEnv(
            bundle=bundle,
            static_teammate=False,
            teammate_policy="independent" if live_teammate else "none",
            opponent_policy=opponent_mode if opponent_mode != "none" else "none",
            action_tier=action_tier,
            fill_others=False,
            shared_alliance_reward=shared_alliance_reward,
            record=False,
        )
        self.possible_agents = ["red_0", "red_1", "blue_0", "blue_1"]
        self.agents: list[str] = []
        self.shared_alliance_reward = shared_alliance_reward
        self.observation_spaces = {a: self._gym.observation_space for a in self.possible_agents}
        self.action_spaces = {a: self._gym.action_space for a in self.possible_agents}

    def observation_space(self, agent: str):
        return self._gym.observation_space

    def action_space(self, agent: str):
        return self._gym.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        opts = dict(options or {})
        opts.setdefault("live_teammate", True)
        opts.setdefault("opponent_mode", self._gym.opponent_policy if self._gym.opponent_policy != "none" else "scripted")
        opts.setdefault("teammate_policy", "independent")
        self._gym.reset(seed=seed, options=opts)
        self.agents = [rid for rid in self.possible_agents if rid in self._gym.world.robots]
        obs = {a: self._gym._obs(a) for a in self.agents}
        infos = {a: self._gym._info(0.0, 0.0, robot_id=a) for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, Any]):
        parsed: dict[str, dict[str, Any]] = {}
        for agent, act in actions.items():
            item = self._gym._parse_action(act)
            if item is None:
                parsed[agent] = {
                    "target_pose": np.array(
                        [
                            self._gym.world.robots[agent].body.x,
                            self._gym.world.robots[agent].body.y,
                            self._gym.world.robots[agent].body.heading,
                        ]
                    ),
                    "speed_frac": 0.2,
                    "mechanism": 0,
                }
            else:
                parsed[agent] = item
        self._gym._step_actions(parsed)
        truncated = self._gym.world.time_s >= self._gym.world.auto_s - 1e-9
        obs: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terms = {a: False for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        infos: dict[str, Any] = {}
        true_delta = 0.0
        for a in list(self.agents):
            info = self._gym._info(true_delta, 0.0, robot_id=a)
            info["true_score_delta"] = 0.0
            obs[a] = self._gym._obs(a)
            infos[a] = info
            rewards[a] = float(self._gym.world.true_score) if self.shared_alliance_reward else float(self._gym.world.true_score)
        if self.shared_alliance_reward:
            red = sum(rewards[a] for a in self.agents if a.startswith("red"))
            for a in self.agents:
                if a.startswith("red"):
                    rewards[a] = red
        if truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self):
        self._gym.close()
