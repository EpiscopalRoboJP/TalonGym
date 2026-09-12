from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.rules.engine import MECHANISM_VERBS
from talongym.sim.world import World


class FTCAutoEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 25}

    def __init__(
        self,
        bundle: LoadedPresets | None = None,
        control_hz: int | None = None,
        static_teammate: bool = True,
        motif_known_at_t0: bool = False,
        record: bool = False,
        teammate_policy: str = "none",
        opponent_policy: str = "none",
        frozen_policy: Any = None,
        action_tier: str = "high_level_waypoint",
        learner_id: str = "red_0",
        fill_others: bool = True,
        shared_alliance_reward: bool = False,
    ) -> None:
        super().__init__()
        self.bundle = bundle or load_bundle()
        ep = (self.bundle.training or {}).get("episode") or {}
        self.control_hz = int(control_hz or ep.get("controlHz") or 25)
        substeps = int(ep.get("physicsSubsteps") or 2)
        self.static_teammate = static_teammate
        self.motif_known_at_t0 = motif_known_at_t0
        self.record = record
        self.teammate_policy = teammate_policy
        self.opponent_policy = opponent_policy
        self.frozen_policy = frozen_policy
        self.action_tier = action_tier
        self.learner_id = learner_id
        self.fill_others = fill_others
        self.shared_alliance_reward = shared_alliance_reward or teammate_policy == "shared_reward"
        self.world = World(self.bundle, control_hz=self.control_hz, substeps=substeps)
        self.K = 6
        self.M = 6
        field = self.bundle.field
        fw = float(field["fieldSizeIn"]["width"]) / 2.0
        fd = float(field["fieldSizeIn"]["depth"]) / 2.0
        cap = int((self.bundle.robot.get("mechanisms") or {}).get("capacity", 3))
        self.capacity = cap
        vars_spec = list(self.bundle.scoring.get("matchVariables") or [])
        self.match_var_specs = vars_spec
        enums = [list((v.get("domain") or {}).get("enum") or []) for v in vars_spec]
        self.match_enums = enums or [["A", "B", "C"]]
        enum_w = max((len(e) for e in self.match_enums), default=3)
        n_vars = max(1, len(vars_spec))
        self.observation_space = spaces.Dict(
            {
                "pose_noisy": spaces.Box(low=np.array([-fw, -fd, -np.pi], np.float32), high=np.array([fw, fd, np.pi], np.float32)),
                "vel_noisy": spaces.Box(-60, 60, shape=(3,), dtype=np.float32),
                "time_remaining_s": spaces.Box(0, 30, shape=(1,), dtype=np.float32),
                "held_count": spaces.Box(0, cap, shape=(1,), dtype=np.float32),
                "held_colors": spaces.Box(0, 1, shape=(cap,), dtype=np.float32),
                "nearest_pieces": spaces.Box(-200, 200, shape=(self.K, 6), dtype=np.float32),
                "triggers": spaces.Box(0, 1, shape=(4,), dtype=np.float32),
                "vision_tags": spaces.Box(-10, 200, shape=(self.M, 5), dtype=np.float32),
                "match_var_obs": spaces.Box(0, 1, shape=(n_vars, enum_w + 1), dtype=np.float32),
                "teammate_pose_noisy": spaces.Box(-200, 200, shape=(3,), dtype=np.float32),
                "collision": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            }
        )
        if action_tier == "low_level_velocity":
            self.action_space = spaces.Dict(
                {
                    "velocity": spaces.Box(-60, 60, shape=(3,), dtype=np.float32),
                    "mechanism": spaces.Discrete(len(MECHANISM_VERBS)),
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    "target_pose": spaces.Box(low=np.array([-fw, -fd, -np.pi], np.float32), high=np.array([fw, fd, np.pi], np.float32)),
                    "speed_frac": spaces.Box(0.2, 1.0, shape=(1,), dtype=np.float32),
                    "mechanism": spaces.Discrete(len(MECHANISM_VERBS)),
                }
            )
        self.frames: list[dict[str, Any]] = []
        self._last_potential = 0.0
        self._waypoint_log: list[list[float]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        opts = options or {}
        if opts.get("teammate_policy") is not None:
            self.teammate_policy = str(opts["teammate_policy"])
        if opts.get("opponent_policy") is not None:
            self.opponent_policy = str(opts["opponent_policy"])
        teammate = opts.get("teammate_policy", self.teammate_policy)
        opponent = opts.get("opponent_policy", self.opponent_policy)
        live = teammate in {"scripted", "shared_reward", "independent"}
        static = self.static_teammate if teammate in {"none", None} else False
        if teammate == "none" and not live:
            static = opts.get("static_teammate", self.static_teammate)
        opp_mode = opponent if opponent not in {None, "none"} else "none"
        if opts.get("opponent_mode"):
            opp_mode = opts["opponent_mode"]
        full_noise = bool(opts.get("full_noise", True))
        self.world.reset(
            seed=seed,
            static_teammate=static,
            opponent_mode=opp_mode,
            live_teammate=live or bool(opts.get("live_teammate")),
            full_noise=full_noise,
        )
        if opts.get("action_tier"):
            self.action_tier = opts["action_tier"]
        self._match_revealed = bool(self.motif_known_at_t0 or opts.get("motif_known_at_t0"))
        if self._match_revealed:
            for key, val in self.world.match_vars.items():
                self.world.observed_vars[key] = val
        self.frames = [self.world.snapshot()] if self.record else []
        self._waypoint_log = []
        self._last_potential = self._potential()
        return self._obs(self.learner_id), self._info(0.0, 0.0)

    def step(self, action: dict[str, Any] | np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        parsed = self._parse_action(action)
        if parsed is None:
            info = self._info(0.0, -10.0)
            return self._obs(self.learner_id), -10.0, True, False, info
        actions = {self.learner_id: parsed}
        if self.fill_others:
            actions.update(self._filled_others())
        return self._step_actions(actions)

    def _step_actions(self, actions: dict[str, dict[str, Any]]) -> tuple[dict, float, bool, bool, dict]:
        learner = actions.get(self.learner_id) or {}
        if "target_pose" in learner:
            tp = np.asarray(learner["target_pose"], dtype=np.float64).reshape(3)
            self._waypoint_log.append([float(tp[0]), float(tp[1]), float(tp[2])])
        remaining_before = self.world.auto_s - self.world.time_s
        will_end = remaining_before - (1.0 / self.control_hz) <= 1e-6
        result = self.world.step(end_phase=will_end, actions=actions)
        true_delta = float(result["true_score_delta"])
        potential = self._potential()
        shaping = (potential - self._last_potential) - 0.01 / self.control_hz
        if self.world.wall_hit:
            shaping -= 0.5
        if self.world.robot_hit:
            shaping -= 2.0
        if self.world.piece_hit:
            shaping -= 0.05
        self._last_potential = potential
        objective = true_delta + shaping
        truncated = self.world.time_s >= self.world.auto_s - 1e-9
        if self.record:
            self.frames.append(self.world.snapshot())
        return self._obs(self.learner_id), float(objective), False, truncated, self._info(true_delta, shaping)

    def _filled_others(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rid, rs in self.world.robots.items():
            if rid == self.learner_id or not rs.body.dynamic:
                continue
            policy = self._policy_for(rid)
            if policy is None:
                continue
            act = policy(self._obs(rid), self._info(0.0, 0.0, robot_id=rid))
            parsed = self._parse_action(act)
            if parsed:
                out[rid] = parsed
        return out

    def _policy_for(self, rid: str):
        from talongym.training.policies import scripted_auto

        alliance = rid.split("_")[0]
        if alliance == "red" and rid != self.learner_id:
            if self.teammate_policy in {"scripted", "shared_reward", "independent"}:
                return scripted_auto
            return None
        if alliance == "blue":
            if self.opponent_policy == "scripted":
                return scripted_auto
            if self.opponent_policy == "frozen_policy" and self.frozen_policy is not None:
                return self.frozen_policy
        return None

    def _parse_action(self, action: dict[str, Any] | np.ndarray) -> dict[str, Any] | None:
        if isinstance(action, dict):
            if "velocity" in action or (self.action_tier == "low_level_velocity" and "target_pose" not in action):
                vel = np.asarray(action.get("velocity", action.get("target_pose", [0, 0, 0])), dtype=np.float64).reshape(-1)
                if vel.size < 3 or not np.all(np.isfinite(vel[:3])):
                    return None
                return {"velocity": vel[:3], "mechanism": int(action.get("mechanism", 0))}
            target = np.asarray(action["target_pose"], dtype=np.float64).reshape(3)
            speed = float(np.asarray(action["speed_frac"]).reshape(-1)[0])
            mech = int(action["mechanism"])
            if not np.all(np.isfinite(target)) or not np.isfinite(speed):
                return None
            return {"target_pose": target, "speed_frac": speed, "mechanism": mech}
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.action_tier == "low_level_velocity":
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return None
            mech = int(np.clip(np.round(arr[3] if arr.size > 3 else 0), 0, len(MECHANISM_VERBS) - 1))
            return {"velocity": arr[:3], "mechanism": mech}
        if arr.size < 5:
            return None
        target = arr[:3]
        speed = float(arr[3])
        mech = int(np.clip(np.round(arr[4]), 0, len(MECHANISM_VERBS) - 1))
        if not np.all(np.isfinite(target)) or not np.isfinite(speed):
            return None
        return {"target_pose": target, "speed_frac": speed, "mechanism": mech}

    def _potential(self) -> float:
        rs = self.world.robots.get(self.learner_id) or self.world.actor()
        floor = [
            p
            for p in self.world.pieces.values()
            if not p.held_by and not p.scored and not p.in_flight
        ]
        if rs.held:
            goal = next((e for e in self.world.elements if e.get("type") == "goal" and e.get("alliance") == rs.body.alliance), None)
            if goal:
                gx, gy = float(goal["pose"]["x"]), float(goal["pose"]["y"])
                return -0.02 * math_hypot(rs.body.x - gx, rs.body.y - gy)
        if floor:
            p = min(floor, key=lambda q: (q.x - rs.body.x) ** 2 + (q.y - rs.body.y) ** 2)
            return -0.02 * math_hypot(rs.body.x - p.x, rs.body.y - p.y)
        return 0.0

    def _obs(self, robot_id: str | None = None) -> dict[str, np.ndarray]:
        rid = robot_id or self.learner_id
        rs = self.world.robots.get(rid) or self.world.actor()
        noisy = np.array(
            [
                rs.body.x + float(self.world.rng.normal(0, self.world.pos_noise)),
                rs.body.y + float(self.world.rng.normal(0, self.world.pos_noise)),
                wrap_heading(rs.body.heading + float(self.world.rng.normal(0, self.world.heading_noise))),
            ],
            dtype=np.float32,
        )
        held_colors = np.zeros(self.capacity, dtype=np.float32)
        for i, pid in enumerate(rs.held[: self.capacity]):
            color = self.world.pieces[pid].attrs.get("color")
            held_colors[i] = 1.0 if color == "G" else 0.5
        pieces = np.zeros((self.K, 6), dtype=np.float32)
        floor = [p for p in self.world.pieces.values() if not p.scored]
        floor.sort(key=lambda p: (p.x - rs.body.x) ** 2 + (p.y - rs.body.y) ** 2)
        for i, p in enumerate(floor[: self.K]):
            pieces[i] = [
                p.x - rs.body.x,
                p.y - rs.body.y,
                1.0 if "green" in p.type_id else 0.0,
                1.0 if p.attrs.get("color") == "G" else 0.0,
                1.0,
                1.0 if p.held_by and p.held_by != rs.body.id else 0.0,
            ]
        vision = np.zeros((self.M, 5), dtype=np.float32)
        hits = rs.vision_hits or self.world.vision_hits
        for i, hit in enumerate(hits[: self.M]):
            vision[i] = [
                hit["tagId"] / 30.0,
                hit["bearing"],
                hit["range"],
                1.0 if hit["visible"] else 0.0,
                1.0 if hit["occluded"] else 0.0,
            ]
        n_vars, enum_w = self.observation_space["match_var_obs"].shape
        match_obs = np.zeros((n_vars, enum_w), dtype=np.float32)
        for i, spec in enumerate(self.match_var_specs or [{"id": "_"}]):
            domain = list((spec.get("domain") or {}).get("enum") or [])
            observed = self.world.observed_vars.get(spec.get("id"))
            if observed in domain:
                match_obs[i, domain.index(observed)] = 1.0
            else:
                match_obs[i, -1] = 1.0
        teammate = np.zeros(3, dtype=np.float32)
        mate_id = "red_1" if rid != "red_1" else "red_0"
        if mate_id in self.world.robots:
            t = self.world.robots[mate_id].body
            teammate = np.array([t.x, t.y, t.heading], dtype=np.float32)
        occ = self.world.prev_occupancy
        trig = np.array(
            [
                1.0 if rs.body.id in occ.get("red_start_zone", set()) else 0.0,
                1.0 if rs.body.id in occ.get("red_gate_contact", set()) else 0.0,
                1.0 if rs.body.id in occ.get("red_square", set()) else 0.0,
                1.0 if rs.held else 0.0,
            ],
            dtype=np.float32,
        )
        return {
            "pose_noisy": noisy,
            "vel_noisy": np.array([rs.body.vx, rs.body.vy, rs.body.omega], dtype=np.float32),
            "time_remaining_s": np.array([max(0.0, self.world.auto_s - self.world.time_s)], dtype=np.float32),
            "held_count": np.array([len(rs.held)], dtype=np.float32),
            "held_colors": held_colors,
            "nearest_pieces": pieces,
            "triggers": trig,
            "vision_tags": vision,
            "match_var_obs": match_obs,
            "teammate_pose_noisy": teammate,
            "collision": np.array([1.0 if self.world.wall_hit or self.world.robot_hit else 0.0], dtype=np.float32),
        }

    def _info(self, true_delta: float, shaping: float, robot_id: str | None = None) -> dict[str, Any]:
        rid = robot_id or self.learner_id
        rs = self.world.robots.get(rid) or self.world.actor()
        return {
            "true_score": self.world.true_score,
            "true_score_delta": true_delta,
            "shaping": shaping,
            "collision_time_s": rs.collision_time_s,
            "first_contact_s": rs.first_contact_s,
            "entered_restricted": rs.entered_restricted,
            "privileged": {
                "matchVars": dict(self.world.match_vars),
                "pose": [rs.body.x, rs.body.y, rs.body.heading],
                "season": (self.bundle.field.get("season") or {}).get("slug"),
                "restrictedEntry": bool(self.world.accumulators.get("restricted_entry")),
            },
            "waypoint_log": list(self._waypoint_log),
        }


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(x, y))


def wrap_heading(h: float) -> float:
    return float((h + np.pi) % (2 * np.pi) - np.pi)


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(obs)]
    return np.concatenate(parts)


class FlatBoxEnv(gym.Env):
    """SB3-friendly Box obs/action wrapper around FTCAutoEnv (RLlib / legacy)."""

    def __init__(self, env: FTCAutoEnv) -> None:
        super().__init__()
        self.env = env
        sample = flatten_obs(env.observation_space.sample())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=sample.shape, dtype=np.float32)
        if getattr(env, "action_tier", "high_level_waypoint") == "low_level_velocity":
            self.action_space = spaces.Box(-60, 60, shape=(4,), dtype=np.float32)
        else:
            pose = env.action_space["target_pose"]
            self.action_space = spaces.Box(
                low=np.concatenate([pose.low, np.array([0.2, 0.0], np.float32)]),
                high=np.concatenate([pose.high, np.array([1.0, float(len(MECHANISM_VERBS) - 1)], np.float32)]),
                dtype=np.float32,
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return flatten_obs(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return flatten_obs(obs), rew, term, trunc, info

    def close(self):
        self.env.close()


class BoxActionDictObsEnv(gym.Wrapper):
    """Keep Dict observations for MultiInputLstmPolicy; flatten mixed actions for SB3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.unwrapped
        self._tier = getattr(base, "action_tier", "high_level_waypoint")
        if self._tier == "low_level_velocity":
            self.action_space = spaces.Box(-60, 60, shape=(4,), dtype=np.float32)
        else:
            pose = base.action_space["target_pose"]
            n_mech = float(len(MECHANISM_VERBS) - 1)
            self.action_space = spaces.Box(
                low=np.concatenate([pose.low, np.array([0.2, 0.0], np.float32)]),
                high=np.concatenate([pose.high, np.array([1.0, n_mech], np.float32)]),
                dtype=np.float32,
            )


class EncoderOnlyObsAssertWrapper(gym.Wrapper):
    """Fail closed if a match variable is observed while its vision tag is not visible."""

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._assert_obs(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._assert_obs(obs)
        return obs, info

    def _assert_obs(self, obs: dict[str, np.ndarray]) -> None:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        world = getattr(env, "world", None)
        if world is None:
            return
        if getattr(env, "_match_revealed", False):
            return
        visible_roles = {hit.get("role") for hit in world.vision_hits if hit.get("visible")}
        for i, spec in enumerate(getattr(env, "match_var_specs", []) or []):
            role = (spec.get("observeVia") or {}).get("tagRole")
            if not role:
                continue
            row = obs["match_var_obs"][i]
            observed = row[-1] < 0.5
            if observed and role not in visible_roles:
                raise AssertionError(f"match var {spec.get('id')} observed without visible tag role {role}")

