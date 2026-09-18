"""Chunked CPU physics workers: a few processes, many DummyVecEnv worlds each."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from talongym.presets.loader import LoadedPresets
from talongym.training.compute import recommended_sim_workers

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError:  # pragma: no cover - train_ppo already requires [rl]
    VecEnv = object  # type: ignore[misc,assignment]


_WORKER_ERROR = "__talongym_worker_error__"


def _send_worker_error(remote: Any, exc: BaseException) -> None:
    try:
        remote.send((_WORKER_ERROR, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    except Exception:
        pass


def _recv_worker(remote: Any) -> Any:
    try:
        msg = remote.recv()
    except EOFError as exc:
        raise RuntimeError("sim worker died (EOF)") from exc
    except ConnectionResetError as exc:
        raise RuntimeError("sim worker died (connection reset)") from exc
    if (
        isinstance(msg, tuple)
        and len(msg) == 2
        and isinstance(msg[0], str)
        and msg[0] == _WORKER_ERROR
    ):
        raise RuntimeError(f"sim worker crashed:\n{msg[1]}")
    return msg


def mp_start_method() -> str:
    """Lab is multithreaded; never fork() from a FastAPI + training thread."""
    return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"


def mp_context() -> mp.context.BaseContext:
    return mp.get_context(mp_start_method())


def split_chunk_sizes(n_envs: int, n_workers: int) -> list[int]:
    n_envs = max(1, int(n_envs))
    n_workers = max(1, min(int(n_workers), n_envs))
    base, rem = divmod(n_envs, n_workers)
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def pin_mujoco_single_thread() -> None:
    """One OpenMP thread per process so N workers do not oversubscribe the machine."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import mujoco

        setter = getattr(mujoco, "mj_setNumThreads", None)
        if setter is not None:
            setter(1)
    except Exception:
        pass


def worker_env_from_spec(spec: dict[str, Any]) -> gym.Env:
    """Top-level, picklable env factory. Rebuilds presets from dicts (Lab SQLite overlays)."""
    pin_mujoco_single_thread()
    from talongym.env.ftc_auto import BoxActionDictObsEnv, EncoderOnlyObsAssertWrapper, FTCAutoEnv
    from talongym.training.curriculum import (
        motif_known_at_t0,
        opponent_for,
        teammate_for,
    )
    from talongym.training.ppo import CurriculumEnv, SharedFracProgress, load_trained_policy
    from talongym.training.privileged import PrivilegedObsWrapper

    bundle = LoadedPresets(
        field=spec["field"],
        robot=spec["robot"],
        scoring=spec["scoring"],
        training=spec.get("training"),
    )
    training = spec.get("training")
    progress = spec.get("progress")
    if progress is None:
        progress = SharedFracProgress(spec.get("progressFrac"))
    frozen = spec.get("frozenPolicy")
    path = spec.get("frozenPolicyPath")
    if frozen is None and path:
        frozen = load_trained_policy(path, device="cpu")
    frac = float(progress.frac)
    env = FTCAutoEnv(
        bundle=bundle,
        record=False,
        motif_known_at_t0=motif_known_at_t0(training, frac),
        teammate_policy=teammate_for(training, frac),
        opponent_policy=opponent_for(training, frac),
        action_tier=str(spec.get("actionTier") or "waypoint"),
        frozen_policy=frozen,
        match_setup=spec.get("matchSetup"),
    )
    wrapped = EncoderOnlyObsAssertWrapper(env)
    priv = PrivilegedObsWrapper(wrapped)
    boxed = BoxActionDictObsEnv(priv)
    return CurriculumEnv(boxed, progress, training)


def _env_thunk(spec: dict[str, Any]) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return worker_env_from_spec(spec)

    return _init


def _concat_obs(obs_list: Sequence[Any], observation_space: spaces.Space) -> Any:
    first = obs_list[0]
    if isinstance(observation_space, spaces.Dict) or isinstance(first, dict):
        return {key: np.concatenate([obs[key] for obs in obs_list], axis=0) for key in first}
    return np.concatenate(list(obs_list), axis=0)


class _RoundRobinFns:
    """Picklable thunk that yields a new env from a list of factories, in order."""

    def __init__(self, fns: list[Callable[[], gym.Env]]) -> None:
        self._fns = list(fns)
        self._i = 0

    def __call__(self) -> gym.Env:
        fn = self._fns[self._i % len(self._fns)]
        self._i += 1
        return fn()


def _chunk_worker(
    remote: Any,
    parent_remote: Any,
    spec: dict[str, Any] | None,
    n_chunk: int,
    env_fn: Callable[[], gym.Env] | None,
) -> None:
    parent_remote.close()
    pin_mujoco_single_thread()
    from stable_baselines3.common.vec_env import DummyVecEnv

    thunk: Callable[[], gym.Env]
    if env_fn is not None:
        thunk = env_fn.var if hasattr(env_fn, "var") else env_fn
    else:
        thunk = _env_thunk(spec or {})
    try:
        venv = DummyVecEnv([thunk for _ in range(int(n_chunk))])
    except Exception as exc:
        _send_worker_error(remote, exc)
        remote.close()
        return
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "step":
                obs, rews, dones, infos = venv.step(data)
                remote.send((obs, rews, dones, infos, list(venv.reset_infos)))
            elif cmd == "reset":
                seeds, options_list = data
                venv._seeds = list(seeds)
                venv._options = list(options_list)
                obs = venv.reset()
                remote.send((obs, list(venv.reset_infos)))
            elif cmd == "close":
                venv.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((venv.observation_space, venv.action_space))
            elif cmd == "get_attr":
                remote.send(venv.get_attr(data))
            elif cmd == "has_attr":
                remote.send(venv.has_attr(data))
            elif cmd == "set_attr":
                venv.set_attr(data[0], data[1])
                remote.send(None)
            elif cmd == "env_method":
                remote.send(venv.env_method(data[0], *data[1], **data[2]))
            elif cmd == "is_wrapped":
                remote.send(venv.env_is_wrapped(data))
            elif cmd == "render":
                remote.send(venv.get_images() if hasattr(venv, "get_images") else None)
            else:
                raise NotImplementedError(f"unknown chunk worker command {cmd!r}")
    except KeyboardInterrupt:
        venv.close()
        remote.close()
    except Exception as exc:
        _send_worker_error(remote, exc)
        try:
            venv.close()
        except Exception:
            pass
        try:
            remote.close()
        except Exception:
            pass


class ChunkedSubprocVecEnv(VecEnv):
    """SB3 VecEnv: N processes, each stepping a DummyVecEnv chunk of worlds."""

    def __init__(
        self,
        n_envs: int,
        n_workers: int,
        *,
        spec: dict[str, Any] | None = None,
        env_fns: list[Callable[[], gym.Env]] | None = None,
    ) -> None:
        if spec is None and not env_fns:
            raise ValueError("ChunkedSubprocVecEnv needs spec= or env_fns=")
        self.waiting = False
        self.closed = False
        n_envs = max(1, int(n_envs))
        sizes = split_chunk_sizes(n_envs, n_workers)
        self._sizes = sizes
        offsets: list[int] = []
        cursor = 0
        for size in sizes:
            offsets.append(cursor)
            cursor += size
        ctx = mp_context()
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in sizes])
        self.processes = []
        try:
            for i, (work_remote, remote, size) in enumerate(zip(self.work_remotes, self.remotes, sizes)):
                env_fn_for_worker: Callable[[], gym.Env] | None = None
                if env_fns is not None:
                    from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

                    env_fn_for_worker = CloudpickleWrapper(_RoundRobinFns(env_fns[offsets[i] : offsets[i] + size]))
                args = (work_remote, remote, spec, size, env_fn_for_worker)
                process = ctx.Process(target=_chunk_worker, args=args, daemon=True)
                process.start()
                self.processes.append(process)
                work_remote.close()

            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = _recv_worker(self.remotes[0])
        except Exception:
            self.close()
            raise
        super().__init__(n_envs, observation_space, action_space)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], n_envs: int, n_workers: int) -> ChunkedSubprocVecEnv:
        return cls(n_envs, n_workers, spec=spec)

    def step_async(self, actions: np.ndarray) -> None:
        start = 0
        for remote, size in zip(self.remotes, self._sizes):
            remote.send(("step", actions[start : start + size]))
            start += size
        self.waiting = True

    def step_wait(self):
        results = [_recv_worker(remote) for remote in self.remotes]
        self.waiting = False
        obs_parts, rews, dones, infos, reset_parts = zip(*results)
        self.reset_infos = [info for part in reset_parts for info in part]
        info_flat = [info for part in infos for info in part]
        return (
            _concat_obs(obs_parts, self.observation_space),
            np.concatenate(rews, axis=0),
            np.concatenate(dones, axis=0),
            info_flat,
        )

    def reset(self):
        start = 0
        for remote, size in zip(self.remotes, self._sizes):
            remote.send(("reset", (self._seeds[start : start + size], self._options[start : start + size])))
            start += size
        results = [_recv_worker(remote) for remote in self.remotes]
        obs_parts, reset_parts = zip(*results)
        self.reset_infos = [info for part in reset_parts for info in part]
        self._reset_seeds()
        self._reset_options()
        return _concat_obs(obs_parts, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except (EOFError, ConnectionResetError, OSError, BrokenPipeError):
                    pass
            self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        self.closed = True

    def get_images(self):
        for remote in self.remotes:
            remote.send(("render", None))
        images = []
        for remote in self.remotes:
            part = _recv_worker(remote)
            if part:
                images.extend(part)
        return images

    def has_attr(self, attr_name: str) -> bool:
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(_recv_worker(remote) for remote in self.remotes)

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        for remote in self.remotes:
            remote.send(("get_attr", attr_name))
        values: list[Any] = []
        for remote in self.remotes:
            values.extend(_recv_worker(remote))
        indices = self._get_indices(indices)
        return [values[i] for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for remote in self.remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in self.remotes:
            _recv_worker(remote)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        for remote in self.remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        values: list[Any] = []
        for remote in self.remotes:
            values.extend(_recv_worker(remote))
        indices = self._get_indices(indices)
        return [values[i] for i in indices]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None) -> list[bool]:
        for remote in self.remotes:
            remote.send(("is_wrapped", wrapper_class))
        values: list[bool] = []
        for remote in self.remotes:
            values.extend(_recv_worker(remote))
        indices = self._get_indices(indices)
        return [values[i] for i in indices]


def make_train_vec_env(
    n_envs: int,
    spec: dict[str, Any],
    *,
    n_workers: int | None = None,
) -> tuple[Any, int]:
    """DummyVecEnv when workers==1; otherwise chunked subprocesses. Returns (venv, n_workers)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    n_envs = max(1, int(n_envs))
    workers = recommended_sim_workers(n_envs) if n_workers is None else max(1, min(int(n_workers), n_envs))
    if workers <= 1:
        venv = DummyVecEnv([_env_thunk(spec) for _ in range(n_envs)])
        return venv, 1
    remote_spec = {k: v for k, v in spec.items() if k not in {"progress", "frozenPolicy"}}
    venv = ChunkedSubprocVecEnv.from_spec(remote_spec, n_envs, workers)
    return venv, workers
