"""Optional RLlib trainer. Laptop default remains RecurrentPPO."""

from __future__ import annotations

from typing import Any, Callable


def train_rllib(
    total_steps: int = 2048,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = log or (lambda m: None)
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
    except ImportError as exc:
        raise RuntimeError("RLlib requires pip install -e '.[scale]' (ray[rllib])") from exc
    emit("Using Ray RLlib PPO (optional scale extra)")
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    try:
        from talongym.env.ftc_auto import FlatBoxEnv, FTCAutoEnv
        from talongym.training.compute import cuda_available

        def _make(_ctx=None):
            return FlatBoxEnv(FTCAutoEnv(record=False, static_teammate=False))

        # Ray/RLlib's GPU support is CUDA-only (no Apple Metal backend), so this only
        # ever requests a GPU on NVIDIA/ROCm machines; everywhere else it stays CPU.
        cfg = (
            PPOConfig()
            .environment(_make)
            .env_runners(num_env_runners=0)
            .training(train_batch_size=max(128, min(total_steps, 1024)))
            .framework("torch")
            .resources(num_gpus=1 if cuda_available() else 0)
        )
        algo = cfg.build()
        result = algo.train()
        algo.stop()
        return {"algo": "rllib_ppo", "steps": int(result.get("num_env_steps_sampled", total_steps)), "metrics": result}
    finally:
        ray.shutdown()
