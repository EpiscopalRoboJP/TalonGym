from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from talongym.env.ftc_auto import FTCAutoEnv
from talongym.presets.loader import LoadedPresets
from talongym.training.compute import recommended_eval_workers
from talongym.training.curriculum import objective_value
from talongym.training.vec_env import mp_context


def bootstrap_ci(samples: list[float], confidence: float = 0.95, n_boot: int = 2000, rng: np.random.Generator | None = None) -> dict[str, float]:
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0}
    means = []
    n = arr.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(float(arr[idx].mean()))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_boot)]
    hi = means[int((1.0 - alpha) * n_boot)]
    return {
        "mean": float(arr.mean()),
        "lo": lo,
        "hi": hi,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def policy_spec_for(policy: Callable[[dict, dict], Any]) -> dict[str, Any] | None:
    """Picklable policy descriptor, or None if this callable must stay in-process (lambdas)."""
    from talongym.training.policies import scripted_auto

    if policy is scripted_auto:
        return {"kind": "scripted"}
    path = getattr(policy, "path", None)
    if path:
        return {"kind": "checkpoint", "path": str(path)}
    return None


def _bundle_payload(bundle: LoadedPresets | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "field": bundle.field,
        "robot": bundle.robot,
        "scoring": bundle.scoring,
        "training": bundle.training,
    }


def _bundle_from_payload(payload: dict[str, Any] | None) -> LoadedPresets | None:
    if not payload:
        return None
    return LoadedPresets(
        field=payload["field"],
        robot=payload["robot"],
        scoring=payload["scoring"],
        training=payload.get("training"),
    )


def _load_eval_policy(spec: dict[str, Any]) -> Callable[[dict, dict], Any]:
    if spec.get("kind") == "checkpoint":
        from talongym.training.ppo import load_trained_policy

        return load_trained_policy(spec["path"], device="cpu")
    from talongym.training.policies import scripted_auto

    return scripted_auto


def _run_one(
    policy: Callable[[dict, dict], Any],
    seed: int,
    bundle: LoadedPresets | None,
    record: bool,
    match_setup: dict[str, Any] | None,
) -> dict[str, Any]:
    if hasattr(policy, "reset_lstm"):
        policy.reset_lstm()
    env = FTCAutoEnv(bundle=bundle, record=record, match_setup=match_setup)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    hit = False
    while not terminated and not truncated:
        action = policy(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        if float(obs["collision"][0]) > 0.5:
            hit = True
    rs = env.world.actor()
    result = {
        "score": float(info["true_score"]),
        "collision": hit,
        "collision_time_s": float(rs.collision_time_s),
        "first_contact_s": rs.first_contact_s,
        "entered_restricted": bool(rs.entered_restricted or env.world.accumulators.get("restricted_entry")),
        "frames": list(env.frames) if record else [],
        "seed": int(seed),
    }
    env.close()
    return result


def _eval_job(job: dict[str, Any]) -> dict[str, Any]:
    policy = _load_eval_policy(job["policy"])
    bundle = _bundle_from_payload(job.get("bundle"))
    return _run_one(policy, int(job["seed"]), bundle, False, job.get("matchSetup"))


def _rows_from_sequential(
    seed_list: list[int],
    policy: Callable[[dict, dict], Any],
    bundle: LoadedPresets | None,
    record_best: bool,
    match_setup: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return [_run_one(policy, seed, bundle, record_best, match_setup) for seed in seed_list]


def run_trials(
    n_trials: int,
    policy: Callable[[dict, dict], Any],
    bundle: LoadedPresets | None = None,
    seed0: int = 10_000_000,
    record_best: bool = True,
    seeds: list[int] | None = None,
    match_setup: dict[str, Any] | None = None,
    objective: str = "mean_true_score",
) -> dict[str, Any]:
    seed_list = list(seeds) if seeds is not None else [seed0 + i for i in range(n_trials)]
    n_workers = recommended_eval_workers(len(seed_list))
    spec = policy_spec_for(policy)
    if spec is not None and n_workers > 1 and len(seed_list) > 1:
        jobs = [
            {
                "policy": spec,
                "seed": seed,
                "bundle": _bundle_payload(bundle),
                "matchSetup": match_setup,
            }
            for seed in seed_list
        ]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context()) as pool:
            rows = list(pool.map(_eval_job, jobs))
        if record_best and rows:
            best = max(rows, key=lambda row: float(row["score"]))
            recorded = _run_one(policy, int(best["seed"]), bundle, True, match_setup)
            best["frames"] = recorded["frames"]
    else:
        rows = _rows_from_sequential(seed_list, policy, bundle, record_best, match_setup)

    scores: list[float] = []
    best_frames: list[dict] = []
    best_score = -1e9
    collisions = 0
    collision_times: list[float] = []
    first_contacts: list[float] = []
    restricted = 0
    for row in rows:
        scores.append(row["score"])
        collision_times.append(row["collision_time_s"])
        if row["first_contact_s"] is not None:
            first_contacts.append(float(row["first_contact_s"]))
        if row["collision"]:
            collisions += 1
        if row["entered_restricted"]:
            restricted += 1
        if record_best and row["score"] > best_score:
            best_score = row["score"]
            best_frames = row["frames"]
    n = max(1, len(seed_list))
    report = bootstrap_ci(scores)
    report.update(
        {
            "nTrials": len(seed_list),
            "scores": scores,
            "seeds": seed_list,
            "collisionRate": collisions / n,
            "collisionTimeMean": float(np.mean(collision_times)) if collision_times else 0.0,
            "firstContactS": float(np.mean(first_contacts)) if first_contacts else None,
            "restrictedEntryRate": restricted / n,
            "bestScore": best_score,
            "bestFrames": best_frames,
            "bestLabelEligible": len(seed_list) >= 500,
            "objective": objective,
            "objectiveValue": objective_value(report, objective),
            "evalWorkers": n_workers if spec is not None and n_workers > 1 and len(seed_list) > 1 else 1,
        }
    )
    return report


def run_paired_trials(
    policy_a: Callable[[dict, dict], Any],
    policy_b: Callable[[dict, dict], Any],
    n_trials: int,
    bundle: LoadedPresets | None = None,
    seed0: int = 10_000_000,
    pair_common_random_numbers: bool = True,
) -> dict[str, Any]:
    seeds = [seed0 + i for i in range(n_trials)]
    a = run_trials(n_trials, policy_a, bundle=bundle, seed0=seed0, record_best=False, seeds=seeds)
    b_seeds = list(seeds) if pair_common_random_numbers else [seed0 + 1_000_000 + i for i in range(n_trials)]
    b = run_trials(n_trials, policy_b, bundle=bundle, seed0=seed0, record_best=False, seeds=b_seeds)
    overlap = not (a.get("hi", 0) < b.get("lo", 0) or b.get("hi", 0) < a.get("lo", 0))
    eligible = n_trials >= 500 and not overlap
    return {
        "a": a,
        "b": b,
        "paired": pair_common_random_numbers,
        "ciOverlap": overlap,
        "bestLabelEligible": eligible,
    }
