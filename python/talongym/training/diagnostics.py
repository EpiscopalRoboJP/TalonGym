"""Episode health for training: launches, wall contact, mechanism verbs, and route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from talongym.presets.loader import LoadedPresets, load_bundle
from talongym.training.policies import scripted_auto

DEFAULT_WALL_CONTACT_LIMIT_S = 8.0
BASELINE_MIN_LAUNCHES = 3
BASELINE_MIN_SCORE = 3.0


class TrainingContractError(RuntimeError):
    """The env/scripted baseline cannot physically launch or score; do not train."""


@dataclass
class EpisodeHealth:
    true_score: float = 0.0
    launches: int = 0
    scored_pieces: int = 0
    missed_launches: int = 0
    wall_contact_s: float = 0.0
    held_terminal: int = 0
    fire_steps: int = 0
    intake_steps: int = 0
    idle_steps: int = 0
    steps: int = 0
    terminal_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reached_launch: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fire_frac(self) -> float:
        return 0.0 if self.steps <= 0 else self.fire_steps / self.steps

    @property
    def healthy(self) -> bool:
        return self.launches > 0 and self.wall_contact_s <= DEFAULT_WALL_CONTACT_LIMIT_S

    def as_metrics(self, prefix: str = "eval") -> dict[str, Any]:
        return {
            f"{prefix}LaunchCount": self.launches,
            f"{prefix}ScoredPieces": self.scored_pieces,
            f"{prefix}WallContactS": self.wall_contact_s,
            f"{prefix}HeldTerminal": self.held_terminal,
            f"{prefix}MechanismFireFrac": self.fire_frac,
            f"{prefix}ReachedLaunch": self.reached_launch,
            f"{prefix}CheckpointHealthy": self.healthy,
        }


def _actor(frame: dict[str, Any]) -> dict[str, Any]:
    robots = frame.get("robots") or []
    return next((row for row in robots if row.get("id") == "red_0"), robots[0] if robots else {})


def summarize_episode(frames: list[dict[str, Any]] | None) -> EpisodeHealth:
    rows = list(frames or [])
    if not rows:
        return EpisodeHealth(warnings=("no frames recorded",))
    last = rows[-1]
    actor = _actor(last)
    verbs = [str(_actor(frame).get("lastVerb") or "idle") for frame in rows]
    fire_steps = sum(1 for verb in verbs if verb == "score")
    intake_steps = sum(1 for verb in verbs if verb == "intake")
    launched_ids: set[str] = set()
    for frame in rows:
        for piece in frame.get("pieces") or []:
            if piece.get("launchedBy") or piece.get("launchedAt") is not None or piece.get("inFlight") or piece.get("scored"):
                launched_ids.add(str(piece.get("id") or ""))
    scored = len({str(piece.get("id")) for piece in last.get("pieces") or [] if piece.get("scored")})
    launched = {str(piece.get("id")) for piece in last.get("pieces") or [] if piece.get("launchedBy") or piece.get("launchedAt") is not None}
    in_flight = {str(piece.get("id")) for piece in last.get("pieces") or [] if piece.get("inFlight")}
    launches = max(len(launched_ids), len(launched | in_flight), int(last.get("launchAttempts") or 0))
    pose = (float(actor.get("x") or 0.0), float(actor.get("y") or 0.0), float(actor.get("headingDeg") or 0.0))
    warnings: list[str] = []
    if launches <= 0:
        warnings.append("zero physical launches")
    wall_s = 0.0
    prev_t: float | None = None
    for frame in rows:
        t = float(frame.get("t") or 0.0)
        if prev_t is not None and (frame.get("collision") or {}).get("wall"):
            wall_s += max(0.0, t - prev_t)
        prev_t = t
    wall_s = max(
        wall_s,
        float(actor.get("collisionTimeS") or (last.get("collision") or {}).get("collisionTimeS") or 0.0),
    )
    if wall_s > DEFAULT_WALL_CONTACT_LIMIT_S:
        warnings.append(f"wall contact {wall_s:.1f}s exceeds {DEFAULT_WALL_CONTACT_LIMIT_S:.0f}s")
    if fire_steps <= 0:
        warnings.append("mechanism never issued score")
    return EpisodeHealth(
        true_score=float(last.get("trueScore") or 0.0),
        launches=launches,
        scored_pieces=scored,
        missed_launches=int(last.get("missedLaunches") or 0),
        wall_contact_s=wall_s,
        held_terminal=len(actor.get("held") or []),
        fire_steps=fire_steps,
        intake_steps=intake_steps,
        idle_steps=sum(1 for verb in verbs if verb in {"idle", ""}),
        steps=len(rows),
        terminal_pose=pose,
        reached_launch=any(verb == "score" for verb in verbs) or launches > 0,
        warnings=tuple(warnings),
    )


def record_and_summarize(
    policy,
    bundle: LoadedPresets | None = None,
    *,
    seed: int = 0,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], EpisodeHealth]:
    from talongym.env.ftc_auto import FTCAutoEnv

    env = FTCAutoEnv(bundle=bundle or load_bundle(), record=True)
    obs, info = env.reset(seed=seed, options=dict(options or {}))
    term = trunc = False
    while not term and not trunc:
        action = policy(obs, info)
        obs, _, term, trunc, info = env.step(action)
    frames = list(env.frames)
    env.close()
    return frames, summarize_episode(frames)


def assert_scripted_baseline_scores(bundle: LoadedPresets | None = None, *, seed: int = 1) -> EpisodeHealth:
    """Fail closed if the physical scripted AUTO cannot launch and score preloads."""
    bundle = bundle or load_bundle()
    _, at_spot = record_and_summarize(
        scripted_auto,
        bundle,
        seed=seed,
        options={"full_noise": False, "curriculum_spawn": "launch", "static_teammate": False},
    )
    if at_spot.launches < BASELINE_MIN_LAUNCHES:
        raise TrainingContractError(
            "scripted AUTO launched too few pieces from the launch pose; "
            "fix the robot/field physical launch path before training. "
            + "; ".join(at_spot.warnings)
        )
    if at_spot.true_score < BASELINE_MIN_SCORE and at_spot.scored_pieces < 1:
        raise TrainingContractError(
            "scripted AUTO fired but scored nothing from the launch pose; "
            "physical occupancy scoring is broken. "
            + "; ".join(at_spot.warnings)
        )
    return at_spot


def mean_health(rows: list[EpisodeHealth]) -> EpisodeHealth:
    if not rows:
        return EpisodeHealth(warnings=("no eval episodes",))
    return EpisodeHealth(
        true_score=float(np.mean([row.true_score for row in rows])),
        launches=int(round(float(np.mean([row.launches for row in rows])))),
        scored_pieces=int(round(float(np.mean([row.scored_pieces for row in rows])))),
        missed_launches=int(round(float(np.mean([row.missed_launches for row in rows])))),
        wall_contact_s=float(np.mean([row.wall_contact_s for row in rows])),
        held_terminal=int(round(float(np.mean([row.held_terminal for row in rows])))),
        fire_steps=int(round(float(np.mean([row.fire_steps for row in rows])))),
        intake_steps=int(round(float(np.mean([row.intake_steps for row in rows])))),
        idle_steps=int(round(float(np.mean([row.idle_steps for row in rows])))),
        steps=int(round(float(np.mean([row.steps for row in rows])))),
        terminal_pose=rows[0].terminal_pose,
        reached_launch=any(row.reached_launch for row in rows),
        warnings=tuple(dict.fromkeys(warning for row in rows for warning in row.warnings)),
    )
