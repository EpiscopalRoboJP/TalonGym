"""Season-declared training reward terms. No field poses or year-specific names."""

from __future__ import annotations

from typing import Any

KNOWN_TERM_KINDS = frozenset(
    {
        "trueScoreDelta",
        "earlyVolumeBonus",
        "collisionPenalty",
        "missedLaunchPenalty",
        "timeCost",
    }
)

DEFAULT_REWARD: dict[str, Any] = {"terms": [{"kind": "trueScoreDelta"}]}


def normalize_reward_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    terms = [dict(term) for term in ((raw or {}).get("terms") or [])]
    for term in terms:
        kind = term.get("kind")
        if kind not in KNOWN_TERM_KINDS:
            raise ValueError(f"unknown reward term kind {kind!r}")
    if not any(term.get("kind") == "trueScoreDelta" for term in terms):
        terms.insert(0, {"kind": "trueScoreDelta"})
    if not terms:
        terms = [{"kind": "trueScoreDelta"}]
    return {"terms": terms}


def resolve_reward_config(
    scoring: dict[str, Any] | None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Training-run `reward` overrides scoring; otherwise scoring; else true score only."""
    if training and isinstance(training.get("reward"), dict):
        return normalize_reward_config(training["reward"])
    if scoring and isinstance(scoring.get("reward"), dict):
        return normalize_reward_config(scoring["reward"])
    return normalize_reward_config(DEFAULT_REWARD)


def volume_ids_for_tag(
    elements: list[dict[str, Any]] | None,
    tag: str | None,
    alliance: str | None,
) -> list[str]:
    if not tag:
        return []
    ids: list[str] = []
    for element in elements or []:
        tags = element.get("tags") or []
        if tag not in tags:
            continue
        el_alliance = element.get("alliance")
        if el_alliance and alliance and el_alliance != alliance:
            continue
        ident = element.get("triggerId") or element.get("id")
        if ident:
            ids.append(str(ident))
    return ids


def _truthy(value: Any) -> bool:
    return bool(value) and value is not False


class RewardTracker:
    """Episode tracker: first occupancy times plus configured extra terms."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = normalize_reward_config(config)
        self._first: dict[int, float] = {}

    def reset(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self.config = normalize_reward_config(config)
        self._first.clear()

    def observe(
        self,
        *,
        robot_id: str,
        alliance: str | None,
        occupancy: dict[str, set[str]] | None,
        time_s: float,
        elements: list[dict[str, Any]] | None,
    ) -> None:
        occ = occupancy or {}
        for index, term in enumerate(self.config["terms"]):
            if term.get("kind") != "earlyVolumeBonus":
                continue
            if index in self._first:
                continue
            ids = volume_ids_for_tag(elements, term.get("volumeTag"), alliance)
            if any(robot_id in (occ.get(vid) or set()) for vid in ids):
                self._first[index] = float(time_s)

    def step(
        self,
        *,
        true_delta: float,
        phase_end: bool,
        accumulators: dict[str, Any] | None,
        true_score: float,
        phase_duration: float,
        wall_hit: bool = False,
        robot_hit: bool = False,
        piece_hit: bool = False,
        missed_launches: int = 0,
        dt: float = 0.0,
        robot_id: str = "",
        alliance: str | None = None,
        occupancy: dict[str, set[str]] | None = None,
        time_s: float = 0.0,
        elements: list[dict[str, Any]] | None = None,
    ) -> tuple[float, float]:
        self.observe(
            robot_id=robot_id,
            alliance=alliance,
            occupancy=occupancy,
            time_s=time_s,
            elements=elements,
        )
        extra = 0.0
        acc = accumulators or {}
        duration = max(float(phase_duration), 1e-9)
        for index, term in enumerate(self.config["terms"]):
            kind = term.get("kind")
            if kind == "trueScoreDelta":
                continue
            if kind == "earlyVolumeBonus":
                extra += self._early_volume_bonus(
                    term,
                    index,
                    phase_end=phase_end,
                    accumulators=acc,
                    true_score=true_score,
                    duration=duration,
                )
            elif kind == "collisionPenalty":
                if wall_hit:
                    extra -= float(term.get("wall") or 0.0)
                if robot_hit:
                    extra -= float(term.get("robot") or 0.0)
                if piece_hit:
                    extra -= float(term.get("piece") or 0.0)
            elif kind == "missedLaunchPenalty":
                extra -= float(term.get("perLaunch") or 0.0) * int(missed_launches)
            elif kind == "timeCost":
                extra -= float(term.get("perSecond") or 0.0) * float(dt)
        return float(true_delta) + extra, extra

    def _early_volume_bonus(
        self,
        term: dict[str, Any],
        index: int,
        *,
        phase_end: bool,
        accumulators: dict[str, Any],
        true_score: float,
        duration: float,
    ) -> float:
        when = str(term.get("on") or "phaseEnd")
        if when == "phaseEnd" and not phase_end:
            return 0.0
        require = term.get("requireAccumulator")
        if require and not _truthy(accumulators.get(require)):
            return 0.0
        t_first = self._first.get(index)
        if t_first is None:
            return 0.0
        excluded = 0.0
        for acc_id in term.get("excludeAccumulators") or []:
            excluded += float(accumulators.get(acc_id) or 0.0)
        gameplay = max(0.0, float(true_score) - excluded)
        scale = float(term.get("scale") or 0.0)
        frac = max(0.0, 1.0 - min(1.0, t_first / duration))
        return gameplay * scale * frac
