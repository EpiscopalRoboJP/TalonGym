import pytest

from talongym.presets.loader import load_preset, validate_document
from talongym.training.reward import RewardTracker, resolve_reward_config, volume_ids_for_tag

PARK_ELEMENTS = [
    {
        "id": "home_pad",
        "triggerId": "home_pad",
        "alliance": "red",
        "tags": ["nest"],
        "isTrigger": True,
    },
    {
        "id": "away_pad",
        "triggerId": "away_pad",
        "alliance": "blue",
        "tags": ["nest"],
        "isTrigger": True,
    },
]


def _bonus_cfg(**overrides):
    term = {
        "kind": "earlyVolumeBonus",
        "volumeTag": "nest",
        "requireAccumulator": "nested",
        "excludeAccumulators": ["endgame_points"],
        "scale": 0.1,
        "on": "phaseEnd",
    }
    term.update(overrides)
    return {"terms": [{"kind": "trueScoreDelta"}, term]}


def test_training_reward_overrides_scoring():
    scoring = {"reward": {"terms": [{"kind": "trueScoreDelta"}, {"kind": "timeCost", "perSecond": 1.0}]}}
    training = {"reward": {"terms": [{"kind": "trueScoreDelta"}]}}
    cfg = resolve_reward_config(scoring, training)
    assert [t["kind"] for t in cfg["terms"]] == ["trueScoreDelta"]


def test_scoring_reward_used_when_training_omits():
    scoring = load_preset("scoring", "biobuzz_2026_scoring_v1")
    cfg = resolve_reward_config(scoring, {"algorithm": {"name": "recurrent_ppo"}})
    kinds = [t["kind"] for t in cfg["terms"]]
    assert kinds == ["trueScoreDelta", "earlyVolumeBonus"]
    bonus = next(t for t in cfg["terms"] if t["kind"] == "earlyVolumeBonus")
    assert bonus["volumeTag"] == "park"
    assert bonus["requireAccumulator"] == "parked_auto"


def test_default_is_true_score_only():
    cfg = resolve_reward_config({}, None)
    assert cfg["terms"] == [{"kind": "trueScoreDelta"}]


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown reward term kind"):
        resolve_reward_config({"reward": {"terms": [{"kind": "driveToLaunchPose"}]}}, None)


def test_unknown_kind_fails_schema():
    scoring = dict(load_preset("scoring", "biobuzz_2026_scoring_v1"))
    scoring.pop("_kind", None)
    scoring["reward"] = {"terms": [{"kind": "notATerm"}]}
    assert validate_document("scoring", scoring)

    training = dict(load_preset("training", "biobuzz_auto_lightweight"))
    training.pop("_kind", None)
    training["reward"] = {"terms": [{"kind": "notATerm"}]}
    assert validate_document("training", training)


def test_shipped_scoring_reward_validates():
    scoring = dict(load_preset("scoring", "biobuzz_2026_scoring_v1"))
    scoring.pop("_kind", None)
    assert validate_document("scoring", scoring) == []


def test_volume_ids_filter_by_tag_and_alliance():
    assert volume_ids_for_tag(PARK_ELEMENTS, "nest", "red") == ["home_pad"]
    assert volume_ids_for_tag(PARK_ELEMENTS, "nest", "blue") == ["away_pad"]


def test_mid_episode_reward_is_true_delta_only():
    tracker = RewardTracker(_bonus_cfg())
    reward, extra = tracker.step(
        true_delta=3.0,
        phase_end=False,
        accumulators={"nested": True, "endgame_points": 0},
        true_score=20.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={"home_pad": {"red_0"}},
        time_s=10.0,
        elements=PARK_ELEMENTS,
    )
    assert reward == 3.0
    assert extra == 0.0


def test_earlier_volume_entry_pays_larger_bonus():
    def bonus_at(t_enter: float) -> float:
        tracker = RewardTracker(_bonus_cfg())
        tracker.step(
            true_delta=0.0,
            phase_end=False,
            accumulators={},
            true_score=0.0,
            phase_duration=30.0,
            robot_id="red_0",
            alliance="red",
            occupancy={"home_pad": {"red_0"}},
            time_s=t_enter,
            elements=PARK_ELEMENTS,
        )
        _reward, extra = tracker.step(
            true_delta=20.0,
            phase_end=True,
            accumulators={"nested": True, "endgame_points": 5.0},
            true_score=25.0,
            phase_duration=30.0,
            robot_id="red_0",
            alliance="red",
            occupancy={"home_pad": {"red_0"}},
            time_s=30.0,
            elements=PARK_ELEMENTS,
        )
        return extra

    early = bonus_at(10.0)
    late = bonus_at(20.0)
    # gameplay = 25 - 5 = 20; scale 0.1 → 2.0 * (1 - t/30)
    assert early == pytest.approx(20.0 * 0.1 * (1 - 10 / 30))
    assert late == pytest.approx(20.0 * 0.1 * (1 - 20 / 30))
    assert early > late


def test_no_volume_entry_or_flag_means_zero_bonus():
    tracker = RewardTracker(_bonus_cfg())
    _reward, extra = tracker.step(
        true_delta=20.0,
        phase_end=True,
        accumulators={"nested": True, "endgame_points": 0},
        true_score=20.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={},
        time_s=30.0,
        elements=PARK_ELEMENTS,
    )
    assert extra == 0.0

    tracker.reset()
    tracker.step(
        true_delta=0.0,
        phase_end=False,
        accumulators={},
        true_score=0.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={"home_pad": {"red_0"}},
        time_s=8.0,
        elements=PARK_ELEMENTS,
    )
    _reward, extra = tracker.step(
        true_delta=20.0,
        phase_end=True,
        accumulators={"nested": False, "endgame_points": 0},
        true_score=20.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={"home_pad": {"red_0"}},
        time_s=30.0,
        elements=PARK_ELEMENTS,
    )
    assert extra == 0.0


def test_excluded_accumulators_are_not_multiplied():
    tracker = RewardTracker(_bonus_cfg())
    tracker.step(
        true_delta=0.0,
        phase_end=False,
        accumulators={},
        true_score=0.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={"home_pad": {"red_0"}},
        time_s=0.0,
        elements=PARK_ELEMENTS,
    )
    _reward, extra = tracker.step(
        true_delta=8.0,
        phase_end=True,
        accumulators={"nested": True, "endgame_points": 8.0},
        true_score=8.0,
        phase_duration=30.0,
        robot_id="red_0",
        alliance="red",
        occupancy={"home_pad": {"red_0"}},
        time_s=30.0,
        elements=PARK_ELEMENTS,
    )
    assert extra == 0.0
