from talongym.eval.harness import run_paired_trials
from talongym.training.policies import scripted_auto


def idle_policy(obs, info=None):
    pose = obs["pose_noisy"]
    return {
        "target_pose": pose,
        "speed_frac": [0.2],
        "mechanism": 0,
    }


def test_crn_same_seeds_and_metrics():
    report = run_paired_trials(scripted_auto, idle_policy, n_trials=4, seed0=42, pair_common_random_numbers=True)
    assert report["paired"] is True
    assert report["a"]["seeds"] == report["b"]["seeds"]
    assert "collisionTimeMean" in report["a"]
    assert "restrictedEntryRate" in report["a"]
    assert report["bestLabelEligible"] is False
