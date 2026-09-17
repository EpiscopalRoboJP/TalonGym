from pathlib import Path

import numpy as np
import pytest

from talongym.presets.loader import load_preset
from talongym.training.curriculum import curriculum_unlocks
from talongym.training.distill import collect_episodes, collect_full_actions


def test_biobuzz_curriculum_has_launch_stages():
    training = load_preset("training", "biobuzz_auto_lightweight")
    assert training["algorithm"]["name"] == "bc_then_ppo"
    early = curriculum_unlocks(training, 0.0)
    mid = curriculum_unlocks(training, 0.5)
    late = curriculum_unlocks(training, 0.99)
    assert "scripted_launch" not in early
    assert "ballistic_launch" not in early
    assert "motif_known_at_t0" not in early
    assert "scripted_launch" not in mid
    assert "ballistic_launch" not in mid
    assert "full_noise" in late


@pytest.mark.require_mesh
def test_collect_full_actions_includes_mechanism():
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    xs, ys = collect_full_actions(n_steps=8, bundle=bundle, seed=0)
    assert len(xs) >= 8
    assert ys.shape[1] == 5
    assert np.isfinite(ys).all()


@pytest.mark.require_mesh
def test_bc_pairs_reproduce_score_verb_from_launch_pose():
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "gobilda_mecanum_starter", "biobuzz_2026_scoring_v1")
    _obs, acts, _returns, _length = collect_episodes(
        1,
        bundle=bundle,
        seed=1,
        perturb=False,
        options={"full_noise": False, "curriculum_spawn": "launch", "static_teammate": False},
    )
    assert acts.shape[1] == 5
    assert (acts[:, 4] >= 1.5).any(), "scripted AUTO must fire (mechanism 2) from the launch pose"


@pytest.mark.require_mesh
def test_bc_pairs_reproduce_score_verb_from_legal_spawn():
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "gobilda_mecanum_starter", "biobuzz_2026_scoring_v1")
    _obs, acts, _returns, _length = collect_episodes(
        1,
        bundle=bundle,
        seed=1,
        perturb=False,
        options={"full_noise": False, "curriculum_spawn": "legal", "static_teammate": False},
    )
    assert acts.shape[1] == 5
    assert (acts[:, 4] >= 1.5).any(), "scripted AUTO must fire after driving from a legal spawn"


@pytest.mark.require_mesh
def test_collect_episodes_returns_whole_aligned_episodes():
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    obs, acts, returns, length = collect_episodes(2, bundle=bundle, seed=0)
    assert len(obs) == acts.shape[0] == returns.shape[0] == 2 * length
    assert acts.shape[1] == 5
    assert np.isfinite(acts).all() and np.isfinite(returns).all()
    assert "_privileged" in obs[0]


@pytest.mark.require_mesh
def test_bc_warmup_reduces_or_runs(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from talongym.env.ftc_auto import BoxActionDictObsEnv, EncoderOnlyObsAssertWrapper, FTCAutoEnv
    from talongym.presets.loader import load_bundle
    from talongym.training.asymmetric import AsymmetricLstmPolicy
    from talongym.training.distill import bc_warmup
    from talongym.training.privileged import PrivilegedObsWrapper

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")

    def make():
        env = FTCAutoEnv(bundle=bundle, record=False)
        return BoxActionDictObsEnv(PrivilegedObsWrapper(EncoderOnlyObsAssertWrapper(env)))

    venv = DummyVecEnv([make])
    model = RecurrentPPO(AsymmetricLstmPolicy, venv, verbose=0, n_steps=16, batch_size=16, n_epochs=1)
    mse = bc_warmup(model, bundle, n_steps=32, epochs=5)
    venv.close()
    assert np.isfinite(mse)
