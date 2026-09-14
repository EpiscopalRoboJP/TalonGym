from pathlib import Path

import numpy as np
import pytest

from talongym.presets.loader import load_preset
from talongym.training.curriculum import ballistic_launch, curriculum_unlocks, scripted_launch
from talongym.training.distill import collect_full_actions


def test_biobuzz_curriculum_has_launch_stages():
    training = load_preset("training", "biobuzz_auto_lightweight")
    assert training["algorithm"]["name"] == "bc_then_ppo"
    early = curriculum_unlocks(training, 0.0)
    mid = curriculum_unlocks(training, 0.5)
    late = curriculum_unlocks(training, 0.99)
    assert "scripted_launch" in early
    assert "motif_known_at_t0" not in early
    assert scripted_launch(training, 0.0) is True
    assert ballistic_launch(training, 0.5) is True
    assert "full_noise" in late


def test_collect_full_actions_includes_mechanism():
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    xs, ys = collect_full_actions(n_steps=8, bundle=bundle, seed=0)
    assert len(xs) >= 8
    assert ys.shape[1] == 5
    assert np.isfinite(ys).all()


def test_bc_warmup_reduces_or_runs(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO

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
    mse = bc_warmup(model, bundle, n_steps=32)
    venv.close()
    assert np.isfinite(mse)
