import pytest

from talongym.env.ftc_auto import EncoderOnlyObsAssertWrapper, FTCAutoEnv
from talongym.presets.loader import load_bundle
from talongym.training.privileged import PRIV_KEY, PrivilegedObsWrapper


def test_privileged_absent_from_actor_obs():
    env = FTCAutoEnv(record=False)
    obs, info = env.reset(seed=0)
    assert PRIV_KEY not in obs
    assert "privileged" in info
    assert "pose" in info["privileged"]
    env.close()


def test_privileged_wrapper_not_visible_to_encoder_assert():
    bundle = load_bundle()
    inner = EncoderOnlyObsAssertWrapper(FTCAutoEnv(bundle=bundle, record=False, motif_known_at_t0=False))
    wrapped = PrivilegedObsWrapper(inner)
    obs, info = wrapped.reset(seed=0)
    assert PRIV_KEY in obs
    actor = wrapped.unwrapped._obs()
    assert PRIV_KEY not in actor
    wrapped.close()


def test_actor_extractor_drops_privileged_key():
    pytest.importorskip("torch")
    from gymnasium import spaces
    import numpy as np

    from talongym.training.asymmetric import ActorDictExtractor, CriticDictExtractor
    from talongym.training.privileged import PRIV_DIM, PRIV_KEY

    space = spaces.Dict(
        {
            "pose_noisy": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            PRIV_KEY: spaces.Box(-1, 1, shape=(PRIV_DIM,), dtype=np.float32),
        }
    )
    actor = ActorDictExtractor(space, features_dim=8)
    critic = CriticDictExtractor(space, features_dim=8)
    assert PRIV_KEY not in actor._keys
    assert PRIV_KEY in critic._keys
