import pytest

from talongym.training.rllib import train_rllib


def test_rllib_requires_extra():
    try:
        import ray  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="scale"):
            train_rllib(total_steps=8)
    else:
        pytest.skip("ray installed; skip missing-extra assertion")
