import pytest


def test_train_ppo_requires_rl_or_runs():
    from talongym.training.ppo import train_ppo

    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="RecurrentPPO"):
            train_ppo(total_steps=128, n_envs=1, allow_scripted=False)
        return
    result = train_ppo(total_steps=256, n_envs=1, allow_scripted=False, eval_episodes=0)
    assert result["algo"] == "recurrent_ppo"
    assert result["steps"] >= 256
    assert result.get("checkpoint")
