import pytest


def test_train_ppo_requires_rl_or_runs():
    from talongym.training.ppo import train_ppo

    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="RecurrentPPO"):
            train_ppo(total_steps=128, n_envs=1, allow_scripted=False)
        return
    seen: list[dict] = []
    result = train_ppo(
        total_steps=256,
        n_envs=1,
        allow_scripted=False,
        eval_episodes=0,
        on_metrics=lambda m: seen.append(dict(m)),
    )
    assert result["algo"] in {"recurrent_ppo", "bc_then_ppo"}
    assert result["steps"] >= 256
    assert result.get("checkpoint")
    assert seen, "RecurrentPPO must stream metrics during learn(), not only after the chunk"
    assert any(int(m.get("envSteps") or 0) < 256 for m in seen)
    assert any(float(m.get("progressFrac") or 0) > 0 for m in seen)
