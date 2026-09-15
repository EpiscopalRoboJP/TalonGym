import pytest


def test_train_ppo_requires_rl_or_runs(monkeypatch):
    import talongym.training.distill as distill
    from talongym.training.ppo import train_ppo

    # Plumbing only: stop BC after its first epoch instead of training to convergence.
    monkeypatch.setattr(distill, "BC_TARGET_LOSS", float("inf"))

    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="RecurrentPPO"):
            train_ppo(total_steps=128, n_envs=1, allow_scripted=False)
        return
    result = train_ppo(total_steps=256, n_envs=1, allow_scripted=False, eval_episodes=0)
    assert result["algo"] in {"recurrent_ppo", "bc_then_ppo"}
    assert result["steps"] >= 256
    assert result.get("checkpoint")


def test_eval_drop_rolls_back_to_best_and_halves_learning_rate(monkeypatch, tmp_path):
    pytest.importorskip("sb3_contrib")
    import talongym.training.distill as distill
    import talongym.training.ppo as ppo

    monkeypatch.setattr(distill, "BC_TARGET_LOSS", float("inf"))
    # First call scores the clone, the second the policy after one PPO chunk.
    evals = iter([[28.0, 28.0], [3.0, 3.0]])
    seen_options = []

    def fake_eval(adapter, bundle, seeds, record_first=False, options=None):
        seen_options.append(options)
        return next(evals, [28.0, 28.0]), []

    monkeypatch.setattr(ppo, "_eval_true_scores", fake_eval)
    metrics: list[dict] = []
    logs: list[str] = []
    result = ppo.train_ppo(
        total_steps=256, n_envs=1, eval_episodes=2, save_dir=tmp_path, on_metrics=metrics.append, log=logs.append
    )
    assert metrics[-1]["anchorRollbacks"] == 1
    lr = float(result["model"].lr_schedule(1.0))
    assert lr < 3e-4 and abs(metrics[-1]["learningRate"] - lr) < 1e-12
    assert any("rolled back to best" in m for m in logs)
    assert seen_options[0]["ballistic_launch"] is True and seen_options[0]["full_noise"] is True
