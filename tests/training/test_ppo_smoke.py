import pytest


def _stub_bc_warmup(model, bundle, n_steps, log=lambda m: None, epochs=2500):
    log("BC warmup stub")
    return 0.0


def test_train_ppo_requires_rl_or_runs(monkeypatch):
    monkeypatch.setenv("TALONGYM_ALLOW_MISSING_MESH", "1")
    import talongym.training.distill as distill
    from talongym.training.ppo import train_ppo

    monkeypatch.setattr(distill, "bc_warmup", _stub_bc_warmup)
    monkeypatch.setattr(distill, "BC_TARGET_LOSS", float("inf"))

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
    assert any(m.get("startupPhase") == "creating_envs" for m in seen)
    assert any(int(m.get("envSteps") or 0) < 256 for m in seen)
    assert any(float(m.get("progressFrac") or 0) > 0 for m in seen)


def test_eval_drop_rolls_back_to_best_and_halves_learning_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("TALONGYM_ALLOW_MISSING_MESH", "1")
    pytest.importorskip("sb3_contrib")
    import sb3_contrib

    import talongym.training.diagnostics as diagnostics
    import talongym.training.distill as distill
    import talongym.training.ppo as ppo

    monkeypatch.setattr(distill, "bc_warmup", _stub_bc_warmup)
    monkeypatch.setattr(distill, "BC_TARGET_LOSS", float("inf"))
    monkeypatch.setattr(
        diagnostics,
        "assert_scripted_baseline_scores",
        lambda bundle=None, seed=1: diagnostics.EpisodeHealth(true_score=8, launches=4, scored_pieces=1),
    )

    def fake_learn(self, total_timesteps=0, callback=None, reset_num_timesteps=True, **kwargs):
        if reset_num_timesteps:
            self.num_timesteps = 0
        self.num_timesteps = int(getattr(self, "num_timesteps", 0) or 0) + int(total_timesteps)
        if not hasattr(self, "_logger"):
            from types import SimpleNamespace

            self._logger = SimpleNamespace(name_to_value={})
        if callback is not None:
            callback.model = self
            callback.num_timesteps = self.num_timesteps
            callback._on_training_start()
            callback._on_rollout_end()
        return self

    monkeypatch.setattr(sb3_contrib.RecurrentPPO, "learn", fake_learn)
    monkeypatch.setattr(ppo, "record_policy_episode", lambda *args, **kwargs: [])
    evals = iter([[28.0, 28.0], [3.0, 3.0]])
    seen_options = []
    healthy_frames = [
        {
            "trueScore": 28,
            "launchAttempts": 4,
            "t": 1.0,
            "robots": [{"id": "red_0", "lastVerb": "score", "held": [], "collisionTimeS": 0.2}],
            "pieces": [{"id": "p1", "launchedBy": "red_0", "scored": True}],
            "collision": {"wall": False, "collisionTimeS": 0.2},
        }
    ]

    def fake_eval(adapter, bundle, seeds, record_first=False, match_setup=None, options=None):
        seen_options.append(options)
        scores = next(evals, [28.0, 28.0])
        return scores, healthy_frames

    monkeypatch.setattr(ppo, "_eval_true_scores", fake_eval)
    metrics: list[dict] = []
    logs: list[str] = []
    result = ppo.train_ppo(
        total_steps=16384,
        n_envs=1,
        eval_episodes=2,
        save_dir=tmp_path,
        on_metrics=metrics.append,
        log=logs.append,
    )
    assert metrics[-1]["anchorRollbacks"] == 1
    lr = float(result["model"].lr_schedule(1.0))
    assert lr < 3e-4 and abs(metrics[-1]["learningRate"] - lr) < 1e-12
    assert any("rolled back to best" in m for m in logs)
    assert seen_options[0]["full_noise"] is True
