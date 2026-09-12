from talongym.env.ftc_auto import FTCAutoEnv


def test_unread_match_var_is_sentinel():
    env = FTCAutoEnv(record=False)
    obs, info = env.reset(seed=0)
    priv = info["privileged"]["matchVars"]
    assert priv, "expected randomized match variables"
    row = obs["match_var_obs"][0]
    assert row[-1] == 1.0
    assert row[:-1].sum() == 0.0
    env.world.match_vars = {k: "LEAK" for k in env.world.match_vars}
    obs2 = env._obs()
    assert obs2["match_var_obs"][0, -1] == 1.0
    env.close()


def test_curriculum_can_reveal_at_t0():
    env = FTCAutoEnv(motif_known_at_t0=True)
    obs, info = env.reset(seed=1)
    assert obs["match_var_obs"][0, -1] == 0.0
    assert obs["match_var_obs"][0, :-1].sum() == 1.0
    env.close()
