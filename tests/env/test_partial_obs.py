from talongym.env.ftc_auto import FTCAutoEnv


def test_biobuzz_has_no_auto_match_vars():
    env = FTCAutoEnv(record=False)
    obs, info = env.reset(seed=0)
    priv = info["privileged"]["matchVars"]
    assert priv == {}
    row = obs["match_var_obs"][0]
    assert row[-1] == 1.0
    assert row[:-1].sum() == 0.0
    env.close()


def test_curriculum_reveal_is_a_no_op_without_match_vars():
    env = FTCAutoEnv(motif_known_at_t0=True)
    obs, _info = env.reset(seed=1)
    assert obs["match_var_obs"][0, -1] == 1.0
    assert obs["match_var_obs"][0, :-1].sum() == 0.0
    env.close()
