from talongym.env.ftc_auto import EncoderOnlyObsAssertWrapper, FTCAutoEnv


def test_motif_not_leaked_when_tag_not_visible():
    env = EncoderOnlyObsAssertWrapper(FTCAutoEnv(record=False, motif_known_at_t0=False))
    obs, info = env.reset(seed=0)
    assert obs["match_var_obs"][0, -1] == 1.0
    priv = info["privileged"]["matchVars"]
    env.env.world.match_vars = {k: "LEAK" for k in priv}
    obs2 = env.env._obs()
    assert obs2["match_var_obs"][0, -1] == 1.0
    env.close()
