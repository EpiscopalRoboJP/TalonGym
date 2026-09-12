from talongym.training.policies import scripted_auto

__all__ = ["scripted_auto", "record_policy_episode", "train_ppo"]


def __getattr__(name: str):
    if name in {"record_policy_episode", "train_ppo", "load_trained_policy", "RecurrentPolicyAdapter"}:
        from talongym.training import ppo

        return getattr(ppo, name)
    raise AttributeError(name)
