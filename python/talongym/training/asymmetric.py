"""Asymmetric MultiInput LSTM: actor encoder-only, critic sees privileged state."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from talongym.training.privileged import PRIV_KEY

try:
    from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:  # pragma: no cover
    BaseFeaturesExtractor = object  # type: ignore
    RecurrentMultiInputActorCriticPolicy = object  # type: ignore


OBS_SCALE_CAP = 100.0


def _sorted_keys(observation_space: spaces.Dict, include_privileged: bool) -> list[str]:
    keys = sorted(observation_space.spaces.keys())
    if include_privileged:
        return keys
    return [k for k in keys if k != PRIV_KEY]


class SplitDictExtractor(BaseFeaturesExtractor):  # type: ignore[misc]
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, include_privileged: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self._keys = _sorted_keys(observation_space, include_privileged)
        n = 0
        # Raw inputs span inches (up to +/-200) and 0/1 flags; unscaled, they saturate the LSTM
        # and the actor ignores its observations. Divide each key by its bound, capped at OBS_SCALE_CAP.
        self._scales: dict[str, float] = {}
        for key in self._keys:
            box = cast(spaces.Box, observation_space.spaces[key])
            n += int(np.prod(cast(tuple[int, ...], box.shape)))
            bound = float(np.max(np.abs(np.concatenate([np.ravel(box.low), np.ravel(box.high)]))))
            self._scales[key] = min(max(bound, 1.0), OBS_SCALE_CAP) if np.isfinite(bound) else OBS_SCALE_CAP
        self.linear = nn.Sequential(nn.Linear(max(1, n), features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self._keys:
            if key not in observations:
                continue
            parts.append(observations[key].flatten(start_dim=1) / self._scales[key])
        if not parts:
            batch = next(iter(observations.values())).shape[0]
            x = torch.zeros(batch, 1, device=next(iter(observations.values())).device)
        else:
            x = torch.cat(parts, dim=1)
        return self.linear(x)


class ActorDictExtractor(SplitDictExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim, include_privileged=False)


class CriticDictExtractor(SplitDictExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim, include_privileged=True)


class AsymmetricLstmPolicy(RecurrentMultiInputActorCriticPolicy):  # type: ignore[misc]
    """RecurrentPPO policy whose actor cannot read `_privileged`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", ActorDictExtractor)
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)
        dim = int(getattr(self, "features_dim", 256) or 256)
        self.vf_features_extractor = CriticDictExtractor(cast(spaces.Dict, self.observation_space), features_dim=dim)
