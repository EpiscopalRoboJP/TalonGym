from pathlib import Path

import numpy as np

from talongym.training.distill import distill_mlp


def test_distill_writes_artifact(tmp_path: Path):
    x = np.random.randn(32, 12).astype(np.float32)
    y = np.random.randn(32, 3).astype(np.float32)
    path = distill_mlp(x, y, tmp_path / "ff.onnx")
    assert path.exists()
