from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from static_bot.tools.export_checkpoint import export_checkpoint


def write_checkpoint(path: Path, *, action_count: int = 12, channels: int = 8) -> None:
    np.savez(
        path,
        board_height=np.array([5]),
        board_width=np.array([5]),
        channels=np.array([channels]),
        action_count=np.array([action_count]),
        hidden_size=np.array([4]),
        residual_blocks=np.array([1]),
        **{
            "stem_conv.weight": np.arange(4 * 3 * 3 * channels, dtype=np.float32).reshape(
                4, 3, 3, channels
            ),
            "stem_bn.weight": np.ones(4, dtype=np.float32),
            "stem_bn.bias": np.zeros(4, dtype=np.float32),
            "stem_bn.running_mean": np.zeros(4, dtype=np.float32),
            "stem_bn.running_var": np.ones(4, dtype=np.float32),
        },
    )


def test_export_checkpoint_writes_manifest_and_float32_blob(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ez-policy-value-3x3-iter007-sims25.npz"
    out_dir = tmp_path / "public" / "assets" / "bots" / "iter007"
    write_checkpoint(checkpoint)

    exported = export_checkpoint(checkpoint, out_dir, set_latest=True)

    manifest = json.loads(exported.manifest_path.read_text(encoding="utf8"))
    latest = json.loads((out_dir.parent / "latest.json").read_text(encoding="utf8"))

    assert manifest["iteration"] == 7
    assert manifest["board"] == {"rows": 3, "cols": 3, "height": 5, "width": 5}
    assert manifest["model"]["actionCount"] == 12
    assert manifest["model"]["actionIds"][0] == "h:0:0"
    assert manifest["weights"]["stem_conv.weight"]["byteOffset"] == 0
    assert exported.weights_path.stat().st_size == manifest["weightsByteLength"]
    assert latest["manifest"] == "iter007/manifest.json"


def test_export_checkpoint_rejects_incompatible_action_count(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.npz"
    write_checkpoint(checkpoint, action_count=13)

    with pytest.raises(ValueError, match="action count"):
        export_checkpoint(checkpoint, tmp_path / "out")
