#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dots_boxes_mcts.encoding import CHANNEL_NAMES, action_ids

METADATA_KEYS = (
    "board_height",
    "board_width",
    "channels",
    "action_count",
    "hidden_size",
    "residual_blocks",
)

ITERATION_RE = re.compile(r"iter(?P<iteration>\d+)")


@dataclass(frozen=True)
class ExportedBot:
    out_dir: Path
    manifest_path: Path
    weights_path: Path
    manifest: dict[str, Any]


def infer_iteration(path: Path) -> int | None:
    match = ITERATION_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("iteration"))


def default_output_dir(checkpoint: Path, public_dir: Path) -> Path:
    iteration = infer_iteration(checkpoint)
    if iteration is None:
        raise ValueError("Could not infer iteration from checkpoint name; pass --out.")
    return public_dir / "assets" / "bots" / f"iter{iteration:03d}"


def checkpoint_metadata(data: np.lib.npyio.NpzFile) -> dict[str, int]:
    missing = sorted(set(METADATA_KEYS) - set(data.files))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata fields: {', '.join(missing)}")
    return {key: int(data[key][0]) for key in METADATA_KEYS}


def export_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    set_latest: bool = False,
    public_dir: Path | None = None,
) -> ExportedBot:
    data = np.load(checkpoint)
    metadata = checkpoint_metadata(data)
    rows, cols = board_size_from_metadata(metadata)
    ids = action_ids(rows, cols)
    validate_metadata(metadata=metadata, action_count=len(ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.bin"
    manifest_path = out_dir / "manifest.json"

    offset = 0
    weights: dict[str, dict[str, Any]] = {}
    with weights_path.open("wb") as weights_file:
        for name in (key for key in data.files if key not in METADATA_KEYS):
            array = np.asarray(data[name], dtype="<f4", order="C")
            payload = array.tobytes(order="C")
            weights_file.write(payload)
            weights[name] = {
                "shape": list(array.shape),
                "dtype": "float32",
                "byteOffset": offset,
                "byteLength": len(payload),
            }
            offset += len(payload)

    created_at = datetime.now(timezone.utc).isoformat()
    iteration = infer_iteration(checkpoint)
    manifest: dict[str, Any] = {
        "version": 1,
        "format": "dots-boxes-ez-float32-v1",
        "createdAt": created_at,
        "sourceCheckpoint": str(checkpoint),
        "iteration": iteration,
        "board": {
            "rows": rows,
            "cols": cols,
            "height": metadata["board_height"],
            "width": metadata["board_width"],
        },
        "model": {
            "channels": metadata["channels"],
            "channelNames": list(CHANNEL_NAMES),
            "actionCount": metadata["action_count"],
            "actionIds": list(ids),
            "hiddenSize": metadata["hidden_size"],
            "residualBlocks": metadata["residual_blocks"],
            "batchNormEpsilon": 1.0e-5,
        },
        "weights": weights,
        "weightsFile": "weights.bin",
        "weightsByteLength": offset,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")

    if set_latest:
        latest_dir = public_dir / "assets" / "bots" if public_dir is not None else out_dir.parent
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_payload = {
            "version": 1,
            "iteration": iteration,
            "manifest": f"{out_dir.name}/manifest.json",
            "createdAt": created_at,
        }
        (latest_dir / "latest.json").write_text(
            json.dumps(latest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )

    return ExportedBot(
        out_dir=out_dir,
        manifest_path=manifest_path,
        weights_path=weights_path,
        manifest=manifest,
    )


def board_size_from_metadata(metadata: dict[str, int]) -> tuple[int, int]:
    height = metadata["board_height"]
    width = metadata["board_width"]
    if height % 2 != 1 or width % 2 != 1:
        raise ValueError(f"Expected odd encoded board dimensions, got {height}x{width}.")
    return (height + 1) // 2, (width + 1) // 2


def validate_metadata(*, metadata: dict[str, int], action_count: int) -> None:
    if metadata["channels"] != len(CHANNEL_NAMES):
        raise ValueError(
            f"Checkpoint channel count {metadata['channels']} does not match "
            f"{len(CHANNEL_NAMES)} encoder channels."
        )
    if metadata["action_count"] != action_count:
        raise ValueError(
            f"Checkpoint action count {metadata['action_count']} does not match "
            f"{action_count} generated board actions."
        )
    if metadata["hidden_size"] <= 0:
        raise ValueError("Checkpoint hidden_size must be positive.")
    if metadata["residual_blocks"] <= 0:
        raise ValueError("Checkpoint residual_blocks must be positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an EpsilonZero MLX .npz checkpoint for the static browser bot."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=Path("static_bot/public"),
        help="Static public directory used for the default --out and latest.json.",
    )
    parser.add_argument(
        "--set-latest",
        action="store_true",
        help="Update assets/bots/latest.json to point at this exported bot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or default_output_dir(args.checkpoint, args.public_dir)
    exported = export_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=out_dir,
        set_latest=args.set_latest,
        public_dir=args.public_dir,
    )
    print(
        json.dumps(
            {
                "manifest": str(exported.manifest_path),
                "weights": str(exported.weights_path),
                "weightsByteLength": exported.manifest["weightsByteLength"],
                "iteration": exported.manifest["iteration"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
