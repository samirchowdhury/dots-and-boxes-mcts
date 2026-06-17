from __future__ import annotations

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def build_extension() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "dots_boxes_mcts" / "_cpp" / "fast_ez_mcts.cpp"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not report an extension suffix.")
    include_dir = sysconfig.get_paths()["include"]
    output = repo_root / "dots_boxes_mcts" / f"_fast_ez_mcts_cpp{suffix}"
    cxx = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")[0]

    command = [
        cxx,
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-shared",
        "-fPIC",
    ]
    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    command.extend([f"-I{include_dir}", str(source), "-o", str(output)])
    subprocess.run(command, check=True)
    return output


def main() -> None:
    output = build_extension()
    print(output)


if __name__ == "__main__":
    main()
