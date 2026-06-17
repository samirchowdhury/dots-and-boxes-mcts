from __future__ import annotations

import sys

from setuptools import Extension, find_packages, setup


compile_args = ["-std=c++17", "-O3"]
if sys.platform != "win32":
    compile_args.extend(["-Wall", "-Wextra"])


setup(
    packages=find_packages(),
    ext_modules=[
        Extension(
            "dots_boxes_mcts._fast_ez_mcts_cpp",
            sources=["dots_boxes_mcts/_cpp/fast_ez_mcts.cpp"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
)
