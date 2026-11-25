"""Launch the existing training entry point with an isolated sampler adapter."""

import runpy
import sys

mode = sys.argv.pop(1)
if mode == "hybrid":
    from train_adapter import install

    install()
elif mode != "native":
    raise ValueError(mode)
sys.path.insert(0, ".")
runpy.run_path("train_boundery.py", run_name="__main__")
