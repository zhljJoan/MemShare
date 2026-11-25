"""Launch the existing training entry point with an isolated sampler adapter."""

import os
import runpy
import sys

mode = sys.argv.pop(1)
if mode in ("hybrid", "fixed", "graph"):
    from train_adapter import install

    if directory := os.getenv("MEMSHARE_PROFILE_SUBMISSION"):
        from profile_submission import install as profile_submission

        profile_submission(directory)

    install(fixed=mode != "hybrid", replay=mode == "graph")
elif mode != "native":
    raise ValueError(mode)
sys.path.insert(0, ".")
runpy.run_path("train_boundery.py", run_name="__main__")
