"""Optional CPU-clock diagnostics for the LASTFM training run."""

import atexit
import json
import os
from pathlib import Path

import numpy as np
from probe import SplitSampler


def install(directory):
    original = SplitSampler.sample
    values = []

    def sample(self, *args, **kwargs):
        result, stats = original(self, *args, **kwargs, profile=True)
        if kwargs["batch"] >= 76 and stats["gpu_queries"]:
            values.append((stats["host_submit_us"], stats["dispatch_us"]))
        return result, stats

    def finish():
        if values:
            path = Path(directory) / f"host-{os.environ['RANK']}.json"
            median = np.median(values, axis=0)
            path.write_text(
                json.dumps(
                    dict(
                        batches=len(values),
                        host_submit_us=median[0],
                        dispatch_us=median[1],
                    )
                )
                + "\n"
            )

    SplitSampler.sample = sample
    atexit.register(finish)
