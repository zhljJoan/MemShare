import torch

import logging

__version__ = "0.1.0"

try:
    from .lib import libstarrygl as ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl.so, some features may not be available.")

try:
    from .lib import libstarrygl_sampler as sampler_ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl_sampler.so, some features may not be available.")
    