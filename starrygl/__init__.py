import torch

import logging

try:
    from .lib import libstarrygl_ops as ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl.so, some features may not be available.")

try:
    from .lib import libstarrygl_ops_sampler as sampler_ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl_sampler.so, some features may not be available.")