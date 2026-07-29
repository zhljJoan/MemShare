import torch

import importlib.util
import logging
import os
import sys

__version__ = "0.1.0"

try:
    from .lib import libstarrygl as ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl.so, some features may not be available.")

try:
    _sampler_library = os.environ.get("STARRYGL_SAMPLER_LIBRARY")
    if _sampler_library:
        _sampler_module_name = "starrygl.lib.libstarrygl_sampler"
        _sampler_spec = importlib.util.spec_from_file_location(
            _sampler_module_name, _sampler_library
        )
        if _sampler_spec is None or _sampler_spec.loader is None:
            raise ImportError(
                "unable to create a module spec for {}".format(_sampler_library)
            )
        sampler_ops = importlib.util.module_from_spec(_sampler_spec)
        sys.modules[_sampler_module_name] = sampler_ops
        _sampler_spec.loader.exec_module(sampler_ops)
    else:
        from .lib import libstarrygl_sampler as sampler_ops
except Exception as e:
    logging.error(e)
    logging.error("unable to import libstarrygl_sampler.so, some features may not be available.")
