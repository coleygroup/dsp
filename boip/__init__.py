from .objectives import build_objective, discretize
from .optimize import optimize


try:
    from . import _version

    __version__ = _version.version
except ModuleNotFoundError:
    __version__ = "0.1.0"
