from .objectives import build_objective, discretize
from .optimize import optimize

from . import _version

__version__ = _version.get_versions()["version"]
