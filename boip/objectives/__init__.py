from typing import Set
from botorch import test_functions
from botorch.test_functions import SyntheticTestFunction

from .cosines import Cosines
from .discretize import discretize


def build_objective(obj: str) -> SyntheticTestFunction:
    obj = obj.upper()

    if obj == "BEALE":
        return test_functions.Beale(negate=True)
    if obj == "BRANIN":
        return test_functions.Branin(negate=True)
    if obj == "BUKIN":
        return test_functions.Bukin(negate=True)
    if obj == "COSINE8":
        return test_functions.Cosine8(negate=True)
    if obj == "COSINES":
        return Cosines(negate=True)
    if obj == "DROP-WAVE":
        return test_functions.DropWave(negate=True)
    if obj == "MICHALEWICZ":
        return test_functions.Michalewicz(negate=True)
    if obj == "LEVY":
        return test_functions.Levy(negate=True)

    raise ValueError(f"Invalid objective. got: {obj}")


def valid_objectives() -> Set[str]:
    return {
        "BEALE",
        "BRANIN",
        "BUKIN",
        "COSINE8",
        "COSINES",
        "DROP-WAVE",
        "MICHALEWICZ",
        "LEVY",
    }
