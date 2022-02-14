from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence
from boip import objectives

from boip.initialize import InitMode


def int_or_float(arg: str):
    try:
        value = int(arg)
    except ValueError:
        value = float(arg)

    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--objective",
        choices=objectives.valid_objectives(),
        help="the test function to use (case insensitive)",
    )
    parser.add_argument(
        "-c",
        "--num-choices",
        type=int,
        default=10000,
        help="the number of points with which to discretize the objective function",
    )
    parser.add_argument("-N", type=int, default=10, help="the number of initialization points")
    parser.add_argument("-q", "--batch-size", type=int, default=10)
    parser.add_argument(
        "-T",
        type=int,
        default=100,
        help="the number iterations to perform optimization",
    )
    parser.add_argument(
        "-R",
        "--repeats",
        type=int,
        help="the number of repetitions to perform. If not specified, then collate.py must be run afterwards for further analysis",
    )
    parser.add_argument(
        "-ds",
        "--discretization-seed",
        type=int,
        default=42,
        help="the random seed to use for discrete landscapes",
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        help="the minimum hit probability needed to retain a given point during pruning",
    )
    parser.add_argument(
        "--k-or-threshold",
        type=int_or_float,
        help="the rank of the predictions (int) or absolute threshold (float) to use when determing what constitutes a predicted hit",
    )
    parser.add_argument(
        "--use-observed-threshold",
        action="store_true",
        help="if using rank-based hit thresholding, calculate the threshold from the k-th best observation, rather than the k-th best predicted mean",
    )
    parser.add_argument("-g", "--gamma", type=float, default=1.0)
    parser.add_argument("--output-dir", help="the directory under which to save the outputs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--init-mode", type=InitMode.from_str, default=InitMode.UNIFORM)
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args(argv)

    args.k_or_threshold = args.k_or_threshold or args.N

    return args
