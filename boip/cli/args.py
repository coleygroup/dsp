from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("-o", "--objective")
    parser.add_argument(
        "-c",
        "--num-choices",
        type=int,
        default=100000,
        help="the number of points with which to discretize the objective function",
    )
    parser.add_argument("-N", type=int, default=100, help="the number of initialization points")
    parser.add_argument("-q", "--batch-size", type=int, default=int)
    parser.add_argument(
        "-T", type=int, default=100, help="the number iterations to perform optimization"
    )
    parser.add_argument(
        "-R",
        "--repeats",
        type=int,
        help="the number of repetitions to perform. If not specified, then collate.py must be run afterwards for further analysis.",
    )
    parser.add_argument(
        "-ds",
        "--discretization-seed",
        type=int,
        help="the random seed to use for discrete landscapes",
    )
    parser.add_argument("-p", "--prob", type=float)
    parser.add_argument("-a", "--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", help="the directory under which to save the outputs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)

    return parser.parse_args(argv)
