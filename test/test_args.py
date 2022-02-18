from random import random
import shlex
import pytest

import numpy as np

from boip.cli.args import parse_args


@pytest.fixture(params=[1, 0.5, random(), -10, 5e6, 42])
def k_or_threshold(request):
    argv = shlex.split(f"--k-or-threshold {request.param}")

    return argv, request.param


@pytest.fixture(params=[-1, 0, 10000])
def N(request):
    argv = shlex.split(f"-N {request.param}")

    return argv, request.param


def test_k_or_threshold(k_or_threshold):
    argv, k_or_threshold = k_or_threshold

    args = parse_args(argv)

    np.testing.assert_almost_equal(args.k_or_threshold, k_or_threshold)


def test_k_or_threshold_None(N):
    argv, N = N

    args = parse_args(argv)

    assert args.k_or_threshold == N
