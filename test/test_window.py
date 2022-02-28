import pytest

from dsp.window import window_size, WINDOW_THRESHOLD


@pytest.mark.parametrize("t", [-2, -1])
def test_window_neg(t):
    with pytest.raises(ValueError):
        window_size(t)


@pytest.mark.parametrize("t", range(WINDOW_THRESHOLD))
def test_window_below_threshold(t):
    q = window_size(t)
    assert q == 2**t


@pytest.mark.parametrize("t", range(WINDOW_THRESHOLD, 2 * WINDOW_THRESHOLD))
def test_window_above_threshold(t):
    q = window_size(t)
    assert q == 2 ** (WINDOW_THRESHOLD) + (t - WINDOW_THRESHOLD)
