WINDOW_THRESHOLD = 4


def window_size(t: int) -> int:
    """calculate the window size given the number of iterations without any changes in input space
    size"""
    if t < 0:
        raise ValueError(f"t must be greater than 0! got {t}")

    if t <= WINDOW_THRESHOLD:
        return 2**t
    else:
        return 2**WINDOW_THRESHOLD + (t - WINDOW_THRESHOLD)
