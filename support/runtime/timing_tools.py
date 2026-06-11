# timing_tools.py

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def timed(repeats: int = 1) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that times a function.

    If repeats > 1, the function is called repeatedly and the final
    result is returned.

    Example:
        @timed(repeats=100)
        def my_function():
            ...
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                for _ in range(repeats - 1):
                    result = func(*args, **kwargs)

                return result

            finally:
                elapsed_s = time.perf_counter() - start_time
                elapsed_ms = elapsed_s * 1000.0
                avg_ms = elapsed_ms / repeats

                print(
                    f"{func.__module__}.{func.__qualname__}: "
                    f"{elapsed_ms:.3f} ms total, "
                    f"{avg_ms:.3f} ms avg over {repeats} call(s)"
                )

        return wrapper

    return decorator