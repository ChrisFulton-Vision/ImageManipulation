from support.io.my_logging import LOG

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    prange = range
    _HAVE_NUMBA = False
    LOG.warning("Use of numba is recommend for blazing fast performance, but user does not have it!\n")
    LOG.warning("pip install numba\n")
try:
    _CAN_CACHE = ("__file__" in globals() and globals()["__file__"] is not None)
except Exception:
    _CAN_CACHE = False

def _njit(*dargs, **dkwargs):
    """
    Usage:
        @njit
        def f(...): ...

        @njit(cache=True, fastmath=False)
        def g(...): ...
    """
    if not _HAVE_NUMBA:
        # No numba: decorator is identity
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]  # used as @_njit
        def decorator(func):
            return func       # used as @_njit(...)
        return decorator

    # If cache requested but we're not in a file-backed module, disable it.
    if dkwargs.get("cache", False) and not _CAN_CACHE:
        dkwargs = dict(dkwargs)
        dkwargs["cache"] = False

    # Have numba: forward to numba.njit
    return njit(*dargs, **dkwargs)