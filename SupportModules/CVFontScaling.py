from functools import lru_cache
from typing import Optional, Callable

def _make_text_fn(scale: float) -> Callable[[Optional[float]], float]:
    @lru_cache(maxsize=1)
    def _core(v: float | int) -> float:
        return v * scale

    def api(v: float | int | None = None) -> float:
        # v is provided: compute (and cache) + remember last result
        if v is not None:
            res = _core(v)
            api._last = res  # type: ignore[attr-defined]
            return res
        # v is None: reuse last result without a cache lookup
        try:
            return api._last  # type: ignore[attr-defined]
        except AttributeError as e:
            raise ValueError(
                "No cached value yet—call the function once with a number first."
            ) from e

    return api

# Precomputed scales (single multiply per call)
_SMALL = 0.25 / 640.0
_MED   = 0.50 / 640.0
_LRG   = 0.75 / 640.0

small_text = _make_text_fn(_SMALL)
med_text   = _make_text_fn(_MED)
lrg_text   = _make_text_fn(_LRG)

# Precompute for default image of 864x864
small_text(864)
med_text(864)
lrg_text(864)
