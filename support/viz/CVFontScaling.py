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

def _compute_thickness_triplet(v: float | int) -> tuple[int, int, int]:
    """
    Compute (small, med, large) thicknesses with guaranteed separation:
      med >= small + 1
      large >= med + 1
    """
    small = max(1, int(round(v * _SMALL_THICK)))
    med   = max(small + 1, int(round(v * _MED_THICK)))
    large = max(med + 1,   int(round(v * _LRG_THICK)))
    return small, med, large


def _make_thickness_fn(index: int) -> Callable[[Optional[float]], int]:
    @lru_cache(maxsize=1)
    def _core(v: float | int) -> int:
        return _compute_thickness_triplet(v)[index]

    def api(v: float | int | None = None) -> int:
        if v is not None:
            s, m, l = _compute_thickness_triplet(v)
            small_thick._last = s  # type: ignore[attr-defined]
            med_thick._last   = m  # type: ignore[attr-defined]
            lrg_thick._last   = l  # type: ignore[attr-defined]
            return (s, m, l)[index]
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

# Thickness scales
_SMALL_THICK = 1.0 / 640.0
_MED_THICK   = 1.5 / 640.0
_LRG_THICK   = 2.0 / 640.0

small_thick = _make_thickness_fn(0)
med_thick   = _make_thickness_fn(1)
lrg_thick   = _make_thickness_fn(2)

# Precompute for default image of 864x864
small_text(864)
med_text(864)
lrg_text(864)

small_thick(864)
med_thick(864)
lrg_thick(864)