# support/io/configStore.py

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from yaml import safe_load, dump


# ----------------------------- small protocol -----------------------------

class AfterScheduler(Protocol):
    """Tk-like scheduler protocol."""
    def after(self, delay_ms: int, callback: Callable[[], None]) -> Any: ...
    def after_cancel(self, handle: Any) -> None: ...


# ----------------------------- config store ------------------------------

@dataclass
class LoadResult:
    yaml_path: Optional[str]
    loaded_yaml: bool


class ConfigStore:
    """
    Stores:
      1) A pointer file (pickle) -> path to latest YAML config
      2) A YAML config file -> serialized CameraConfig

    Designed to be GUI-agnostic:
      - If you pass a Tk-like scheduler, it can debounce writes (saveToCache).
      - If no scheduler is available, it will always write immediately.
    """

    def __init__(
        self,
        cache_pointer_path: str | Path,
        configs_dir: str | Path = "Configs",
        scheduler: Optional[AfterScheduler] = None,
    ) -> None:
        self.cache_pointer_path = Path(cache_pointer_path)
        self.configs_dir = Path(configs_dir)
        self.scheduler = scheduler
        self._save_debounce_id: Any | None = None

    # ----------------------------- loading -----------------------------

    def load_from_cache(self, cam_config: Any) -> LoadResult:
        """
        Equivalent of CameraGui.loadFromCache(), minus UI side effects.

        - Reads cache pointer pickle (YAML path) into cam_config.configFilepath
        - If YAML exists, loads YAML and calls cam_config.fromDict(data)
        - Returns LoadResult so caller can update UI + run update_post_newCamConfig()
        """
        yaml_path: Optional[str] = None
        loaded_yaml = False

        cache_path = self.cache_pointer_path
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    yaml_path = pickle.load(f)
                    if isinstance(yaml_path, Path):
                        yaml_path = str(yaml_path)
                    cam_config.configFilepath = yaml_path
            except Exception:
                # Corrupt cache pointer -> ignore
                yaml_path = None

        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    data = safe_load(f) or {}
                # Expected on your CameraConfig. :contentReference[oaicite:3]{index=3}
                cam_config.fromDict(data)
                loaded_yaml = True
            except Exception:
                loaded_yaml = False

        return LoadResult(yaml_path=yaml_path, loaded_yaml=loaded_yaml)

    # ----------------------------- flushing -----------------------------

    def flush_now(self, cam_config: Any) -> None:
        """
        Equivalent of _flush_cache_now().

        Writes:
          - cache_pointer_path: pickle of cam_config.configFilepath
          - cam_config.configFilepath YAML: dump(cam_config.toDict, f) (supports property or method)

        Matches original behavior. :contentReference[oaicite:4]{index=4}
        """
        cache_path = self.cache_pointer_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # pointer to most-recent config YAML
        with cache_path.open("wb") as f:
            pickle.dump(cam_config.configFilepath, f)

        # ensure Configs exists (original did this) :contentReference[oaicite:5]{index=5}
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # write YAML
        yaml_path = Path(cam_config.configFilepath)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        cfg_dict = cam_config.toDict
        if cfg_dict is None:
            raise AttributeError("cam_config has no toDict (property or method)")

        with open(yaml_path, "w") as f:
            dump(cfg_dict, f)

    # ----------------------------- saving (debounced) -----------------------------

    def save_to_cache(self, cam_config: Any, immediate: bool = False, delay_ms: int = 500) -> None:
        """
        Equivalent of saveToCache().

        - immediate=True: cancels pending debounce and flushes right now
        - otherwise: debounces using scheduler.after()

        Matches original behavior. :contentReference[oaicite:7]{index=7}
        """
        # If no scheduler (or asked), do immediate write
        if immediate or self.scheduler is None:
            self._cancel_pending()
            self.flush_now(cam_config)
            return

        # Debounced: cancel pending and schedule new flush
        self._cancel_pending()
        self._save_debounce_id = self.scheduler.after(delay_ms, lambda: self._flush_from_timer(cam_config))

    def _flush_from_timer(self, cam_config: Any) -> None:
        # timer fired, clear handle first (avoid stale cancel attempts)
        self._save_debounce_id = None
        self.flush_now(cam_config)

    def _cancel_pending(self) -> None:
        if self._save_debounce_id is not None and self.scheduler is not None:
            try:
                self.scheduler.after_cancel(self._save_debounce_id)
            except Exception:
                pass
        self._save_debounce_id = None
