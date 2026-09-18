"""Small synchronized wall-clock profiler used by the simulator benchmarks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator


class Profiler:
    """Accumulate named wall-clock regions.

    Callers must check :attr:`enabled` before entering :meth:`region`; this keeps the
    disabled simulator path out of a context manager entirely.  A device synchronizer,
    when supplied, is called on both sides of the measured operation.
    """

    def __init__(self, enabled: bool = False, sync: Callable[[], None] | None = None):
        self.enabled = bool(enabled)
        self.sync = sync
        self._regions: dict[str, dict[str, int | float]] = {}

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self.sync is not None:
            self.sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.sync is not None:
                self.sync()
            elapsed = time.perf_counter() - started
            item = self._regions.setdefault(name, {"calls": 0, "total_s": 0.0})
            item["calls"] = int(item["calls"]) + 1
            item["total_s"] = float(item["total_s"]) + elapsed

    def regions(self) -> dict[str, dict[str, int | float]]:
        """Return a copy suitable for stats and JSON serialization."""
        return {name: {"calls": int(v["calls"]), "total_s": float(v["total_s"])}
                for name, v in self._regions.items()}
