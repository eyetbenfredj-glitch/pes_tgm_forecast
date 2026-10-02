"""
DATA_MODE enforcement (per continuation CDC, section 1 & 34).

DATA_MODE=real:
    The pipeline MUST use real data end-to-end. If a real source (weather or
    production) cannot be obtained, the pipeline must STOP and raise
    RealDataUnavailableError naming exactly what is missing - it must NEVER
    silently substitute synthetic data.

DATA_MODE=demo:
    Synthetic/proxy data is allowed, but every artifact produced under this
    mode must be tagged DEMO_ONLY and must never be presented as validated
    real-world performance.

Read from the DATA_MODE environment variable (see .env.example), default
"demo" (fail-safe: a misconfigured deployment shows visible DEMO_ONLY labels
rather than silently training on synthetic data while claiming to be real).
"""
from __future__ import annotations

import os
from enum import Enum


class DataMode(str, Enum):
    REAL = "real"
    DEMO = "demo"


class RealDataUnavailableError(RuntimeError):
    """Raised when DATA_MODE=real but a required real data source is missing
    or unreachable. Carries the exact missing source so the caller can report
    it instead of silently degrading."""

    def __init__(self, missing_source: str, detail: str = ""):
        self.missing_source = missing_source
        self.detail = detail
        super().__init__(
            f"DATA_MODE=real requires '{missing_source}' but it is unavailable. {detail} "
            f"Refusing to substitute synthetic data. Set DATA_MODE=demo to run in "
            f"clearly-labelled DEMO_ONLY mode instead."
        )


def get_data_mode() -> DataMode:
    raw = os.environ.get("DATA_MODE", "demo").strip().lower()
    try:
        return DataMode(raw)
    except ValueError:
        raise ValueError(f"Invalid DATA_MODE={raw!r}; must be 'real' or 'demo'")


def require_real(source_name: str, available: bool, detail: str = "") -> None:
    """Call at the point where a real source would be used. If DATA_MODE=real
    and the source is not available, raises loudly. If DATA_MODE=demo, is a
    no-op (the caller is expected to fall back to a labelled synthetic path)."""
    if get_data_mode() == DataMode.REAL and not available:
        raise RealDataUnavailableError(source_name, detail)
