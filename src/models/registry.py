from __future__ import annotations

from pathlib import Path

from src.config import paths


def latest_model_path(model_dir: Path = paths.models) -> Path | None:
    candidates = sorted(
        model_dir.glob("*_model.pkl"), key=lambda path: path.stat().st_mtime
    )
    return candidates[-1] if candidates else None
