"""Single-source-of-truth Parquet manifest helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


CORE_REQUIRED_COLUMNS = (
    "uuid",
    "split",
    "rig_npz_uri",
)


def read_manifest(
    path: str | Path,
    split: str | None = None,
    view_indices: tuple[int, ...] | list[int] = (0,),
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(CORE_REQUIRED_COLUMNS)
    required.update(f"color_view_{index}_uri" for index in view_indices)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    if frame["uuid"].duplicated().any():
        raise ValueError(f"Manifest {path} contains duplicate UUIDs")
    if split is not None:
        frame = frame.loc[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"Manifest {path} has no rows for split={split!r}")
    return frame.reset_index(drop=True)


def parse_bos_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "bos" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected bos://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")
