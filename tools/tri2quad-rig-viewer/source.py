from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient
from rigviz import Asset, AssetSummary, IndexedMesh, Skeleton, Skinning


RESULT_COLUMNS = (
    "uuid",
    "source",
    "output_bos_bucket",
    "output_bos_key",
    "artifact_bytes",
    "vertex_count",
    "joint_count",
    "weight_width",
    "max_coordinate_error",
    "status",
)


def _display_rig(artifact) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = artifact["weight_indices"]
    weights = artifact["weight_values"]
    used = np.unique(indices[(indices >= 0) & (weights > 0)])
    remap = np.full(len(artifact["joint_positions"]), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    display_indices = np.where((indices >= 0) & (weights > 0), remap[indices], -1)

    parents = artifact["parents"]
    display_parents = []
    for joint in used:
        parent = int(parents[joint])
        while parent >= 0 and remap[parent] < 0:
            parent = int(parents[parent])
        display_parents.append(int(remap[parent]) if parent >= 0 else -1)
    return used, np.asarray(display_parents, np.int32), display_indices, weights


class TransferResultSource:
    """Lazy view over atomically written transfer result parts and BOS NPZs."""

    def __init__(
        self, parts_dir: Path, *, cache_size: int = 12, bos_client=None
    ) -> None:
        self.parts_dir = parts_dir
        self.cache_size = cache_size
        self._lock = threading.RLock()
        self._parts: set[Path] = set()
        self._records: dict[str, dict] = {}
        self._ordered_ids: list[str] = []
        self._cache: OrderedDict[str, Asset] = OrderedDict()
        self._bos = bos_client or BosClient(
            BceClientConfiguration(
                credentials=BceCredentials(
                    os.environ["BOS_ACCESS_KEY"], os.environ["BOS_SECRET_KEY"]
                ),
                endpoint=os.environ["BOS_ENDPOINT"],
            )
        )

    @property
    def total_count(self) -> int:
        self._refresh()
        return len(self._records)

    def _refresh(self) -> None:
        with self._lock:
            current = set(self.parts_dir.glob("*.parquet"))
            if not self._parts.issubset(current):
                self._parts.clear()
                self._records.clear()
                self._cache.clear()
            new_parts = sorted(current - self._parts)
            if not new_parts:
                return
            for path in new_parts:
                for row in pq.read_table(path, columns=list(RESULT_COLUMNS)).to_pylist():
                    if row["status"] == "accepted":
                        self._records[str(row["uuid"])] = row
                self._parts.add(path)
            self._ordered_ids = sorted(self._records)

    def list_assets(self):
        summaries, _, _ = self.query_assets("", 0, 1000)
        return summaries

    def query_assets(
        self, query: str, offset: int, limit: int
    ) -> tuple[list[AssetSummary], int, int]:
        self._refresh()
        folded = query.casefold()
        with self._lock:
            matching = [
                uuid
                for uuid in self._ordered_ids
                if not folded
                or folded in uuid.casefold()
                or folded in str(self._records[uuid].get("source", "")).casefold()
            ]
            selected = matching[offset : offset + limit]
            summaries = [
                AssetSummary(
                    id=uuid,
                    name=f"{uuid}  {self._records[uuid].get('source', '')}",
                    format="NPZ",
                    bytes=int(self._records[uuid]["artifact_bytes"]),
                )
                for uuid in selected
            ]
            return summaries, len(matching), len(self._records)

    def get_asset(self, asset_id: str) -> Asset:
        self._refresh()
        with self._lock:
            cached = self._cache.pop(asset_id, None)
            if cached is not None:
                self._cache[asset_id] = cached
                return cached
            try:
                row = self._records[asset_id]
            except KeyError as exc:
                raise KeyError(f"unknown accepted asset: {asset_id}") from exc
        payload = self._bos.get_object_as_string(
            str(row["output_bos_bucket"]), str(row["output_bos_key"])
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
            used, parents, weight_indices, weight_values = _display_rig(artifact)
            asset = Asset(
                id=asset_id,
                name=asset_id,
                mesh=IndexedMesh(
                    vertices=artifact["vertices"].tolist(),
                    triangles=artifact["triangles"].tolist(),
                    quads=artifact["quads"].tolist(),
                    skinning=Skinning(
                        joint_indices=weight_indices.tolist(),
                        weights=weight_values.tolist(),
                    ),
                ),
                skeletons=[
                    Skeleton(
                        positions=artifact["joint_positions"][used].tolist(),
                        parents=parents.tolist(),
                        names=artifact["joint_names"][used].astype(str).tolist(),
                        label="Source rig",
                        color="#e7a84f",
                        xray=True,
                    )
                ],
                metadata={
                    "source": str(row.get("source", "")),
                    "vertices": int(row["vertex_count"]),
                    "joints": int(row["joint_count"]),
                    "weight_width": int(row["weight_width"]),
                    "max_coordinate_error": float(row["max_coordinate_error"]),
                    "artifact": f"bos://{row['output_bos_bucket']}/{row['output_bos_key']}",
                },
            )
        with self._lock:
            self._cache[asset_id] = asset
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return asset
