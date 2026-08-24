from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from source import TransferResultSource


class FakeBos:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get_object_as_string(self, bucket: str, key: str) -> bytes:
        if (bucket, key) != ("artifacts", "accepted.npz"):
            raise AssertionError((bucket, key))
        return self.payload


def artifact() -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        triangles=np.asarray([[0, 1, 2]], np.int32),
        quads=np.empty((0, 4), np.int32),
        joint_names=np.asarray(["root", "unused", "tip"]),
        joint_positions=np.asarray([[0, 0, 0], [0, 0.5, 0], [0, 1, 0]], np.float32),
        parents=np.asarray([-1, 0, 1], np.int32),
        weight_indices=np.asarray([[0, 2], [2, -1], [0, -1]], np.int32),
        weight_values=np.asarray([[0.75, 0.25], [1, 0], [1, 0]], np.float32),
    )
    return stream.getvalue()


class TransferResultSourceTest(unittest.TestCase):
    def test_indexes_only_accepted_rows_and_preserves_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            rows = [
                {
                    "uuid": "accepted",
                    "source": "fixture",
                    "output_bos_bucket": "artifacts",
                    "output_bos_key": "accepted.npz",
                    "artifact_bytes": 123,
                    "vertex_count": 3,
                    "joint_count": 2,
                    "weight_width": 2,
                    "max_coordinate_error": 0.0,
                    "status": "accepted",
                },
                {
                    "uuid": "rejected",
                    "source": "fixture",
                    "output_bos_bucket": "",
                    "output_bos_key": "",
                    "artifact_bytes": 0,
                    "vertex_count": 0,
                    "joint_count": 0,
                    "weight_width": 0,
                    "max_coordinate_error": None,
                    "status": "rejected",
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), parts / "part.parquet")
            source = TransferResultSource(parts, bos_client=FakeBos(artifact()))

            summaries, matched, total = source.query_assets("fixture", 0, 10)
            self.assertEqual([summary.id for summary in summaries], ["accepted"])
            self.assertEqual((matched, total), (1, 1))
            asset = source.get_asset("accepted")
            self.assertEqual(asset.mesh.vertices[1], [1.0, 0.0, 0.0])
            self.assertEqual(asset.skeletons[0].names, ["root", "tip"])
            self.assertEqual(asset.skeletons[0].parents, [-1, 0])
            self.assertEqual(asset.mesh.skinning.joint_indices[0], [0, 1])
            self.assertEqual(asset.mesh.skinning.weights[0], [0.75, 0.25])


if __name__ == "__main__":
    unittest.main()
