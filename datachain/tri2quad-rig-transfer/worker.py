from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datachain import BaseWorker

RESULT_SCHEMA = pa.schema(
    [
        ("uuid", pa.string()),
        ("status", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
        ("source", pa.string()),
        ("source_bos_bucket", pa.string()),
        ("source_bos_uri", pa.string()),
        ("final_mesh_uri", pa.string()),
        ("output_bos_bucket", pa.string()),
        ("output_bos_key", pa.string()),
        ("vertex_count", pa.int64()),
        ("joint_count", pa.int64()),
        ("weight_width", pa.int64()),
        ("artifact_bytes", pa.int64()),
        ("max_coordinate_error", pa.float64()),
        ("producer_mode", pa.string()),
        ("deformation_state", pa.string()),
        ("schema", pa.string()),
    ]
)


class Worker(BaseWorker):
    def setup(self):
        self.output_bucket = os.environ["OUTPUT_BOS_BUCKET"]
        self.output_prefix = os.environ["OUTPUT_BOS_PREFIX"]
        self.parts_dir = Path(os.environ["RESULT_PARTS_DIR"])
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    def process_task(self, task):
        payload = dict(task["data"])
        shard_id = str(payload["shard_id"])
        output = self.parts_dir / f"{shard_id}.parquet"
        partial = output.with_suffix(".parquet.partial")
        records = []
        for row in payload["rows"]:
            row = dict(row)
            with tempfile.TemporaryDirectory(prefix="tri2quad-rig-child-") as directory:
                row_path = Path(directory) / "row.json"
                result_path = Path(directory) / "result.json"
                row_path.write_text(json.dumps(row, ensure_ascii=False))
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("asset_cli.py")),
                    "--row",
                    str(row_path),
                    "--result",
                    str(result_path),
                    "--output-bucket",
                    self.output_bucket,
                    "--output-prefix",
                    self.output_prefix,
                ]
                attempts = []
                for _ in range(2):
                    completed = subprocess.run(command, check=False)
                    attempts.append(completed.returncode)
                    if completed.returncode == 0 and result_path.is_file():
                        break
                if result_path.is_file():
                    record = json.loads(result_path.read_text())
                else:
                    record = {
                        "uuid": str(row["uuid"]),
                        "status": "rejected",
                        "reason": "blender_process_crash",
                        "detail": f"isolated child exit codes: {attempts}",
                        "source": str(row.get("source", "")),
                        "source_bos_bucket": str(row.get("bos_bucket", "")),
                        "source_bos_uri": str(row.get("bos_uri", "")),
                        "final_mesh_uri": str(row.get("mesh_path", "")),
                        "output_bos_bucket": "",
                        "output_bos_key": "",
                        "vertex_count": 0,
                        "joint_count": 0,
                        "weight_width": 0,
                        "artifact_bytes": 0,
                        "max_coordinate_error": None,
                        "producer_mode": "",
                        "deformation_state": "",
                        "schema": "nexus.tri2quad-rig.v1",
                    }
            records.append(record)
        try:
            pq.write_table(
                pa.Table.from_pylist(records, schema=RESULT_SCHEMA),
                partial,
                compression="zstd",
            )
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
        return {
            "stage": "tri2quad_rig_transfer",
            "shard_id": shard_id,
            "rows": len(records),
            "part_path": str(output),
            "task_id": task["id"],
        }
