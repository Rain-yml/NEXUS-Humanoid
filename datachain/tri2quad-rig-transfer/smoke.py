from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from artifact import encode_artifact
from bos_io import BOS, split_bos_uri
from mesh_io import read_ply
from pipeline import artifact_key, solve


def main():
    parser = argparse.ArgumentParser(description="Run strict transfer without writing BOS")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--uuid", action="append", required=True)
    parser.add_argument("--output-bucket")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--expected-deformation-state", choices=("rest", "pose"))
    args = parser.parse_args()
    table = pq.read_table(args.manifest)
    bos = BOS()
    for uuid in args.uuid:
        rows = table.filter(pc.equal(table["uuid"], uuid)).to_pylist()
        if len(rows) != 1:
            raise ValueError(f"expected one row for {uuid}, found {len(rows)}")
        row = rows[0]
        bucket, key = split_bos_uri(row["mesh_path"])
        final = read_ply(bos.read(bucket, key))
        source = bos.read(row["bos_bucket"], row["bos_uri"])
        try:
            normalized, result = solve(source, final)
        except Exception as error:
            print(json.dumps({"uuid": uuid, "error": f"{type(error).__name__}: {error}"}), flush=True)
            continue
        if (
            args.expected_deformation_state
            and normalized.deformation_state != args.expected_deformation_state
        ):
            raise RuntimeError(
                f"{uuid}: expected {args.expected_deformation_state} deformation "
                f"state, got {normalized.deformation_state}"
            )
        output_key = ""
        if args.output_bucket:
            output_key = artifact_key(args.output_prefix, uuid)
            bos.write(args.output_bucket, output_key, encode_artifact(uuid, result))
        print(
            json.dumps(
                {
                    "uuid": uuid,
                    "source": row["source"],
                    "vertices": len(result.final_vertices),
                    "joints": len(result.joint_names),
                    "weight_width": result.weight_indices.shape[1],
                    "producer_mode": normalized.producer_mode,
                    "max_coordinate_error": result.max_coordinate_error,
                    "source_to_final": result.source_to_final.tolist(),
                    "deformation_state": normalized.deformation_state,
                    "output_key": output_key,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
