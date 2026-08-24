from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def quoted(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description="Join strict rig provenance to Tri2Quad")
    parser.add_argument("--rigged", type=Path, required=True)
    parser.add_argument("--tri2quad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(
        f"""
        COPY (
            SELECT r.uuid, r.source, r.source_id, r.bos_bucket, r.bos_uri,
                   t.mesh_path, t.num_vertices
            FROM read_parquet('{quoted(args.rigged)}') r
            INNER JOIN read_parquet('{quoted(args.tri2quad)}') t USING (uuid)
            ORDER BY r.uuid
        ) TO '{quoted(args.out)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    total, unique = connection.execute(
        f"SELECT count(*), count(DISTINCT uuid) FROM read_parquet('{quoted(args.out)}')"
    ).fetchone()
    if total != unique:
        raise RuntimeError(f"joined manifest has duplicate UUIDs: rows={total} unique={unique}")
    print(json.dumps({"rows": total, "manifest": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
