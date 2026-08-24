from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description="Finalize strict transfer results")
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    source = q(args.parts / "*.parquet")
    accepted = args.out / "accepted.parquet"
    rejected = args.out / "rejected.parquet"
    all_rows = args.out / "results.parquet"
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{source}', union_by_name=true) ORDER BY uuid) "
        f"TO '{q(all_rows)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    total, unique = connection.execute(
        f"SELECT count(*), count(DISTINCT uuid) FROM read_parquet('{q(all_rows)}')"
    ).fetchone()
    if total != args.expected or unique != args.expected:
        raise RuntimeError(
            f"incomplete result set: expected={args.expected} rows={total} unique={unique}"
        )
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{q(all_rows)}') WHERE status='accepted') "
        f"TO '{q(accepted)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{q(all_rows)}') WHERE status='rejected') "
        f"TO '{q(rejected)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    counts = dict(
        connection.execute(
            f"SELECT reason, count(*) FROM read_parquet('{q(all_rows)}') GROUP BY reason ORDER BY count(*) DESC"
        ).fetchall()
    )
    accepted_count = counts.pop("", 0)
    summary = {
        "schema": "nexus.tri2quad-rest-rig.summary.v1",
        "processed": total,
        "accepted": accepted_count,
        "rejected": total - accepted_count,
        "rejections": counts,
        "acceptance_contract": "rest-rig SSOT with equal-cardinality vertex bijection within two fixed digits_vertex=6 producer roundings",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
