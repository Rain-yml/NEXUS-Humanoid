from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


FIELDS = ("uuid", "source", "source_id", "bos_bucket", "bos_uri", "mesh_path")


def write(path: Path, tasks: list[dict]):
    partial = path.with_suffix(path.suffix + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        json.dump(tasks, handle, ensure_ascii=False)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description="Build bounded DataChain task files")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-task", type=int, default=1)
    parser.add_argument("--tasks-per-file", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.rows_per_task < 1 or args.tasks_per_file < 1 or args.limit < 0:
        parser.error("task sizes must be positive and limit nonnegative")
    parquet = pq.ParquetFile(args.manifest)
    missing = sorted(set(FIELDS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"manifest is missing {missing}")
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("part-*.json.gz*"):
        old.unlink()
    rows = []
    tasks = []
    row_count = task_count = file_count = 0

    def flush_task():
        nonlocal rows, task_count
        if rows:
            tasks.append({"shard_id": f"part-{task_count:07d}", "rows": rows})
            rows = []
            task_count += 1

    def flush_file():
        nonlocal tasks, file_count
        if tasks:
            write(args.out / f"part-{file_count:05d}.json.gz", tasks)
            tasks = []
            file_count += 1

    stop = False
    for batch in parquet.iter_batches(columns=FIELDS, batch_size=65536):
        for row in batch.to_pylist():
            if args.limit and row_count >= args.limit:
                stop = True
                break
            rows.append(row)
            row_count += 1
            if len(rows) == args.rows_per_task:
                flush_task()
            if len(tasks) == args.tasks_per_file:
                flush_file()
        if stop:
            break
    flush_task()
    flush_file()
    summary = {
        "manifest": str(args.manifest.resolve()),
        "rows": row_count,
        "rows_per_task": args.rows_per_task,
        "tasks": task_count,
        "task_files": file_count,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
