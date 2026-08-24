from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bos_io import BOS
from pipeline import process, rejection


def main():
    parser = argparse.ArgumentParser(description="Process one asset in an isolated Blender runtime")
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    row = json.loads(args.row.read_text())
    try:
        record = process(row, BOS(), args.output_bucket, args.output_prefix)
        record["detail"] = ""
    except Exception as error:
        record = rejection(row, error)
    partial = args.result.with_suffix(args.result.suffix + ".partial")
    partial.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(partial, args.result)


if __name__ == "__main__":
    main()
