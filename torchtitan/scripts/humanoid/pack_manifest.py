"""Create deterministic NEXUS layer-packed batches from a humanoid manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def first_fit_decreasing(items: list[tuple[int, list]], budget: int) -> list[list[list]]:
    bins: list[list[list]] = []
    remaining: list[int] = []
    for tokens, record in sorted(items, key=lambda item: (-item[0], item[1])):
        if tokens > budget:
            raise ValueError(f"Layer {record} has {tokens} tokens, above budget {budget}")
        for index, capacity in enumerate(remaining):
            if tokens <= capacity:
                bins[index].append(record)
                remaining[index] -= tokens
                break
        else:
            bins.append([record])
            remaining.append(budget - tokens)
    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--token-budget", required=True, type=int)
    parser.add_argument("--pad-to-multiple", type=int, default=1)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest)
    frame = frame.loc[frame["split"] == args.split].sort_values("uuid").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows for split={args.split!r}")
    if "octree_layer_tokens" not in frame:
        raise ValueError("Manifest is missing octree_layer_tokens")

    items = []
    for sample_id, row in frame.iterrows():
        layer_tokens = [int(value) for value in row["octree_layer_tokens"]]
        if len(layer_tokens) != 9:
            raise ValueError(f"{row.uuid} has {len(layer_tokens)} octree layers, expected 9")
        for layer_id, tokens in enumerate(layer_tokens):
            items.append((tokens, [str(row.uuid), int(sample_id), layer_id]))

    batches = first_fit_decreasing(items, args.token_budget)
    original_batches = len(batches)
    if args.pad_to_multiple < 1:
        raise ValueError("--pad-to-multiple must be positive")
    while len(batches) % args.pad_to_multiple:
        batches.append([list(item) for item in batches[len(batches) % original_batches]])

    payload = {
        "batches": batches,
        "metadata": {
            "manifest": args.manifest.name,
            "split": args.split,
            "token_budget": args.token_budget,
            "source_layers": len(items),
            "original_batches": original_batches,
            "padded_batches": len(batches),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"Packed {len(items)} layers into {original_batches} batches; "
        f"wrote {len(batches)} batches to {args.output}"
    )


if __name__ == "__main__":
    main()
