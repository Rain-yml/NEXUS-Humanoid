#!/usr/bin/env bash
set -euo pipefail

TASK_NAME=${TASK_NAME:?TASK_NAME is required}
MANIFEST=${MANIFEST:?MANIFEST is required}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
RESULT_PARTS_DIR=${RESULT_PARTS_DIR:-${RUN_DIR}/parts}
ROWS_PER_TASK=${ROWS_PER_TASK:-1}
TASKS_PER_FILE=${TASKS_PER_FILE:-256}
LIMIT=${LIMIT:-0}
UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/pyenv}

mkdir -p "$RUN_DIR/tasks" "$RESULT_PARTS_DIR"
rm -f "$RESULT_PARTS_DIR"/part-*.parquet*
UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run --locked --no-sync python make_tasks.py \
  --manifest "$MANIFEST" --out "$RUN_DIR/tasks" \
  --rows-per-task "$ROWS_PER_TASK" --tasks-per-file "$TASKS_PER_FILE" --limit "$LIMIT"

UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run --locked --no-sync python -m datachain clear \
  --task-name "$TASK_NAME" --confirm
for task_file in "$RUN_DIR"/tasks/part-*.json.gz; do
  UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run --locked --no-sync python -m datachain add \
    "$task_file" --task-name "$TASK_NAME"
done
