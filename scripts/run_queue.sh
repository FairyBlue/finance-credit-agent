#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${FINANCE_AGENT_LOG_ROOT:-$PROJECT_ROOT/runs/queue_logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_ROOT"

if [[ "$#" -eq 0 ]]; then
  set -- \
    "$PROJECT_ROOT/configs/arm_a1.yaml" \
    "$PROJECT_ROOT/configs/arm_a2.yaml" \
    "$PROJECT_ROOT/configs/arm_a3.yaml"
fi

TRAIN_PID=""
WATCHDOG_PID=""

cleanup() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill "$TRAIN_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

for CONFIG_PATH in "$@"; do
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Skipping missing config: $CONFIG_PATH" >&2
    continue
  fi
  CONFIG_STEM="$(basename "$CONFIG_PATH" .yaml)"
  TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
  LOG_FILE="$LOG_ROOT/${CONFIG_STEM}_${TIMESTAMP}.log"
  echo "Starting $CONFIG_PATH; log=$LOG_FILE"

  (
    cd "$PROJECT_ROOT" || exit 1
    exec uv run python -m src.agents.run --config "$CONFIG_PATH"
  ) >"$LOG_FILE" 2>&1 &
  TRAIN_PID="$!"

  (
    cd "$PROJECT_ROOT" || exit 1
    exec uv run python scripts/watchdog.py \
      --pid "$TRAIN_PID" \
      --log-file "$LOG_FILE" \
      --events-file "$LOG_ROOT/watchdog_events.jsonl"
  ) &
  WATCHDOG_PID="$!"

  wait "$TRAIN_PID"
  RUN_STATUS="$?"
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  wait "$WATCHDOG_PID" 2>/dev/null || true
  TRAIN_PID=""
  WATCHDOG_PID=""
  echo "Finished $CONFIG_PATH with status $RUN_STATUS"
done

trap - INT TERM EXIT
