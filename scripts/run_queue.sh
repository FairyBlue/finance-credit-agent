#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SMOKE_CONFIG="$PROJECT_ROOT/configs/smoke_1p7b.yaml"
LOG_ROOT="${FINANCE_AGENT_LOG_ROOT:-$PROJECT_ROOT/runs/queue_logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_queue.sh configs/smoke_1p7b.yaml
  scripts/run_queue.sh --formal <config.yaml> [<config.yaml> ...]

No default queue is allowed. Formal configs require the explicit --formal authorization flag.
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

FORMAL_MODE=0
if [[ "$1" == "--formal" ]]; then
  FORMAL_MODE=1
  shift
fi
if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

CONFIGS=()
for CONFIG_ARGUMENT in "$@"; do
  if [[ ! -f "$CONFIG_ARGUMENT" ]]; then
    echo "Config does not exist: $CONFIG_ARGUMENT" >&2
    exit 2
  fi
  CONFIG_DIRECTORY="$(cd "$(dirname "$CONFIG_ARGUMENT")" && pwd -P)"
  CONFIGS+=("$CONFIG_DIRECTORY/$(basename "$CONFIG_ARGUMENT")")
done

if [[ "$FORMAL_MODE" -eq 0 ]]; then
  if [[ "${#CONFIGS[@]}" -ne 1 || "${CONFIGS[0]}" != "$SMOKE_CONFIG" ]]; then
    echo "Only the explicit configs/smoke_1p7b.yaml is allowed without --formal." >&2
    exit 2
  fi
else
  for CONFIG_PATH in "${CONFIGS[@]}"; do
    if [[ "$CONFIG_PATH" == "$SMOKE_CONFIG" ]]; then
      echo "Smoke must run without --formal; do not mix smoke and formal queues." >&2
      exit 2
    fi
  done
fi

mkdir -p "$LOG_ROOT"

TRAIN_PID=""
WATCHDOG_PID=""

cleanup_children() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill "$TRAIN_PID" 2>/dev/null || true
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ -n "$TRAIN_PID" ]]; then
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
}

handle_signal() {
  exit 130
}

trap cleanup_children EXIT
trap handle_signal INT TERM

run_preflight() {
  local config_path="$1"
  local config_stem timestamp preflight_log
  config_stem="$(basename "$config_path" .yaml)"
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  preflight_log="$LOG_ROOT/${config_stem}_${timestamp}_preflight.log"
  echo "Preflighting $config_path; log=$preflight_log"
  (
    cd "$PROJECT_ROOT"
    exec uv run --no-sync python -m src.agents.run \
      --config "$config_path" --preflight-only
  ) >"$preflight_log" 2>&1
}

run_training() {
  local config_path="$1"
  local config_stem timestamp log_file run_status
  config_stem="$(basename "$config_path" .yaml)"
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  log_file="$LOG_ROOT/${config_stem}_${timestamp}.log"
  echo "Starting $config_path; log=$log_file"

  (
    cd "$PROJECT_ROOT"
    exec uv run --no-sync python -m src.agents.run --config "$config_path"
  ) >"$log_file" 2>&1 &
  TRAIN_PID="$!"

  (
    cd "$PROJECT_ROOT"
    exec uv run --no-sync python scripts/watchdog.py \
      --pid "$TRAIN_PID" \
      --log-file "$log_file" \
      --events-file "$LOG_ROOT/watchdog_events.jsonl"
  ) &
  WATCHDOG_PID="$!"

  if wait "$TRAIN_PID"; then
    run_status=0
  else
    run_status="$?"
  fi
  TRAIN_PID=""
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=""
  echo "Finished $config_path with status $run_status"
  return "$run_status"
}

# Validate every requested config before starting the first training process.
for CONFIG_PATH in "${CONFIGS[@]}"; do
  if run_preflight "$CONFIG_PATH"; then
    :
  else
    PREFLIGHT_STATUS="$?"
    echo "Preflight failed for $CONFIG_PATH; no training was started." >&2
    exit "$PREFLIGHT_STATUS"
  fi
done

for CONFIG_PATH in "${CONFIGS[@]}"; do
  if run_training "$CONFIG_PATH"; then
    :
  else
    TRAIN_STATUS="$?"
    echo "Training failed for $CONFIG_PATH; stopping the queue." >&2
    exit "$TRAIN_STATUS"
  fi
done

trap - EXIT INT TERM
cleanup_children
