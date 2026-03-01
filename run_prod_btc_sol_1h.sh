#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Production runner (BTC + SOL)
# -----------------------------
# Usage:
#   ./run_prod_btc_sol_1h.sh start
#   ./run_prod_btc_sol_1h.sh stop
#   ./run_prod_btc_sol_1h.sh restart
#   ./run_prod_btc_sol_1h.sh status
#   ./run_prod_btc_sol_1h.sh logs
#
# Notes:
# - Uses venv python: .venv/bin/python
# - Writes logs to ./logs/
# - Writes pidfiles to ./pids/
# - Status uses pidfiles + kill -0 (reliable)
# - Validates that processes stay alive after startup

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/pids"
PY="$ROOT_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR" "$PID_DIR"

export PYTHONUNBUFFERED=1
export TZ="Australia/Brisbane"

# Scripts (update here if you move files)
BTC_SCRIPT="$ROOT_DIR/code/strategies/envelope/run_btc_trend_1h_v5_prod.py"
SOL_SCRIPT="$ROOT_DIR/code/strategies/envelope/run_sol_bbrsi_1h_v5_prod.py"

# Logs
BTC_LOG="$LOG_DIR/btc_1h.log"
SOL_LOG="$LOG_DIR/sol_1h.log"

# PID files
BTC_PIDFILE="$PID_DIR/btc_1h.pid"
SOL_PIDFILE="$PID_DIR/sol_1h.pid"

die() { echo "[ERR] $*" >&2; exit 1; }

check_prereqs() {
  [[ -x "$PY" ]] || die "Python venv not found/executable: $PY (create venv in $ROOT_DIR/.venv)"
  [[ -f "$BTC_SCRIPT" ]] || die "BTC script not found: $BTC_SCRIPT"
  [[ -f "$SOL_SCRIPT" ]] || die "SOL script not found: $SOL_SCRIPT"
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

read_pidfile() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && cat "$pidfile" 2>/dev/null || true
}

start_one() {
  local name="$1"
  local script="$2"
  local logfile="$3"
  local pidfile="$4"

  # If pidfile exists and process is alive, don't restart
  local oldpid
  oldpid="$(read_pidfile "$pidfile")"
  if is_running_pid "$oldpid"; then
    echo "[OK] $name already running (pid=$oldpid)"
    return 0
  fi

  # Clean stale pidfile
  rm -f "$pidfile"

  echo "[..] Starting $name..."
  cd "$ROOT_DIR"

  # Ensure log file exists (and keep permissions stable)
  touch "$logfile"

  # Start in background, redirect stdout+stderr to logfile
  nohup "$PY" -u "$script" >>"$logfile" 2>&1 &
  local pid=$!

  # Save the REAL PID of the background python process
  echo "$pid" > "$pidfile"

  # Verify it stays alive
  sleep 2
  if is_running_pid "$pid"; then
    echo "[OK] $name started (pid=$pid)"
  else
    echo "[NO] $name died at startup (pid=$pid). Last log lines:"
    tail -n 120 "$logfile" || true
    return 1
  fi
}

status_one() {
  local name="$1"
  local pidfile="$2"
  local pid
  pid="$(read_pidfile "$pidfile")"
  if is_running_pid "$pid"; then
    echo "[OK] $name running (pid=$pid)"
    return 0
  fi
  echo "[NO] $name not running"
  return 1
}

stop_one() {
  local name="$1"
  local pidfile="$2"

  local pid
  pid="$(read_pidfile "$pidfile")"
  if is_running_pid "$pid"; then
    echo "[..] Stopping $name (pid=$pid)"
    kill "$pid" 2>/dev/null || true
    sleep 2
    if is_running_pid "$pid"; then
      echo "[..] Force killing $name (pid=$pid)"
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "[..] $name not running (no live pid)"
  fi
  rm -f "$pidfile"
  echo "[OK] $name stopped"
}

show_logs() {
  echo "===== BTC (last 200) ====="
  tail -n 200 "$BTC_LOG" 2>/dev/null || echo "(no $BTC_LOG yet)"
  echo
  echo "===== SOL (last 200) ====="
  tail -n 200 "$SOL_LOG" 2>/dev/null || echo "(no $SOL_LOG yet)"
}

cmd="${1:-start}"

check_prereqs

case "$cmd" in
  start)
    start_one "BTC 1H" "$BTC_SCRIPT" "$BTC_LOG" "$BTC_PIDFILE"
    start_one "SOL 1H" "$SOL_SCRIPT" "$SOL_LOG" "$SOL_PIDFILE"
    ;;
  stop)
    stop_one "BTC 1H" "$BTC_PIDFILE"
    stop_one "SOL 1H" "$SOL_PIDFILE"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status_one "BTC 1H" "$BTC_PIDFILE" || true
    status_one "SOL 1H" "$SOL_PIDFILE" || true
    ;;
  logs)
    show_logs
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 2
    ;;
esac
