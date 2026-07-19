#!/usr/bin/env bash
# run_server.sh — one-command restart + healthcheck for the claimprocessing FastAPI app.
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/claimprocessing}"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
APP_MODULE="app.main:app"
HOST="0.0.0.0"
PORT="8000"
LOG_FILE="$PROJECT_DIR/reports/server.log"
STARTUP_TIMEOUT=40
DRIFT=0
STOP_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drift)  DRIFT=1; shift ;;
    --port)   PORT="$2"; shift 2 ;;
    --stop)   STOP_ONLY=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1 (use --help)"; exit 2 ;;
  esac
done

port_in_use() { lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null; }

stop_server() {
  echo ">>> Stopping any uvicorn on '$APP_MODULE' ..."
  pkill -f "uvicorn $APP_MODULE" 2>/dev/null
  for i in $(seq 1 8); do
    [[ -z "$(port_in_use)" ]] && { echo ">>> Port $PORT is free."; return 0; }
    sleep 1
  done
  local pids; pids="$(port_in_use)"
  if [[ -n "$pids" ]]; then
    echo ">>> Port $PORT still held by PID(s): $pids — force killing."
    kill -9 $pids 2>/dev/null; sleep 2
  fi
  [[ -z "$(port_in_use)" ]] && echo ">>> Port $PORT is free." \
                            || { echo "!!! Could not free port $PORT."; return 1; }
}

cd "$PROJECT_DIR" || { echo "!!! Cannot cd to $PROJECT_DIR"; exit 1; }
stop_server || exit 1
[[ "$STOP_ONLY" -eq 1 ]] && { echo ">>> --stop done."; exit 0; }

if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "!!! venv not found at $VENV_ACTIVATE (continuing with system python)"
fi

if [[ "$DRIFT" -eq 1 ]]; then
  export ENABLE_DRIFT=1; echo ">>> ENABLE_DRIFT=1 (drift computation ON)"
else
  echo ">>> ENABLE_DRIFT unset (defaults OFF)"
fi

: > "$LOG_FILE"
echo ">>> Starting uvicorn on $HOST:$PORT ..."
nohup uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo ">>> uvicorn PID: $SERVER_PID  (logs: $LOG_FILE)"

echo -n ">>> Waiting for startup"
for i in $(seq 1 "$STARTUP_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo; echo "!!! Server exited during startup. Last log lines:"; tail -n 20 "$LOG_FILE"; exit 1
  fi
  if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
    echo " — UP."; break
  fi
  echo -n "."; sleep 1
  if [[ "$i" -eq "$STARTUP_TIMEOUT" ]]; then
    echo; echo "!!! Timed out after ${STARTUP_TIMEOUT}s. Last log lines:"; tail -n 20 "$LOG_FILE"; exit 1
  fi
done

echo ">>> Healthcheck: POST /eval/drift"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "localhost:$PORT/eval/drift" \
  -H "Content-Type: application/json" -d '{}')
case "$CODE" in
  200) echo ">>> HTTP 200 OK — endpoint healthy. Response:"
       curl -s -X POST "localhost:$PORT/eval/drift" -H "Content-Type: application/json" -d '{}' | python3 -m json.tool ;;
  422) echo "!!! HTTP 422 — body still required. Apply the W8-10-7a optional-body fix." ;;
  000) echo "!!! No response (000) — server up but endpoint unreachable. Check $LOG_FILE." ;;
  *)   echo "!!! Unexpected HTTP $CODE. Check $LOG_FILE." ;;
esac

echo; echo ">>> Done. Tail logs:  tail -f $LOG_FILE"
echo ">>> Stop server:      scripts/run_server.sh --stop"
