#!/usr/bin/env bash
# Run the Lighthouse demo locally: API (FastAPI, port 8787) serving the built UI.
#   apps/lighthouse_demo/run_local.sh            build the UI if needed, start the API
#   apps/lighthouse_demo/run_local.sh --dev      API + Vite dev server with hot reload (UI on :5173)
# Uses your Databricks CLI profile (default: lighthouse) and your dev schema prefix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export DATABRICKS_CONFIG_PROFILE="${DATABRICKS_CONFIG_PROFILE:-lighthouse}"
HOST="$(awk -v p="[$DATABRICKS_CONFIG_PROFILE]" '$0==p{f=1;next} /^\[/{f=0} f && $1=="host"{print $3}' ~/.databrickscfg)"
export DATABRICKS_HOST="${DATABRICKS_HOST:-$HOST}"
USER_SLUG="$(databricks current-user me -o json | python3 -c 'import json,sys,re; u=json.load(sys.stdin)["userName"].split("@")[0].lower(); print(re.sub(r"[^a-z0-9]+","_",u).strip("_"))' 2>/dev/null || echo user)"
export LIGHTHOUSE_PLATFORM_PREFIX="${LIGHTHOUSE_PLATFORM_PREFIX:-dev_${USER_SLUG}_}"
export LIGHTHOUSE_SCHEMA="${LIGHTHOUSE_SCHEMA:-${LIGHTHOUSE_PLATFORM_PREFIX}lighthouse}"
export LIGHTHOUSE_ENVIRONMENT="${LIGHTHOUSE_ENVIRONMENT:-dev}"
export MLFLOW_DISABLE_AGENT_HINT=1
WEB="$ROOT/apps/lighthouse_demo/web"
[ -d "$WEB/node_modules" ] || (cd "$WEB" && npm install --no-audit --no-fund)
echo "workspace $DATABRICKS_HOST | prefix $LIGHTHOUSE_PLATFORM_PREFIX | prompts $LIGHTHOUSE_SCHEMA"
if [ "${1:-}" = "--dev" ]; then
  (cd "$WEB" && npm run dev) &
  echo "UI with hot reload: http://localhost:5173"
else
  [ -f "$WEB/dist/index.html" ] || (cd "$WEB" && npm run build)
  echo "Open http://localhost:8787 (warm up first: curl -X POST localhost:8787/api/warmup)"
fi
exec "$ROOT/.venv/bin/uvicorn" lighthouse_poc.api.app:create_app --factory --port 8787
