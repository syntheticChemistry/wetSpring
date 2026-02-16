#!/usr/bin/env bash
# wetSpring — Start Galaxy and install bioinformatics tools
#
# Usage:
#   ./scripts/start_galaxy.sh
#
# This script:
#   1. Starts Galaxy via Docker Compose
#   2. Waits for Galaxy to become healthy
#   3. Reports the access URL and next steps
set -euo pipefail

GALAXY_DIR="$(cd "$(dirname "$0")/../control/galaxy" && pwd)"
GALAXY_URL="http://localhost:8080"

echo "═══════════════════════════════════════════════════════════"
echo "  wetSpring — Galaxy Bootstrap"
echo "═══════════════════════════════════════════════════════════"
echo

# ── Start Galaxy ──────────────────────────────────────────────────
echo "  [1/3] Starting Galaxy Docker..."
cd "$GALAXY_DIR"
docker compose up -d

# ── Wait for Galaxy to become ready ──────────────────────────────
echo "  [2/3] Waiting for Galaxy to start (this takes 2-5 minutes)..."
echo "         Galaxy image is ~4 GB — first pull may take longer."
echo

MAX_WAIT=600  # 10 minutes max
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if Galaxy API responds
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$GALAXY_URL/api/version" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        VERSION=$(curl -s "$GALAXY_URL/api/version" 2>/dev/null | grep -o '"version_major"[^,]*' | head -1 || echo "unknown")
        echo "  [OK] Galaxy is running! ($VERSION)"
        break
    fi
    echo "    ...waiting (${ELAPSED}s, HTTP $HTTP_CODE)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "  [WARN] Galaxy did not start within ${MAX_WAIT}s."
    echo "         Check: docker compose logs -f"
    exit 1
fi

# ── Report ────────────────────────────────────────────────────────
echo
echo "  [3/3] Galaxy is ready."
echo
echo "═══════════════════════════════════════════════════════════"
echo "  Galaxy UI:    $GALAXY_URL"
echo "  Admin login:  admin@galaxy.org / admin"
echo "  FTP upload:   localhost:8021"
echo "  SFTP upload:  localhost:8022"
echo
echo "  To install bioinformatics tools:"
echo "    1. Log in as admin"
echo "    2. Admin → Install New Tools"
echo "    3. Search and install from tool_lists/amplicon_tools.yml"
echo "       Or use the Galaxy API:"
echo "         pip install ephemeris"
echo "         shed-tools install -g $GALAXY_URL -a <api_key> \\"
echo "           -t tool_lists/amplicon_tools.yml"
echo
echo "  To stop Galaxy:"
echo "    cd control/galaxy && docker compose down"
echo
echo "  Data volumes persist across restarts."
echo "═══════════════════════════════════════════════════════════"
