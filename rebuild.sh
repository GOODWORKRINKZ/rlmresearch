#!/usr/bin/env bash
# rebuild.sh — Rebuild and restart of the RLM research server.
# Usage: ./rebuild.sh [--clean]
#
# Default (no flags): Uses Docker layer cache. Only re-runs steps after
#   the changed file (typically COPY src/ + pip install ~30s).
#   apt-get (~90s) is cached unless Dockerfile changes.
#
# --clean: Full no-cache rebuild (removes image, builds from scratch).
#   Use only when dependencies change or cache is corrupted.

set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="rlmresearch-rlm-server"
CONTAINER_NAME="rlm-research-server"
CLEAN="false"
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN="true"
fi

echo "=== Step 1: Stop and remove old container ==="
docker compose down 2>&1 || true

if [ "$CLEAN" = "true" ]; then
    echo ""
    echo "=== Step 2: Remove old image (--clean mode) ==="
    docker rmi "${IMAGE_NAME}:latest" 2>/dev/null && echo "Old image removed." || echo "No old image to remove."
    echo ""
    echo "=== Step 3: Build fresh image (--no-cache) ==="
    docker compose build --no-cache 2>&1
else
    echo ""
    echo "=== Step 2: Build with layer cache ==="
    docker compose build 2>&1
fi

echo ""
echo "=== Step 3: Start container ==="
docker compose up -d 2>&1

echo ""
echo "=== Step 4: Wait for startup ==="
sleep 3

echo ""
echo "=== Step 5: Verify image and file ==="
CONTAINER_IMAGE=$(docker inspect "${CONTAINER_NAME}" --format '{{.Image}}' 2>/dev/null || echo "UNKNOWN")
LATEST_IMAGE=$(docker images --no-trunc --format '{{.ID}}' "${IMAGE_NAME}:latest" 2>/dev/null || echo "UNKNOWN")
echo "Container image: ${CONTAINER_IMAGE}"
echo "Latest image:    ${LATEST_IMAGE}"

if [ "${CONTAINER_IMAGE}" = "${LATEST_IMAGE}" ]; then
    echo "✅ Container is running the latest image."
else
    echo "❌ MISMATCH — container is NOT running the latest image!"
    exit 1
fi

echo ""
echo "=== Step 6: Verify patch in container ==="
if docker exec "${CONTAINER_NAME}" grep -q "sys.stderr.write.*Patched" /app/src/rlm_assistant/rlm_client.py 2>/dev/null; then
    echo "✅ Patch code is present in container."
else
    echo "❌ Patch code NOT found in container!"
    exit 1
fi

echo ""
echo "=== Step 7: Check server health ==="
for i in 1 2 3 4 5; do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Server is healthy."
        break
    fi
    echo "Waiting for server... ($i/5)"
    sleep 2
done

echo ""
echo "=== Done. Server is running on port 8000 ==="
docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
