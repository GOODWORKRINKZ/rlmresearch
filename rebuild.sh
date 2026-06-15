#!/usr/bin/env bash
# rebuild.sh — Clean rebuild and restart of the RLM research server.
# Usage: ./rebuild.sh
#
# Ensures a FRESH image is built every time by:
# 1. Stopping and removing the old container
# 2. Removing the old image (so no layer cache can sneak in)
# 3. Building with --no-cache
# 4. Starting the new container
# 5. Verifying the patch is applied

set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="rlmresearch-rlm-server"
CONTAINER_NAME="rlm-research-server"

echo "=== Step 1: Stop and remove old container ==="
docker compose down 2>&1 || true

echo ""
echo "=== Step 2: Remove old image ==="
docker rmi "${IMAGE_NAME}:latest" 2>/dev/null && echo "Old image removed." || echo "No old image to remove."

echo ""
echo "=== Step 3: Build fresh image (--no-cache) ==="
docker compose build --no-cache 2>&1

echo ""
echo "=== Step 4: Start container ==="
docker compose up -d 2>&1

echo ""
echo "=== Step 5: Wait for startup ==="
sleep 3

echo ""
echo "=== Step 6: Verify image and file ==="
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
echo "=== Step 7: Verify patch in container ==="
if docker exec "${CONTAINER_NAME}" grep -q "sys.stderr.write.*Patched" /app/src/rlm_assistant/rlm_client.py 2>/dev/null; then
    echo "✅ Patch code is present in container."
else
    echo "❌ Patch code NOT found in container!"
    exit 1
fi

echo ""
echo "=== Step 8: Check server health ==="
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
