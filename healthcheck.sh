#!/bin/bash
# Health check script for topic-classifier service
# Returns 0 (success) if the service is responding (model loading is optional for lazy init)

set -e

# Make request to health endpoint
HEALTH_RESPONSE=$(curl -s -f http://localhost:${SERVICE_PORT:-8009}/api/v1/health || echo "")

if [ -z "$HEALTH_RESPONSE" ]; then
  echo "Health endpoint unreachable"
  exit 1
fi

# Check if status is "healthy"
STATUS_OK=$(echo "$HEALTH_RESPONSE" | grep -o '"status"[[:space:]]*:[[:space:]]*"healthy"' || echo "")

if [ -z "$STATUS_OK" ]; then
  echo "Service reports unhealthy status"
  exit 1
fi

# Optional: Log if model is loaded
MODEL_LOADED=$(echo "$HEALTH_RESPONSE" | grep -o '"model_loaded"[[:space:]]*:[[:space:]]*true' || echo "")
if [ -n "$MODEL_LOADED" ]; then
  echo "Service healthy and model loaded"
else
  echo "Service healthy (model will load on first request)"
fi

exit 0
