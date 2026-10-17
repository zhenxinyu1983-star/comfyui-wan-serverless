#!/bin/bash
# Build and push the serverless worker image.
# Run on a machine with Docker installed (e.g. Docker Desktop).
# Usage: ./build.sh [dockerhub-username] [tag]
# Example: ./build.sh myuser v1
set -e

USERNAME="${1:-}"
TAG="${2:-v1}"

if [ -z "$USERNAME" ]; then
    echo "Usage: ./build.sh <dockerhub-username> [tag]"
    echo "Example: ./build.sh myuser v1"
    exit 1
fi

IMAGE="docker.io/${USERNAME}/comfyui-wan-serverless:${TAG}"

echo "=== Building ${IMAGE} ==="
docker build -t "${IMAGE}" .

echo ""
echo "=== Pushing ${IMAGE} ==="
echo "(you will be prompted for Docker Hub login if not already logged in)"
docker push "${IMAGE}"

echo ""
echo "=== Done ==="
echo "Image: ${IMAGE}"
echo "Use this image when creating the RunPod Serverless endpoint."
