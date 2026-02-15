#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="memoria-server:nonroot-test"
container_name="memoria-server-nonroot-test-$$"

cleanup() {
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "${image_tag}" -f "${repo_root}/server/Dockerfile" "${repo_root}"

docker run -d --name "${container_name}" "${image_tag}" >/dev/null

uid="$(docker exec "${container_name}" id -u)"

if [[ "${uid}" == "0" ]]; then
  echo "FAIL: container is running as root (uid 0)"
  exit 1
fi

echo "PASS: container is running as uid ${uid} (non-root)"
