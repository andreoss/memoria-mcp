#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="memoria-server:nonroot-test"
container_name="memoria-server-nonroot-test-$$"
host_port="18180"

cleanup() {
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "${image_tag}" -f "${repo_root}/server/Dockerfile" "${repo_root}"

docker run -d --name "${container_name}" \
  -e MEMORIA_ALLOW_NO_AUTH=1 \
  -e MEMORIA_JWT_SECRET=docker-nonroot-test-jwt-secret \
  -p "${host_port}:8080" \
  "${image_tag}" >/dev/null

health_ok=false
for _ in $(seq 1 20); do
  if curl -s -o /dev/null "http://127.0.0.1:${host_port}/health"; then
    health_ok=true
    break
  fi
  sleep 0.5
done

if [[ "${health_ok}" != "true" ]]; then
  echo "FAIL: server never became reachable on the published port (host_port=${host_port}) -- this is exactly the class of bug a container binding to 127.0.0.1 internally instead of 0.0.0.0 would cause: docker exec still works, but nothing outside the container's own network namespace can ever reach it"
  docker logs "${container_name}" || true
  exit 1
fi

echo "PASS: server is reachable via the published port, not just via docker exec"

uid="$(docker exec "${container_name}" id -u)"

if [[ "${uid}" == "0" ]]; then
  echo "FAIL: container is running as root (uid 0)"
  exit 1
fi

echo "PASS: container is running as uid ${uid} (non-root)"

if docker exec "${container_name}" sh -c 'touch /root/should-not-be-writable' >/dev/null 2>&1; then
  echo "FAIL: process could write to /root -- uid is non-zero but still has root-equivalent access"
  exit 1
fi

echo "PASS: process cannot write to /root (genuinely unprivileged, not just cosmetically non-zero uid)"
