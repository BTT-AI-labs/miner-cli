#!/usr/bin/env bash
set -euo pipefail

TARGET_USER="${USER:-}"
if [ -z "${TARGET_USER}" ]; then
  echo "unable to determine current user"
  exit 1
fi

if ! getent group docker >/dev/null 2>&1; then
  sudo groupadd docker
fi

if id -nG "${TARGET_USER}" | grep -qw docker; then
  echo "docker group already assigned to ${TARGET_USER}"
  exit 0
fi

sudo usermod -aG docker "${TARGET_USER}"
echo "added ${TARGET_USER} to docker group; open a new shell or run newgrp docker"
