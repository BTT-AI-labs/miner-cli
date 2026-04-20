#!/usr/bin/env bash
set -euo pipefail

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  echo "docker nvidia runtime already configured"
  exit 0
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "docker nvidia runtime configured"
