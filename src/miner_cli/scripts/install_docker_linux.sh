#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "docker already installed"
  exit 0
fi

# Clean up stale Docker apt/yum source config from older installer revisions
# before invoking the official convenience installer.
if [ -d /etc/apt/sources.list.d ]; then
  sudo rm -f /etc/apt/sources.list.d/docker.list
  sudo rm -f /etc/apt/sources.list.d/docker.sources
fi
if [ -d /etc/apt/keyrings ]; then
  sudo rm -f /etc/apt/keyrings/docker.asc
  sudo rm -f /etc/apt/keyrings/docker.gpg
fi
if [ -d /etc/yum.repos.d ]; then
  sudo rm -f /etc/yum.repos.d/docker*.repo
fi

curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
echo "docker installed"
