#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "docker already installed"
  exit 0
fi

sudo pacman -Sy --noconfirm docker docker-compose
sudo systemctl enable --now docker
echo "docker installed"
