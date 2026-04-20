#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia container toolkit already installed"
  exit 0
fi

sudo pacman -Sy --noconfirm nvidia-container-toolkit
echo "nvidia container toolkit installed"
