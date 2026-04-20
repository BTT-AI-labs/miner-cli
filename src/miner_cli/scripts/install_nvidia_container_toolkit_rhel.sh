#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia container toolkit already installed"
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  PKG_MGR="dnf"
else
  PKG_MGR="yum"
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
sudo "${PKG_MGR}" install -y nvidia-container-toolkit
echo "nvidia container toolkit installed"
