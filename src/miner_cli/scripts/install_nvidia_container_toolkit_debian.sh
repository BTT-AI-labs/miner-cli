#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia container toolkit already installed"
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
echo "nvidia container toolkit installed"
