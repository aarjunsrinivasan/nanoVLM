#!/usr/bin/env bash
# Re-provision the parts of the RunPod H100 pod that don't survive a restart. Safe to re-run.
# Only /workspace persists: apt packages, ~/.bashrc and ~/.cache live on the container disk and are wiped.
#
# Usage (after every pod restart):
#   bash scripts/setup_pod.sh && source ~/.bashrc
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPAT_DIR=/usr/local/cuda-13.0/compat   # CUDA 13 forward-compat libcuda: torch 2.14+cu130 on driver 570
UV_CACHE=/workspace/.cache/uv

# 1. System packages
missing=()
for pkg in cuda-compat-13-0 numactl; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Installing: ${missing[*]}"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
else
  echo "System packages already installed: cuda-compat-13-0 numactl"
fi

# 2. Shell env, for new shells and for this script
add_line() { grep -qxF "$1" ~/.bashrc 2>/dev/null || echo "$1" >> ~/.bashrc; }
add_line 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
add_line 'export UV_CACHE_DIR=/workspace/.cache/uv'
add_line '[ -f /workspace/.secrets.env ] && . /workspace/.secrets.env   # WANDB_API_KEY etc., outside the repo'
case ":${LD_LIBRARY_PATH:-}:" in
  *":$COMPAT_DIR:"*) ;;
  *) export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export UV_CACHE_DIR=$UV_CACHE
SECRETS=/workspace/.secrets.env   # lines like: export WANDB_API_KEY=...  (never commit or print this file)
if [ -f "$SECRETS" ]; then chmod 600 "$SECRETS"; . "$SECRETS"; fi

# 2b. GitHub SSH: key, config and known_hosts live in /workspace/.ssh; ~/.ssh is wiped on restart
if [ -f /workspace/.ssh/id_ed25519_github ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  cp /workspace/.ssh/id_ed25519_github /workspace/.ssh/id_ed25519_github.pub /workspace/.ssh/config ~/.ssh/
  chmod 600 ~/.ssh/id_ed25519_github ~/.ssh/config
  [ -f /workspace/.ssh/known_hosts_github ] && cat /workspace/.ssh/known_hosts_github >> ~/.ssh/known_hosts
  sort -u -o ~/.ssh/known_hosts ~/.ssh/known_hosts
  echo "GitHub SSH key restored to ~/.ssh"
else
  echo "Note: no /workspace/.ssh/id_ed25519_github; git push over SSH won't work"
fi

# 3. Python env (no-op when .venv already matches uv.lock)
cd "$REPO_ROOT"
UV_LINK_MODE=copy uv sync --frozen -q

# 4. Checks
echo "GPU / driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
.venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU: check LD_LIBRARY_PATH and cuda-compat-13-0'
print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
"

MANIFEST=.claude/experiments.sha256   # read-only baseline for experiments/
if [ -f "$MANIFEST" ]; then
  if sha256sum -c --quiet "$MANIFEST" && [ "$(find experiments -type f | wc -l)" -eq "$(wc -l < "$MANIFEST")" ]; then
    echo "experiments/ manifest OK (unchanged)"
  else
    echo "WARNING: experiments/ differs from $MANIFEST; it should be read-only" >&2
  fi
else
  echo "No $MANIFEST found; skipping the experiments/ check"
fi

[ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ] \
  || echo "Note: no Hugging Face token. Run '.venv/bin/hf auth login' (saved under \$HF_HOME on /workspace); without it, streaming FineVision may hit Hub rate limits"
[ -n "${WANDB_API_KEY:-}" ] || [ -f ~/.netrc ] \
  || echo "Note: no wandb key. Add 'export WANDB_API_KEY=...' to $SECRETS, or train with --no_log_wandb"
echo "Done. In shells that were already open, run: source ~/.bashrc"
