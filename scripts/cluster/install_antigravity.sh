#!/usr/bin/env bash
set -euo pipefail

# Lightweight login-node setup. Submit heavy project commands through Slurm.
if [[ "$(id -un)" != "mkhoshba" ]]; then
  printf 'Warning: expected user mkhoshba, running as %s\n' "$(id -un)" >&2
fi

command -v curl >/dev/null 2>&1 || {
  echo "curl is required but is not available." >&2
  exit 1
}

installer="${TMPDIR:-/tmp}/antigravity-install.sh"
curl -fsSL https://antigravity.google/cli/install.sh -o "$installer"
bash "$installer"

export PATH="$HOME/.local/bin:$PATH"
command -v agy >/dev/null 2>&1 || {
  echo 'Add export PATH="$HOME/.local/bin:$PATH" to your shell profile.' >&2
  exit 1
}

echo "Antigravity CLI installed at: $(command -v agy)"
agy --version
echo "Run 'agy' in the repository; open its SSH authorization URL locally."

