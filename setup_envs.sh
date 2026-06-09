#!/usr/bin/env bash
# setup_envs.sh — create and populate both Sandhi virtual environments
#
# Run once after cloning:
#   bash setup_envs.sh
#
# Creates:
#   sandhi-env/     analysis  (pyxdf, mne, matplotlib, pandas …)
#   interface-env/  LSL / acquisition  (pylsl, muselsl, pyserial …)
#
# Requires: Python 3.11+, pip, no sudo, no conda

set -euo pipefail

PYTHON=${PYTHON:-python3}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── colour helpers ────────────────────────────────────────────────────────────
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red() { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ── python version check ──────────────────────────────────────────────────────
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.11"
if [[ "$(printf '%s\n' "$REQUIRED" "$PY_VERSION" | sort -V | head -1)" != "$REQUIRED" ]]; then
    red "Python $REQUIRED+ required, found $PY_VERSION. Aborting."
    exit 1
fi
green "Python $PY_VERSION found."

# ── helper: build one venv ────────────────────────────────────────────────────
make_env() {
    local name="$1"
    local reqs="$2"
    local env_path="$REPO_DIR/$name"

    yellow "\n── $name ─────────────────────────────────────────"

    if [[ -d "$env_path" ]]; then
        yellow "  $name already exists — skipping creation."
    else
        "$PYTHON" -m venv "$env_path"
        green "  Created $env_path"
    fi

    local pip="$env_path/bin/pip"
    "$pip" install --quiet --upgrade pip
    "$pip" install --quiet -r "$REPO_DIR/$reqs"
    green "  Installed $reqs into $name"

    # quick smoke-test: every top-level package must be importable
    local first_pkg
    first_pkg=$(grep -m1 -E '^[a-zA-Z]' "$REPO_DIR/$reqs" | sed 's/==.*//')
    "$env_path/bin/python" -c "import $first_pkg" \
        && green "  Smoke test passed ($first_pkg imports OK)" \
        || { red "  Smoke test failed for $first_pkg"; exit 1; }
}

# ── build both envs ───────────────────────────────────────────────────────────
make_env "sandhi-env"    "requirements.txt"
make_env "interface-env" "requirements-interface.txt"

# ── activation reminder ───────────────────────────────────────────────────────
cat <<'EOF'

────────────────────────────────────────────────────
  Environments ready. Activate with:

  Analysis / processing:
    source sandhi-env/bin/activate

  LSL interface / acquisition:
    source interface-env/bin/activate

  PsychoPy (separate — standalone installer):
    https://www.psychopy.org/download.html
────────────────────────────────────────────────────
EOF
