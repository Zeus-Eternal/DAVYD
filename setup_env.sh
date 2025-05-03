#!/usr/bin/env bash
set -euo pipefail

# ─────── Check for Homebrew ───────
if ! command -v brew &>/dev/null; then
  echo "Homebrew not found. Installing Homebrew…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# ─────── Update Homebrew & Install Python3 ───────
echo "Updating Homebrew and installing Python 3…"
brew update
brew install python@3.11

# ─────── Install OS-level deps for Pillow (if you’re using it) ───────
echo "Installing image libraries for Pillow support…"
brew install libjpeg libpng freetype

# ─────── Create & Activate Virtualenv ───────
echo "Creating virtual environment in ./venv…"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

# ─────── Upgrade pip & Install Requirements ───────
echo "Upgrading pip and installing Python packages from requirements.txt…"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ All set! Your venv is active and all dependencies are installed."
echo "To get started:"
echo "  source venv/bin/activate"
echo "  python your_main_script.py"
