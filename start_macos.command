#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
APP_DIR="$SCRIPT_DIR/video_dubber"
PYTHON="$APP_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "未找到虚拟环境，请先运行："
  echo "cd \"$APP_DIR\""
  echo "python3.12 -m venv .venv"
  echo ".venv/bin/python -m pip install -r requirements.txt"
  read "?按回车退出..."
  exit 1
fi

cd "$APP_DIR"
"$PYTHON" run.py

