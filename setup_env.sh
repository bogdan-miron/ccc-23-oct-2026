#!/usr/bin/env bash
# Thin wrapper: ./setup_env.sh venv|torch|core|extra|verify|all  (same as python setup_env.py ...)
exec "$(command -v python3 || command -v python)" "$(cd "$(dirname "$0")" && pwd)/setup_env.py" "$@"
