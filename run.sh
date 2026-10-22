#!/usr/bin/env bash
# Thin wrapper: ./run.sh script.py | ./run.sh -c "code"  (same as python run.py ...)
exec "$(command -v python3 || command -v python)" "$(cd "$(dirname "$0")" && pwd)/run.py" "$@"
