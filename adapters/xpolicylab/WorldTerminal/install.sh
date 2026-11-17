#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}"
# Until published, ship the built SDK wheel alongside this adapter.
python -m pip install "${SCRIPT_DIR}"/wheels/worldterminal_sdk-*.whl
python -c 'from worldterminal_sdk import Client; print("WorldTerminal SDK ready")'
