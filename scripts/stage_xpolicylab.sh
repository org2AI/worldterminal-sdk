#!/usr/bin/env bash
# Installable policy directory for a RoboDojo or RoboTwin XPolicyLab checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
xpl=$(cd "${1:?usage: stage_xpolicylab.sh PATH_TO_XPOLICYLAB}" && pwd)
test -f "$xpl/setup_policy_server.py" && test -d "$xpl/policy" || { echo "Expected an XPolicyLab checkout" >&2; exit 1; }
target="$xpl/policy/WorldTerminal"
if [[ -e "$target" ]]; then echo "$target already exists" >&2; exit 1; fi
stage=$(mktemp -d "$xpl/policy/.worldterminal.XXXXXX")
trap 'rm -rf "$stage"' EXIT
adapter="$root/adapters/xpolicylab/WorldTerminal"
mkdir -p "$stage/WorldTerminal"
cp "$adapter"/*.py "$adapter"/*.sh "$adapter"/deploy.yml "$adapter"/README.md "$stage/WorldTerminal/"
mv "$stage/WorldTerminal" "$target"
echo "Staged $target"
