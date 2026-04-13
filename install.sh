#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <flagscale-repo-path>"
    echo "Example: $0 /path/to/FlagScale"
    exit 1
fi

TARGET="$1"

if [ ! -d "$TARGET" ]; then
    echo "Error: $TARGET does not exist"
    exit 1
fi

SKILLS_DIR="$TARGET/.claude/skills/port-model-flagscale"
mkdir -p "$SKILLS_DIR/scripts" "$SKILLS_DIR/references"
cp "$SCRIPT_DIR/skills/port-model-flagscale/SKILL.md" "$SKILLS_DIR/"
cp "$SCRIPT_DIR/skills/port-model-flagscale/scripts/"*.py "$SKILLS_DIR/scripts/"
cp "$SCRIPT_DIR/skills/port-model-flagscale/references/"*.md "$SKILLS_DIR/references/"

echo "Installed port-model-flagscale skill to $SKILLS_DIR/"
