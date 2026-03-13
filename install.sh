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

SKILLS_DIR="$TARGET/.claude/skills"
mkdir -p "$SKILLS_DIR"
cp "$SCRIPT_DIR/skill/port-model.md" "$SKILLS_DIR/"

echo "Installed port-model skill to $SKILLS_DIR/port-model.md"
