---
name: v1-protection
description: Pre-commit safety hook that blocks any modifications to src/uavbench/ (v1 codebase). Triggers on every file write to enforce the clean-room rule. Use this to prevent accidental v1 edits.
---

# v1 Protection Hook

## Purpose
Enforces Rule A from CLAUDE.md: NEVER modify, refactor, or delete anything under `src/uavbench/`.

## Implementation
Add this as a pre-commit hook (`.claude/hooks/pre-commit.sh`):

```bash
#!/bin/bash
# Block any staged changes to src/uavbench/
if git diff --cached --name-only | grep -q "^src/uavbench/"; then
  echo "❌ BLOCKED: Attempted modification to src/uavbench/ (v1 baseline)"
  echo "   v2 code must go in src/uavbench2/ only."
  echo "   This is a non-negotiable clean-room rule."
  exit 1
fi
```

## Claude Code Hook (settings.json)
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hook": "bash -c 'echo \"$TOOL_INPUT\" | grep -q \"src/uavbench/\" && echo \"{\\\"block\\\": true, \\\"message\\\": \\\"BLOCKED: Cannot modify v1 code in src/uavbench/. Use src/uavbench2/ for v2.\\\"}\" || echo \"{\\\"block\\\": false}\"'"
      }
    ]
  }
}
```
