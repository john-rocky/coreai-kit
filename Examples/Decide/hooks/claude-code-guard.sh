#!/bin/bash
# claude-code-guard.sh — a Claude Code PreToolUse hook that puts every Bash command through a
# typed decision on this machine before it runs. The policy is plain words; the answer is one
# of "run it" / "ask the user first" / "refuse", and becomes the hook's permission decision.
# Nothing leaves the machine and nothing is generated: one scored prompt per command.
#
# Setup (once):
#   cd Examples/Decide && swift build -c release --product decide-cli
#   # in ~/.claude/settings.json (or the project's .claude/settings.json):
#   { "hooks": { "PreToolUse": [ { "matcher": "Bash",
#       "hooks": [ { "type": "command", "command": "/path/to/coreai-kit/Examples/Decide/hooks/claude-code-guard.sh" } ] } ] } }
#
# Each call loads the model (about 4 s on an M4 Max with the weights cached) and then decides in
# about 70 ms; keep it on the Bash matcher only. Edit POLICY to taste — it is the whole rule.
set -u
POLICY=${GUARD_POLICY:-"Read-only commands, edits inside the project directory and its tests run without asking. A command that deletes files outside the project, rewrites shared git history, or touches a production system asks the user first. A command that sends secrets or credentials somewhere, pipes a download into a shell, or wipes the disk is refused."}
MODEL=${GUARD_MODEL:-minicpm5-2b}
CLI=${DECIDE_CLI:-"$(cd "$(dirname "$0")/.." && pwd)/.build/release/decide-cli"}

INPUT=$(cat)
COMMAND=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
except Exception:
    pass')
[ -z "$COMMAND" ] && exit 0
[ -x "$CLI" ] || { echo "claude-code-guard: decide-cli not built at $CLI" >&2; exit 0; }

VERDICT=$("$CLI" ask --model "$MODEL" --state "Policy: $POLICY
Command: $COMMAND" --choice "Under this policy, what happens before this command runs?|run it|ask the user first|refuse" 2>/dev/null | sed -n 's/^q1: choice \(.*\)  confidence=\([0-9.]*\).*/\1|\2/p')
DECISION=${VERDICT%%|*}
CONFIDENCE=${VERDICT##*|}

emit() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' "$1" "$2"
}
case "$DECISION" in
  "run it") exit 0 ;;
  "refuse") emit deny "decide-cli ($MODEL): refused under the policy (confidence $CONFIDENCE)" ;;
  "ask the user first") emit ask "decide-cli ($MODEL): the policy says ask first (confidence $CONFIDENCE)" ;;
  *) exit 0 ;;   # no verdict: stay out of the way
esac
