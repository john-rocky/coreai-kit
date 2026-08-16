#!/bin/bash
# Pin the next-SDK probe's four outcomes. Runs in well under a second, no GPU,
# no network — cheap enough to sit in CI next to the unit tests.
#
# The case that matters is "missing Xcode": on 2026-08-15 that exit code was
# read as a model verdict and flipped the public badge to red.

set -u

here=$(cd "$(dirname "$0")" && pwd)
classify="$here/classify-probe.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

# name | fixture | rc | expected stdout ("" = must print nothing and exit 1)
check() {
	name=$1 fixture=$2 rc=$3 want=$4
	printf '%b' "$fixture" > "$tmp/log"
	got=$("$classify" "$tmp/log" "$rc" 2>/dev/null) || got=""
	if [ "$got" = "$want" ]; then
		printf '  ok    %s -> %s\n' "$name" "${want:-no verdict}"
	else
		printf '  FAIL  %s: expected [%s], got [%s]\n' "$name" "$want" "$got"
		fails=$((fails + 1))
	fi
}

check "generated text" \
	'I am a language model running on device.\n' 0 "status=ready"

# Verbatim from the 2026-07-10 incident log.
check "versioned-IR rejection" \
	'error: expected AICode versioned location, got: loc(fused<...>)\nerror: Failed to convert to versioned IR\nLLVM ERROR: cannot unwrap empty `odiec_module_t`\n' \
	1 "status=not-ready"

# Only the last of the three lines survives when the process aborts early.
check "rejection, truncated" \
	'LLVM ERROR: cannot unwrap empty `odiec_module_t`\n' 1 "status=not-ready"

check "missing Xcode" \
	'xcrun: error: missing DEVELOPER_DIR path: /Applications/Xcode-27.0.0-Beta.3.app\n' \
	1 ""

check "build failure" \
	'error: no such module '"'"'CoreAI'"'"'\n' 1 ""

check "exit 0, no output" '' 0 ""

if [ "$fails" -ne 0 ]; then
	echo "$fails probe classification case(s) failed"
	exit 1
fi
echo "probe classification: all cases pass"
