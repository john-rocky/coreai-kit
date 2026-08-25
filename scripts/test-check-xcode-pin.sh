#!/bin/bash
# Pin the Xcode-pin check's verdicts against a fake /Applications. No Xcode, no
# network, well under a second.
#
# The case that matters is "release build installed alongside the pinned seed".
# That is the shape CI was blind to until 2026-08-25: the pinned beta stays on
# disk after GA, so a directory-existence check keeps passing forever and the
# green describes an SDK nobody ships with. A gate that cannot go red at GA is
# indistinguishable from one that is passing.

set -u

here=$(cd "$(dirname "$0")" && pwd)
check="$here/check-xcode-pin.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

# Write a fake Xcode.app whose version.plist carries the two keys we read.
fake_xcode() {
	dir=$1 short=$2 build=$3
	mkdir -p "$dir/Contents"
	cat > "$dir/Contents/version.plist" <<-PLIST
		<?xml version="1.0" encoding="UTF-8"?>
		<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
		<plist version="1.0">
		<dict>
			<key>CFBundleShortVersionString</key>
			<string>$short</string>
			<key>ProductBuildVersion</key>
			<string>$build</string>
		</dict>
		</plist>
	PLIST
}

# name | pinned path | pinned build | expected exit (0 green, 1 red)
check_case() {
	name=$1 path=$2 build=$3 want=$4
	pin="$tmp/pin"
	printf 'XCODE_PATH=%s\nXCODE_BUILD=%s\n' "$path" "$build" > "$pin"
	XCODE_PIN_FILE="$pin" XCODE_APPS_DIR="$tmp/Applications" \
		GITHUB_ENV="$tmp/github_env" "$check" > "$tmp/out" 2>&1
	got=$?
	if [ "$got" -eq "$want" ]; then
		printf '  ok    %s -> %s\n' "$name" "$([ "$want" -eq 0 ] && echo green || echo red)"
	else
		printf '  FAIL  %s: expected exit %s, got %s\n' "$name" "$want" "$got"
		sed 's/^/          /' "$tmp/out"
		fails=$((fails + 1))
	fi
}

apps="$tmp/Applications"
mkdir -p "$apps"

# The runner as it stands today: the pinned 27 seed, plus the Xcode 26 release
# that every Mac carries. A release build of a DIFFERENT major must not red.
fake_xcode "$apps/Xcode-27.0.0-Beta.5.app" 27.0 27A5237l
fake_xcode "$apps/Xcode.app" 26.1.1 17B100
check_case "pinned seed, older release alongside" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 0

# The whole point. GA installs to /Applications/Xcode.app and the beta stays.
fake_xcode "$apps/Xcode.app" 27.0 27A300
check_case "release build of the pinned train arrives" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 1

# ...and clears the moment the pin names it, without a code change.
check_case "pin moved to the release build" "$apps/Xcode.app" 27A300 0

# A newer seed on the runner is not an error: the pin still truthfully names
# which SDK produced the green. Humans move that pin deliberately.
rm -rf "$apps/Xcode.app"
fake_xcode "$apps/Xcode-27.0.0-Beta.6.app" 27.0 27A5252f
check_case "newer seed installed, pin left on the older one" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 0

# Next minor's seed is a different train — a deliberate choice of target, not
# drift, and the next-SDK gate is the reason to have one installed.
rm -rf "$apps/Xcode-27.0.0-Beta.6.app"
fake_xcode "$apps/Xcode-27.1.0-Beta.1.app" 27.1 27B5001a
check_case "next-minor seed installed" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 0

# The folder name is not evidence: someone drops a different build into the
# same path and the old check never noticed.
rm -rf "$apps/Xcode-27.1.0-Beta.1.app"
fake_xcode "$apps/Xcode-27.0.0-Beta.5.app" 27.0 27A5252f
check_case "path holds a different build than the pin" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 1

# The 2026-08-15 failure: DEVELOPER_DIR pointed at an uninstalled beta.
check_case "pinned Xcode not installed" \
	"$apps/Xcode-27.0.0-Beta.9.app" 27A5299z 1

# A path that exists but is not an Xcode.
mkdir -p "$apps/Xcode-empty.app"
check_case "pinned path is not an Xcode" "$apps/Xcode-empty.app" 27A5237l 1

# DEVELOPER_DIR must actually reach the rest of the job — a green that forgets
# to export it hands `swift build` back to whatever xcode-select points at,
# which is the fail-open this check exists to close.
rm -rf "$apps"
mkdir -p "$apps"
fake_xcode "$apps/Xcode-27.0.0-Beta.5.app" 27.0 27A5237l
: > "$tmp/github_env"
check_case "green run exports DEVELOPER_DIR" \
	"$apps/Xcode-27.0.0-Beta.5.app" 27A5237l 0
if grep -qx "DEVELOPER_DIR=$apps/Xcode-27.0.0-Beta.5.app" "$tmp/github_env"; then
	printf '  ok    DEVELOPER_DIR written to GITHUB_ENV\n'
else
	printf '  FAIL  DEVELOPER_DIR missing from GITHUB_ENV: [%s]\n' "$(cat "$tmp/github_env")"
	fails=$((fails + 1))
fi

# The pin the repository actually ships must be one this check accepts as
# well-formed — a typo in .xcode-pin should fail here, not on the runner.
repo_pin="$here/../.xcode-pin"
XCODE_PIN_SELFTEST=1 . "$check"
# shellcheck disable=SC1090
(
	. "$repo_pin"
	if [ -z "${XCODE_PATH:-}" ] || [ -z "${XCODE_BUILD:-}" ]; then
		echo "  FAIL  .xcode-pin does not set both XCODE_PATH and XCODE_BUILD"
		exit 1
	fi
	if [ -z "$(build_parts "$XCODE_BUILD")" ]; then
		echo "  FAIL  .xcode-pin XCODE_BUILD=$XCODE_BUILD is not a build version"
		exit 1
	fi
	printf '  ok    .xcode-pin parses (%s, %s)\n' "$XCODE_PATH" "$XCODE_BUILD"
) || fails=$((fails + 1))

if [ "$fails" -ne 0 ]; then
	echo "$fails Xcode pin case(s) failed"
	exit 1
fi
echo "Xcode pin: all cases pass"
