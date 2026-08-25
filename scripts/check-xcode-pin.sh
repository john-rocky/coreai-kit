#!/bin/bash
# Enforce .xcode-pin: the runner is using the Xcode this repository says it is.
#
# Until 2026-08-25 every workflow checked `[ -d "$DEVELOPER_DIR" ]` and nothing
# else, which is a check that cannot go red at GA. The release Xcode installs to
# /Applications/Xcode.app and the beta stays where it is, so the directory keeps
# existing and CI keeps publishing green for an SDK nobody ships against. Three
# things are checked instead:
#
#   1. the pinned app is installed          (the 2026-08-15 failure)
#   2. its ProductBuildVersion matches      (the folder name is not evidence)
#   3. no release build of the same Xcode train is installed while the pin
#      names a seed                         (the GA tripwire)
#
# (3) is deliberately narrow. Seed-to-seed drift — beta 6 sitting on the runner
# while the pin still names beta 5 — is not an error: the pin says which SDK
# produced the green and that stays true. What must never pass unnoticed is the
# release build arriving, because from then on the pinned SDK is one no user
# has. A 27.1 seed is a different train and is likewise left alone; the pin is
# then a deliberate choice about which train to test, not drift.
#
# Overridable for the tests: XCODE_PIN_FILE, XCODE_APPS_DIR.

set -u

here=$(cd "$(dirname "$0")" && pwd)
pin_file=${XCODE_PIN_FILE:-$here/../.xcode-pin}
apps=${XCODE_APPS_DIR:-/Applications}

# Read one key out of an Xcode's version.plist. Empty if the app or key is gone.
plist_value() {
	/usr/libexec/PlistBuddy -c "Print :$2" "$1/Contents/version.plist" 2>/dev/null
}

# Split a ProductBuildVersion into "<train> <letter> <seq> <suffix>", e.g.
# 27A5237l -> "27 A 5237 l". Prints nothing for anything unparseable, and every
# caller treats that as "cannot judge" rather than as a verdict.
build_parts() {
	printf '%s\n' "$1" | sed -nE 's/^([0-9]+)([A-Z])([0-9]+)([a-z]*)$/\1 \2 \3 \4/p'
}

# Apple numbers pre-release builds from 5000 up in each train: 27A5237l is a
# seed, 27A300 is not. This is the only signal in the plist that separates them
# — CFBundleShortVersionString reads 27.0 for both.
is_seed_build() {
	set -- $(build_parts "$1")
	[ $# -eq 3 ] || [ $# -eq 4 ] || return 1
	[ "$3" -ge 5000 ]
}

# Major of CFBundleShortVersionString: 27.0 -> 27, 26.1.1 -> 26.
short_major() {
	printf '%s\n' "$1" | sed -nE 's/^([0-9]+).*/\1/p'
}

# Only run the checks when executed. Sourcing gets the functions alone, which is
# what the test does.
[ "${XCODE_PIN_SELFTEST:-0}" = "1" ] && return 0

if [ ! -f "$pin_file" ]; then
	echo "::error::$pin_file is missing. It names the Xcode CI must use."
	exit 1
fi

# shellcheck disable=SC1090
. "$pin_file"

if [ -z "${XCODE_PATH:-}" ] || [ -z "${XCODE_BUILD:-}" ]; then
	echo "::error::$pin_file must set both XCODE_PATH and XCODE_BUILD."
	exit 1
fi

if [ ! -d "$XCODE_PATH" ]; then
	echo "::error::$XCODE_PATH is not installed on this runner." \
		"Install it, or point XCODE_PATH in .xcode-pin at the one that is" \
		"(and update XCODE_BUILD to match)."
	ls -d "$apps"/Xcode*.app 2>/dev/null || true
	exit 1
fi

installed_build=$(plist_value "$XCODE_PATH" ProductBuildVersion)
if [ -z "$installed_build" ]; then
	echo "::error::$XCODE_PATH has no readable Contents/version.plist." \
		"That is not an Xcode, or the install is damaged."
	exit 1
fi

if [ "$installed_build" != "$XCODE_BUILD" ]; then
	echo "::error::$XCODE_PATH is build $installed_build, but .xcode-pin says" \
		"$XCODE_BUILD. The folder name is not evidence of which SDK is in it —" \
		"bump XCODE_BUILD if this replacement is intended."
	exit 1
fi

pinned_major=$(short_major "$(plist_value "$XCODE_PATH" CFBundleShortVersionString)")
set -- $(build_parts "$XCODE_BUILD")
pinned_train=${1:-} pinned_letter=${2:-}

# Say what else is on the runner even when everything passes. A green whose log
# names the alternatives is the cheapest way to notice a seed went stale.
for app in "$apps"/Xcode*.app; do
	[ -d "$app" ] || continue
	other_build=$(plist_value "$app" ProductBuildVersion)
	[ -n "$other_build" ] || continue
	[ "$app" = "$XCODE_PATH" ] && continue
	echo "::notice::also installed: $app ($other_build)"

	# The GA tripwire. Same Xcode train as the pin, and a release build while
	# the pin names a seed.
	is_seed_build "$XCODE_BUILD" || continue
	is_seed_build "$other_build" && continue
	set -- $(build_parts "$other_build")
	[ "${1:-}" = "$pinned_train" ] && [ "${2:-}" = "$pinned_letter" ] || continue

	echo "::error::$app is a RELEASE build ($other_build) of the same Xcode" \
		"train as the pinned seed ($XCODE_BUILD). CI would keep testing an SDK" \
		"that no shipping app is built with. Point .xcode-pin at it — or say in" \
		"a comment there why the seed is still the right target."
	exit 1
done

echo "Xcode pin: $XCODE_PATH ($installed_build, Xcode $pinned_major)"

# Hand DEVELOPER_DIR to the rest of the job. Every step after this one — swift
# build, swift test, the next-SDK probe — inherits it, so the pin is chosen in
# exactly one file.
if [ -n "${GITHUB_ENV:-}" ]; then
	echo "DEVELOPER_DIR=$XCODE_PATH" >> "$GITHUB_ENV"
fi
