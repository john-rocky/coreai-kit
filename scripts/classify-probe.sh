#!/bin/bash
# Decide what the next-SDK probe's output means.
#
#   classify-probe.sh <logfile> <exit-code>
#
# Prints `status=ready` or `status=not-ready` on success. Exits 1, printing
# nothing, when the run says nothing about the models — the caller must then
# leave the published badge alone.
#
# This lives outside next-sdk-gate.yml so it can be tested. Folding an
# infrastructure failure into a model verdict is what published a false red on
# 2026-08-15, and nothing caught it; scripts/test-classify-probe.sh now does.

set -u

log=${1:?usage: classify-probe.sh <logfile> <exit-code>}
rc=${2:?usage: classify-probe.sh <logfile> <exit-code>}

# The versioned-IR rejection, verbatim from the 2026-07-10 incident log
# (coreai-model-zoo knowledge/coreai-torch-041-ir-incident.md). Any of the three
# lines is enough — the process aborts partway through on some bundles.
REJECTION='expected AICode versioned location|Failed to convert to versioned IR|cannot unwrap empty .?odiec_module_t'

if [ "$rc" -eq 0 ] && [ -s "$log" ]; then
	echo "probe generated text under the strict loader" >&2
	echo "status=ready"
elif grep -qE "$REJECTION" "$log"; then
	echo "probe hit the versioned-IR rejection (FB23666783)" >&2
	echo "status=not-ready"
else
	echo "probe failed (exit $rc) without the versioned-IR signature —" \
		"an infrastructure failure, not a model verdict" >&2
	exit 1
fi
