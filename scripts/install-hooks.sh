#!/bin/sh
# Point this clone's hooks at the versioned .githooks directory.
#
# Run once per clone. core.hooksPath is a local config value, so it is not
# inherited by a fresh clone — new checkouts need this again.

set -e

cd "$(git rev-parse --show-toplevel)"
chmod +x .githooks/*
git config core.hooksPath .githooks
echo "core.hooksPath -> .githooks"
