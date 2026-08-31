# GA day — the 0.4.x release on macOS/iOS 27 GA

The runbook for the morning Apple ships macOS/iOS 27 and the release Xcode. Written
ahead of time (2026-08-31) so the day costs an hour, not a search. Everything here is
mechanical; the one decision — *which* 0.4.x number — is step 6.

**The trigger is CI itself.** `scripts/check-xcode-pin.sh` has a GA tripwire: the moment
a release build of the pinned Xcode train lands in `/Applications`, every workflow goes
red with an error naming this exact situation. Do the steps below in one sitting so the
red window stays short.

## 1. OS first, then Xcode

Update the CI runner (and the dev Mac) to the release macOS **before** flipping the
Xcode pin. The CoreAI framework is an OS library: binaries built against the release SDK
on a beta OS can fail at `dlopen` when the SDK generation and OS build disagree — the
failure looks like a broken model, not a version skew. Then install the release Xcode
(it lands in `/Applications/Xcode.app`; the beta stays put).

## 2. Flip `.xcode-pin`

```bash
/usr/libexec/PlistBuddy -c 'Print :ProductBuildVersion' /Applications/Xcode.app/Contents/version.plist
```

Edit `.xcode-pin`: `XCODE_PATH=/Applications/Xcode.app`, `XCODE_BUILD=<that value>`.
The build version is the pin, not the folder name. `./scripts/check-xcode-pin.sh` must
print the release build and exit 0.

## 3. GA wording

```bash
python3 scripts/ga-wording.py --apply
```

Three docs sentences flip from "27 beta" to "27" (the README already reads GA — its
beta wording left with the 2026-08-31 quickstart rework). Then delete
`scripts/ga-wording.py` and its two CI steps (`ci.yml` and `nightly-gate.yml`, the
"GA wording" step in each) in the same commit — after `--apply`, `--check` fails by
design.

## 4. Local gate before pushing

```bash
swift build && swift test
```

plus the hybrid two-turn check from [AGENTS.md](../AGENTS.md): two `ChatSession` turns
on a hybrid bundle (Qwen3.5 / LFM2.5 / Granite 4) on the Mac — the second turn is where
an engine-visible change shows, and an SDK generation flip is exactly the kind of change
the hermetic tests cannot see. The `coreai-models` pin (`exact: "0.2.4-zoo"`) does not
move for GA.

## 5. Push and watch

One commit: `.xcode-pin`, the three docs files, the deleted script and CI steps,
CHANGELOG (step 6). CI and the nightly gate must go green on the release toolchain
before anything is tagged.

## 6. Tag and release notes

This is a patch release — API unchanged, the toolchain and wording moved — so the next
free 0.4.x:

```bash
# CHANGELOG.md: retitle "## [Unreleased]" to "## [0.4.x] — <date>" in the same push
git tag 0.4.x && git push origin 0.4.x
gh release create 0.4.x --title "0.4.x — macOS/iOS 27 GA" --notes-file <notes>
```

Release notes = that CHANGELOG section, led by one line the funnel can quote: built and
gated on release macOS/iOS 27 and release Xcode 27, no API changes, `from: "0.4.0"`
resolvers pick it up automatically.

## 7. Devices

Re-run the on-device numbers only after the phones are on release iOS 27, and say which
build produced them ([AGENTS.md](../AGENTS.md): no device numbers you did not measure).
Nothing in the tag waits for this — device re-measurement is follow-up, not gate.
