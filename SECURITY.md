# Security

CoreAIKit downloads model weights over the network and runs them on device. This page states
what protects that path, what does not, and what to change if you are shipping something you
have to support.

## Reporting

Open a [security advisory](../../security/advisories/new) for anything disclosure itself would
make worse — a catalog entry resolving somewhere unexpected, a model repository that looks
tampered with, a way to make the package fetch from an unintended host. Ordinary bugs go in
[issues](../../issues). This is a community package maintained by one person: expect a reply
in days.

## The trust boundary

**Two hosts, not one.** `ModelCatalog` fetches `catalog.json` from `raw.githubusercontent.com`
at runtime; the bundles themselves come from Hugging Face. Compromise of either changes what
your app loads.

**Pins are the integrity mechanism.** Every catalog entry names an immutable Hugging Face
revision, and `CatalogEntry.modelID(path:)` carries it into every file the entry resolves,
including the sibling subtrees of multi-part models. A later push to a model repository cannot
change what a pinned consumer receives.

**The offline fallback is pinned too.** When the catalog fetch fails, `ModelCatalog.load()`
falls back to the compiled-in catalog. That fallback used to carry no revisions — and
`revision ?? "main"` meant a network failure silently downgraded a shipped app from the gated
bytes to whatever a model repository's `main` held at that moment. It now carries the same
pins, generated from `catalog.json` by `scripts/gen-builtin-pins.py`, with CI and a unit test
failing if the two drift or if any entry loses its pin.

**What is not done**, stated plainly:

- `catalog.json` is fetched over TLS but is **not signed**. Anyone who can change that file in
  the repository's default branch can change which revision your app resolves next launch.
- Downloaded bundle files carry **no independent checksum manifest** — integrity rests on the
  Hugging Face revision and on TLS, not on something you can verify offline.
- The catalog fetch has **no TTL or caching policy**; it is fetched when asked.
- `.aimodel` bundles are **not code-signed**. A bundle is data a runtime executes; treat one
  from any source the way you treat a binary dependency.

## If you are shipping a production app

The defaults optimise for "a new model appears and existing apps can offer it." A production
app usually wants the opposite, and the API supports it:

- **Host the catalog yourself.** `ModelCatalog.load(from:)` and `entry(forID:from:)` take a
  URL. Point them at a `catalog.json` you control, and the runtime dependency on this
  repository's default branch disappears.
- **Mirror the bundles** into storage you control rather than fetching a personal Hugging Face
  namespace at runtime, and verify them once on the way in.
- **Pin the package** to a tag, not a branch, and re-verify after any OS or toolchain bump —
  the `coreai-core` wheel is OS-coupled, and a beta bump has already invalidated previously
  exported bundles once (FB23666783).
- **Re-run the conversion** from the [zoo](https://github.com/john-rocky/coreai-model-zoo)'s
  recipe if the artifact's provenance has to be yours. Every recipe exists so that is
  possible, and `conversion/coreai_gate.py` re-checks the result against the original model.

## Scope

In scope: this package's download, catalog resolution, and model-loading paths. Out of scope:
model *content* — what a model says, what it was trained on, its licence — which belongs to the
upstream model and is linked from each zoo card. Compromise of a third-party contributor's
Hugging Face namespace is outside our control; report it and the affected entries will be
removed from the catalog pending resolution.
