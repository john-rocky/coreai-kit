# How the catalog is verified

[![CI](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml)
[![Nightly build + pins](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml)
[![Next-SDK models](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-kit%2Fgate-status%2Fnext-sdk.json)](https://github.com/john-rocky/coreai-kit/actions/workflows/next-sdk-gate.yml)

**0.8.0 targets Xcode 27 (`27A266a`) on macOS 27 (`26A428`), the release builds.** See the
[validation record](GETTING_STARTED.md#release-041-validation) for the tested Mac,
OS/SDK builds and model revisions of the 0.4.1 train; 0.4.2, 0.5.0, 0.6.0, 0.7.0, 0.7.1, 0.7.2, 0.7.3 and 0.8.0 re-ran the same gates on the release
toolchain (CHANGELOG). Device rows were measured on the iOS 27 RC (24A435).
On an iPhone 18 Pro with iOS 27.2 (`24B5099f`), Qwen3 0.6B answers with unreadable text.
CoreAIKit 0.7.3 and 0.8.0 both do this; the cause is not known
([Known issues](../CHANGELOG.md#known-issues)).

The models are converted, not vendored, so the question that matters before you depend on
this is *what was checked, by whom, and can you check it again.* All 80 catalog entries:

- **Pinned to an immutable Hugging Face revision**, so a resolved model is the exact bytes
  that were gated — never "whatever is on `main` today." CI re-checks every pin
  (`scripts/pin-catalog.py --check`, run by the nightly gate above).
- **Gated against the original model before enrollment** — the export is stepped against the
  fp32/fp16 reference implementation on fixed inputs (token-exact for LLMs, `cos ≥ 0.999`
  otherwise), then re-gated after compression, then run on real hardware. The gate that
  produced each row and its proof strength are on that model's card.
- **Shipped with the recipe that produced them.** `models/<model>/recipe.toml` in the
  [model zoo](https://github.com/john-rocky/coreai-model-zoo) records the exact script and
  flags; `zoo_convert.py run <name>` rebuilds the bundle from the same checkpoint.

These gates are run by the maintainer — so don't take them on faith, re-run them. Checking a
published bundle against the model it claims to come from is one command, no GPU and no
device needed:

```bash
python3 conversion/zoo_verify.py mlboydaisuke/Gemma-4-12B-CoreAI   # one repo
python3 conversion/zoo_verify.py --all                             # the whole catalog, minutes
```

It compares tokenizer, chat template, context length and declared precision against the
source model each bundle names in its own `metadata.json`.

That checks a bundle is *described* correctly, not that it still *computes* correctly — the
numerical check is `conversion/coreai_gate.py`, which rebuilds the reference model in fp32 and
compares a greedy decode token for token. It runs outside the maintainer's tree (point it at
your own `llm-runner` and overlay interpreter) and writes a transcript: pinned revision, exact
`input_ids`, both sides' tokens, verdict. Re-running the engine side against a published
transcript needs only the bundle and `llm-runner` — no oracle, no fp32 download.

If you are shipping something you have to support, re-running the recipe yourself is cheap and
leaves you owning the artifact.

## Versioning & stability

Tagged releases, SemVer, a pinned model catalog (each entry carries the verified
Hugging Face revision), and CI + a nightly end-to-end gate on macOS 27. See
[`docs/STABILITY.md`](STABILITY.md) and [`CHANGELOG.md`](../CHANGELOG.md).
