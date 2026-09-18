# Hub repository layout and card template

A proposed convention for the Hugging Face org `coreai-community`. Draft, 2026-09-18. My 28 repositories
in the org will follow it; any Core AI repository can. Comments are welcome as an issue on `john-rocky/coreai-kit`.

A repository with more than one bundle should say which folder runs on which platform. The convention has
four parts: one folder per loadable unit, named for what it is; a manifest at the repository root that maps
platform to folder; a card with a variant table; one branch, with a tag per export. A repository with one
`.aimodel` at the root already fits: one unit, nothing to declare.

## 1. Repository structure

```
<Model>-CoreAI/
├── README.md          the card (section 3)
├── coreai-kit.json    the manifest: platform → variant path (section 2)
└── <variant>/         one loadable unit per folder
```

A variant is a path that holds exactly one loadable unit, in one of two forms:

- a bundle directory: `metadata.json` plus the assets it names (`*.aimodel/`, or compiled `*.aimodelc/`),
  and `tokenizer/` for language models. Apple's loader opens this directory; CoreAIKit downloads it as one unit;
- a single `.aimodel` directory, for a one-network model. Apple's loader does not open one; CoreAIKit loads it directly.

A variant sits one level down (`int8/`), or two when a grouping folder holds siblings (`gpu-pipelined/<bundle>/`).
CoreAIKit downloads nothing outside the variant path.

## 2. Variant names and the manifest

A folder name says what the bundle is, from these tokens in this order. A name carries only the tokens that
describe the bundle; the bundle's own exported name goes below it.

| Token | Values | Example |
|---|---|---|
| platform | `ios`, `macos` | `ios-ane-h18p/`: iOS only, Neural Engine, one device architecture |
| engine | `gpu-pipelined`, `ane` | `gpu-pipelined/<bundle>/`: GPU pipelined engine |
| device architecture | `h18p` (what an ahead-of-time bundle is compiled for; today the iPhone 17 Pro) | `ios-ane-h18p/` |
| quantization | `int8`, `int4` | `int8/`: GPU, quantization alone |

The name cannot be relied on for the platform: `gpu-pipelined/` holds bundles that run on macOS and iOS next
to bundles that run on macOS alone. The manifest, `coreai-kit.json` at the repository root, is where the
platform is declared. It is the block CoreAIKit's catalog holds today, one per model; reading it from the
repository instead is the next step I described on huggingface.js #2466
(https://github.com/huggingface/huggingface.js/pull/2466):

```json
{ "kind": "chat",
  "variants": { "macos": { "path": "int8" },
                "ios": { "path": "int8" },
                "ios-ane-h18p": { "path": "ios-ane-h18p" } } }
```

`kind` uses CoreAIKit's catalog vocabulary (`chat`, `vlm`, `tts`, …). Keys are a platform (`macos`, `ios`) or a
platform plus a device key (`ios-ane-<arch>`). A repository with one unit needs no manifest: the same comment
describes a fallback that reads the repository tree. Apple's loader does not read the file.

## 3. Card

```yaml
---
license: <upstream license>
base_model: <upstream/repo>
base_model_relation: quantized     # drop when the weights are unchanged
pipeline_tag: <task>               # text-generation, automatic-speech-recognition, …
library_name: coreai               # the value these repositories use today
tags: [coreai, apple, on-device]   # plus whatever the author adds
---
```

The first screen, below the front matter: a title, one sentence on the upstream model and what the repository
holds, the variant table, and a load snippet (CoreAIKit's today:
`ChatSession(model: ModelID("<org>/<Model>-CoreAI", path: "int8"))`).

| Variant | Path | Size | Requires | Tested on |
|---|---|---|---|---|
| GPU, int8 | `int8/` | <MB> | macOS 27, iOS 27 | <devices> |
| Neural Engine, ahead-of-time | `ios-ane-h18p/` | <MB> | iOS 27 | <devices> |

## 4. Mirrors and versions

A mirror carries one line at the top of the body, "Mirror of `<source>` — the canonical repo. Updates land
there first.", and keeps the source's folders and manifest. My 28 mirrors in the org carry that line today.
Their front matter predates this template; I will bring them up to it, unless the originals move into the
org first. Existing repositories keep their folder names and paths, so every catalog pin and every `path:`
in an app keeps working; the manifest and the card table are additions.

Versions live on one branch, `main`, with a Hub tag per export (`v2`, `v3`). During the beta a suggestion I
received on X was branches plus card metadata, rather than one repository per version. This page keeps one
repository per model and versions as git refs, with two changes. A tag rather than a branch: the Hub's tree
and resolve endpoints accept a tag as the revision, so an app pins a tag or a commit in `ModelID(revision:)`
with no change in the library. A branch would not survive the library's cache: CoreAIKit keys a downloaded
bundle by its revision string and checks only that it is present, so an app pinned to a moving branch keeps
the first bundle it got. That is a limit in my code, not in the Hub. And the OS a variant was tested on goes
in the variant table, where a cell carries OS, device and date, rather than in a front-matter tag. iOS 27
and macOS 27 have shipped; if a later OS changes the format, the tag and the table record it.
